"""CLI command for Phase 0 criterion-level benchmark.

This is the BENCHMARK HARNESS — it connects:
  HF dataset -> sampler -> evaluate_criterion() -> metrics -> run artifacts

The evaluate_criterion() call is the reusable core (validate/evaluator.py).
Everything else here is benchmark-specific orchestration.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import click
import structlog
import yaml
from dotenv import load_dotenv

load_dotenv()

from collections import defaultdict

from trialmatch.data.hf_loader import load_annotations, load_annotations_from_file
from trialmatch.data.sampler import filter_by_keywords, stratified_sample
from trialmatch.evaluation.metrics import (
    TrialVerdict,
    aggregate_to_trial_verdict,
    compute_evidence_overlap,
    compute_metrics,
)
from trialmatch.models.gemini import GeminiAdapter
from trialmatch.models.medgemma import ENDPOINT_URL as _MEDGEMMA_DEFAULT_URL, MedGemmaAdapter
from trialmatch.models.schema import RunResult
from trialmatch.tracing.run_manager import RunManager
from trialmatch.validate.evaluator import evaluate_criterion

logger = structlog.get_logger()


async def run_model_benchmark(
    adapter,
    sample,
    budget_max: float = 5.0,
    max_concurrent: int = 1,
    timeout_seconds: float = 300.0,
) -> list:
    """Run all sampled criterion pairs through one model.

    Calls the REUSABLE evaluate_criterion() for each pair,
    passing raw text extracted from the HF annotations.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []
    total_cost = 0.0
    budget_exceeded = False

    async def _evaluate_one(i: int, annotation):
        nonlocal total_cost, budget_exceeded
        if budget_exceeded:
            return None

        async with semaphore:
            if budget_exceeded:
                return None

            logger.info(
                "evaluating",
                pair=f"{i + 1}/{len(sample.pairs)}",
                patient=annotation.patient_id,
                trial=annotation.trial_id,
                criterion_type=annotation.criterion_type,
                model=adapter.name,
            )

            try:
                result = await evaluate_criterion(
                    patient_note=annotation.note,
                    criterion_text=annotation.criterion_text,
                    criterion_type=annotation.criterion_type,
                    adapter=adapter,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                logger.error(
                    "pair_error",
                    pair=f"{i + 1}/{len(sample.pairs)}",
                    model=adapter.name,
                    error=str(exc)[:200],
                )
                from trialmatch.models.schema import CriterionResult, CriterionVerdict, ModelResponse
                result = CriterionResult(
                    verdict=CriterionVerdict.UNKNOWN,
                    reasoning=f"Error: {str(exc)[:200]}",
                    evidence_sentences=[],
                    model_response=ModelResponse(
                        text=f"ERROR: {str(exc)[:200]}",
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=0.0,
                        estimated_cost=0.0,
                        token_count_estimated=True,
                    ),
                )

            total_cost += result.model_response.estimated_cost

            logger.info(
                "pair_completed",
                pair=f"{i + 1}/{len(sample.pairs)}",
                model=adapter.name,
                verdict=result.verdict.value,
                correct=(result.verdict == annotation.expert_label),
                latency_ms=round(result.model_response.latency_ms),
                cost_usd=round(result.model_response.estimated_cost, 4),
                cumulative_cost_usd=round(total_cost, 4),
            )

            if total_cost > budget_max:
                logger.warning("budget_exceeded", total_cost=total_cost, max=budget_max)
                budget_exceeded = True

            return result

    if max_concurrent == 1:
        # Sequential path: simple, deterministic ordering, clear logging
        for i, annotation in enumerate(sample.pairs):
            result = await _evaluate_one(i, annotation)
            if result is None:
                break
            results.append(result)
    else:
        # Concurrent path for Tier A scale
        tasks = [_evaluate_one(i, a) for i, a in enumerate(sample.pairs)]
        raw_results = await asyncio.gather(*tasks)
        results = [r for r in raw_results if r is not None]

    return results


async def run_phase0(config: dict, dry_run: bool = False):
    """Execute Phase 0 criterion-level benchmark."""
    data_cfg = config.get("data", {})
    fixture_path = data_cfg.get("fixture_path")

    # --- STAGE: INGEST ---
    logger.info("stage_ingest_start")
    if fixture_path:
        annotations = load_annotations_from_file(Path(fixture_path))
    else:
        annotations = load_annotations()
    logger.info("stage_ingest_complete", annotations=len(annotations))

    # --- STAGE: FILTER (optional keyword filter) ---
    keyword_filter = data_cfg.get("keyword_filter", [])
    if keyword_filter:
        annotations = filter_by_keywords(annotations, keyword_filter)
        logger.info("stage_filter_applied", keywords=keyword_filter, remaining=len(annotations))
        if len(annotations) == 0:
            logger.error("no_annotations_after_filter", keywords=keyword_filter)
            click.echo(f"ERROR: No annotations match keywords {keyword_filter}", err=True)
            return

    # --- STAGE: SAMPLE ---
    logger.info("stage_sample_start", total_annotations=len(annotations))
    n_pairs = data_cfg.get("n_pairs", 20)
    seed = data_cfg.get("seed", 42)
    sample = stratified_sample(annotations, n_pairs=n_pairs, seed=seed)
    logger.info("stage_sample_complete", n_pairs=len(sample.pairs))

    if dry_run:
        click.echo(f"Dry run: would evaluate {len(sample.pairs)} pairs with models")
        for i, a in enumerate(sample.pairs):
            click.echo(
                f"  {i + 1}. [{a.criterion_type}] Patient {a.patient_id} x "
                f'Trial {a.trial_id}: "{a.criterion_text[:60]}" '
                f"(expert={a.expert_label.value})"
            )
        return

    budget_max = config.get("budget", {}).get("max_cost_usd", 5.0)
    budget_warn = config.get("budget", {}).get("warn_at_usd", budget_max * 0.6)
    timeout_seconds = config.get("evaluation", {}).get("timeout_seconds", 300.0)
    run_mgr = RunManager()

    for model_cfg in config.get("models", []):
        if model_cfg["provider"] == "huggingface":
            hf_token = os.environ.get("HF_TOKEN", "")
            if not hf_token:
                click.echo("ERROR: HF_TOKEN env var required for MedGemma. Skipping.", err=True)
                continue
            adapter = MedGemmaAdapter(
                hf_token=hf_token,
                endpoint_url=model_cfg.get("endpoint_url", _MEDGEMMA_DEFAULT_URL),
                model_name=model_cfg["name"],
                use_chat_api=model_cfg.get("use_chat_api", False),
            )
        elif model_cfg["provider"] == "google":
            api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not api_key:
                click.echo("ERROR: GOOGLE_API_KEY env var required for Gemini. Skipping.", err=True)
                continue
            adapter = GeminiAdapter(api_key=api_key)
        elif model_cfg["provider"] == "vertex":
            from trialmatch.models.vertex_medgemma import VertexMedGemmaAdapter

            project_id = model_cfg.get("project_id") or os.environ.get("GCP_PROJECT_ID", "")
            region = model_cfg.get("region") or os.environ.get("GCP_REGION", "us-central1")
            endpoint_id = model_cfg.get("endpoint_id") or os.environ.get(
                "VERTEX_ENDPOINT_ID", ""
            )
            if not project_id or not endpoint_id:
                click.echo(
                    "ERROR: project_id and endpoint_id required for Vertex AI. "
                    "Set GCP_PROJECT_ID and VERTEX_ENDPOINT_ID env vars or in config. Skipping.",
                    err=True,
                )
                continue
            adapter = VertexMedGemmaAdapter(
                project_id=project_id,
                region=region,
                endpoint_id=endpoint_id,
                model_name=model_cfg["name"],
            )
        else:
            logger.error("unknown_provider", provider=model_cfg["provider"])
            continue

        max_concurrent = model_cfg.get("max_concurrent", 1)
        logger.info(
            "stage_eval_start",
            model=adapter.name,
            pairs=len(sample.pairs),
            max_concurrent=max_concurrent,
            timeout_seconds=timeout_seconds,
            budget_max_usd=budget_max,
        )

        eval_start = time.perf_counter()
        results = await run_model_benchmark(
            adapter,
            sample,
            budget_max=budget_max,
            max_concurrent=max_concurrent,
            timeout_seconds=timeout_seconds,
        )
        eval_elapsed = time.perf_counter() - eval_start
        total_run_cost = sum(r.model_response.estimated_cost for r in results)

        logger.info(
            "stage_eval_complete",
            model=adapter.name,
            completed_pairs=len(results),
            total_pairs=len(sample.pairs),
            total_cost_usd=round(total_run_cost, 4),
            elapsed_sec=round(eval_elapsed, 1),
        )

        if total_run_cost >= budget_warn:
            logger.warning(
                "budget_warning",
                model=adapter.name,
                spent_usd=round(total_run_cost, 4),
                warn_at_usd=budget_warn,
                budget_max_usd=budget_max,
            )

        # --- STAGE: METRICS ---
        logger.info("stage_metrics_start", model=adapter.name, pairs=len(results))
        predicted = [r.verdict for r in results]
        actual = [a.expert_label for a in sample.pairs[: len(results)]]
        metrics = compute_metrics(predicted, actual)

        overlaps = [
            compute_evidence_overlap(r.evidence_sentences, a.expert_sentences)
            for r, a in zip(results, sample.pairs, strict=False)
        ]
        metrics["mean_evidence_overlap"] = sum(overlaps) / len(overlaps) if overlaps else 0.0

        gpt4_labels = [a.gpt4_label for a in sample.pairs[: len(results)]]
        gpt4_metrics = compute_metrics(gpt4_labels, actual)
        metrics["gpt4_baseline_accuracy"] = gpt4_metrics["accuracy"]
        metrics["gpt4_baseline_f1_macro"] = gpt4_metrics["f1_macro"]

        # --- Trial-level aggregation ---
        trial_groups: dict[tuple, list] = defaultdict(list)
        for result, annotation in zip(results, sample.pairs[: len(results)], strict=False):
            key = (annotation.patient_id, annotation.trial_id)
            trial_groups[key].append((result.verdict, annotation.criterion_type))

        trial_verdicts = {k: aggregate_to_trial_verdict(v) for k, v in trial_groups.items()}
        n_eligible = sum(1 for v in trial_verdicts.values() if v == TrialVerdict.ELIGIBLE)
        n_excluded = sum(1 for v in trial_verdicts.values() if v == TrialVerdict.EXCLUDED)
        n_not_relevant = sum(1 for v in trial_verdicts.values() if v == TrialVerdict.NOT_RELEVANT)
        n_uncertain = sum(1 for v in trial_verdicts.values() if v == TrialVerdict.UNCERTAIN)

        metrics["trial_level"] = {
            "n_unique_trials": len(trial_verdicts),
            "ELIGIBLE": n_eligible,
            "EXCLUDED": n_excluded,
            "NOT_RELEVANT": n_not_relevant,
            "UNCERTAIN": n_uncertain,
        }

        logger.info(
            "stage_metrics_complete",
            model=adapter.name,
            accuracy=round(metrics["accuracy"], 3),
            f1_macro=round(metrics["f1_macro"], 3),
            cohens_kappa=round(metrics["cohens_kappa"], 3),
            gpt4_accuracy=round(metrics["gpt4_baseline_accuracy"], 3),
            trial_level_n_unique=len(trial_verdicts),
        )

        # --- STAGE: SAVE ---
        run_id = run_mgr.generate_run_id(adapter.name)
        run_result = RunResult(
            run_id=run_id,
            model_name=adapter.name,
            results=results,
            metrics=metrics,
        )
        logger.info("stage_save_start", run_id=run_id, model=adapter.name)
        run_dir = run_mgr.save_run(
            run_result,
            config=config,
            annotations=sample.pairs[: len(results)],
        )
        logger.info("stage_save_complete", run_id=run_id, run_dir=str(run_dir))

        click.echo(f"\n{'=' * 60}")
        click.echo(f"Model: {adapter.name}")
        click.echo(f"Run ID: {run_id}")
        click.echo(f"Results saved to: {run_dir}")
        click.echo(f"Accuracy: {metrics['accuracy']:.2%}")
        click.echo(f"Macro-F1: {metrics['f1_macro']:.2%}")
        click.echo(f"MET/NOT_MET F1: {metrics['f1_met_not_met']:.2%}")
        click.echo(f"Cohen's kappa: {metrics['cohens_kappa']:.3f}")
        click.echo(f"Evidence Overlap: {metrics['mean_evidence_overlap']:.2%}")
        tl = metrics["trial_level"]
        click.echo(
            f"Trial-level (partial coverage, {tl['n_unique_trials']} unique trials):"
        )
        click.echo(
            f"  ELIGIBLE={tl['ELIGIBLE']} EXCLUDED={tl['EXCLUDED']} "
            f"NOT_RELEVANT={tl['NOT_RELEVANT']} UNCERTAIN={tl['UNCERTAIN']}"
        )
        click.echo("--- GPT-4 Baseline ---")
        click.echo(f"GPT-4 Accuracy: {metrics['gpt4_baseline_accuracy']:.2%}")
        click.echo(f"GPT-4 Macro-F1: {metrics['gpt4_baseline_f1_macro']:.2%}")
        click.echo(f"{'=' * 60}")


@click.command("phase0")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
@click.option("--dry-run", is_flag=True, help="Show sampled pairs without calling models")
def phase0_cmd(config_path: str | None, dry_run: bool):
    """Run Phase 0 benchmark: 20-pair criterion-level MedGemma vs Gemini comparison."""
    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "data": {"n_pairs": 20, "seed": 42},
            "models": [
                {"name": "medgemma-1.5-4b", "provider": "huggingface", "max_concurrent": 1},
                {"name": "gemini-3-pro", "provider": "google", "max_concurrent": 1},
            ],
            "budget": {"max_cost_usd": 5.0},
            "evaluation": {"timeout_seconds": 300},
        }

    if dry_run:
        click.echo(f"Config: {json.dumps(config, indent=2, default=str)}")

    asyncio.run(run_phase0(config, dry_run=dry_run))
