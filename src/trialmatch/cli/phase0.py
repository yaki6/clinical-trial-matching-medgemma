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
from pathlib import Path

import click
import structlog
import yaml

from trialmatch.data.hf_loader import load_annotations, load_annotations_from_file
from trialmatch.data.sampler import stratified_sample
from trialmatch.evaluation.metrics import compute_evidence_overlap, compute_metrics
from trialmatch.models.gemini import GeminiAdapter
from trialmatch.models.medgemma import MedGemmaAdapter
from trialmatch.models.schema import RunResult
from trialmatch.tracing.run_manager import RunManager
from trialmatch.validate.evaluator import evaluate_criterion

logger = structlog.get_logger()


async def run_model_benchmark(adapter, sample, budget_max: float = 5.0):
    """Run all sampled criterion pairs through one model.

    Calls the REUSABLE evaluate_criterion() for each pair,
    passing raw text extracted from the HF annotations.
    """
    results = []
    total_cost = 0.0

    for i, annotation in enumerate(sample.pairs):
        logger.info(
            "evaluating",
            pair=f"{i + 1}/{len(sample.pairs)}",
            patient=annotation.patient_id,
            trial=annotation.trial_id,
            criterion_type=annotation.criterion_type,
            model=adapter.name,
        )

        result = await evaluate_criterion(
            patient_note=annotation.note,
            criterion_text=annotation.criterion_text,
            criterion_type=annotation.criterion_type,
            adapter=adapter,
        )

        total_cost += result.model_response.estimated_cost
        if total_cost > budget_max:
            logger.warning("budget_exceeded", total_cost=total_cost, max=budget_max)
            break

        results.append(result)

    return results


async def run_phase0(config: dict, dry_run: bool = False):
    """Execute Phase 0 criterion-level benchmark."""
    data_cfg = config.get("data", {})
    fixture_path = data_cfg.get("fixture_path")

    if fixture_path:
        annotations = load_annotations_from_file(Path(fixture_path))
    else:
        annotations = load_annotations()

    logger.info("data_loaded", annotations=len(annotations))

    n_pairs = data_cfg.get("n_pairs", 20)
    seed = data_cfg.get("seed", 42)
    sample = stratified_sample(annotations, n_pairs=n_pairs, seed=seed)
    logger.info("sampled", n_pairs=len(sample.pairs))

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
    run_mgr = RunManager()

    for model_cfg in config.get("models", []):
        if model_cfg["provider"] == "huggingface":
            adapter = MedGemmaAdapter(hf_token=os.environ.get("HF_TOKEN", ""))
        elif model_cfg["provider"] == "google":
            adapter = GeminiAdapter(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        else:
            logger.error("unknown_provider", provider=model_cfg["provider"])
            continue

        logger.info("running_model", model=adapter.name)
        results = await run_model_benchmark(adapter, sample, budget_max=budget_max)

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

        run_id = run_mgr.generate_run_id(adapter.name)
        run_result = RunResult(
            run_id=run_id,
            model_name=adapter.name,
            results=results,
            metrics=metrics,
        )
        run_dir = run_mgr.save_run(run_result, config=config)

        click.echo(f"\n{'=' * 60}")
        click.echo(f"Model: {adapter.name}")
        click.echo(f"Run ID: {run_id}")
        click.echo(f"Results saved to: {run_dir}")
        click.echo(f"Accuracy: {metrics['accuracy']:.2%}")
        click.echo(f"Macro-F1: {metrics['f1_macro']:.2%}")
        click.echo(f"MET/NOT_MET F1: {metrics['f1_met_not_met']:.2%}")
        click.echo(f"Cohen's kappa: {metrics['cohens_kappa']:.3f}")
        click.echo(f"Evidence Overlap: {metrics['mean_evidence_overlap']:.2%}")
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
                {"name": "medgemma-1.5-4b", "provider": "huggingface"},
                {"name": "gemini-3-pro", "provider": "google"},
            ],
            "budget": {"max_cost_usd": 5.0},
        }

    if dry_run:
        click.echo(f"Config: {json.dumps(config, indent=2, default=str)}")

    asyncio.run(run_phase0(config, dry_run=dry_run))
