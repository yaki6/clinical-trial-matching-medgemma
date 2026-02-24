#!/usr/bin/env python3
"""MedPix Multimodal Benchmark Runner.

Compares MedGemma 4B (HF Inference, multimodal) vs Gemini 3 Pro (AI Studio)
on radiology diagnosis prediction and imaging findings extraction.

Usage:
    uv run python scripts/run_medpix_benchmark.py --config configs/medpix_bench.yaml
    uv run python scripts/run_medpix_benchmark.py --config configs/medpix_bench.yaml --dry-run
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import structlog
import yaml
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trialmatch.evaluation.multimodal_metrics import (
    compute_aggregate_metrics,
    parse_model_response,
    score_diagnosis_exact,
    score_diagnosis_substring,
    score_findings_rouge,
)
from trialmatch.models.gemini import GeminiAdapter
from trialmatch.models.medgemma import MedGemmaAdapter
from trialmatch.models.vertex_medgemma import VertexMedGemmaAdapter

logger = structlog.get_logger()

PROMPT_TEMPLATE = """You are a board-certified radiologist with expertise in diagnostic imaging.

CLINICAL HISTORY:
{history}

TASK:
Analyze the provided medical image in the context of the clinical history above.

Provide your analysis in the following format:

DIAGNOSIS: [Your primary diagnosis based on the imaging findings and clinical history]

FINDINGS: [Detailed description of the imaging findings, including:
- Location and laterality of abnormalities
- Size and morphology of lesions
- Associated findings (effusions, lymphadenopathy, etc.)
- Normal structures and their appearance
- Comparison with expected normal anatomy]

DIFFERENTIAL: [Top 2-3 differential diagnoses if the primary diagnosis is uncertain]"""

# Simplified prompt for MedGemma 4B — optimized for small model instruction-following
PROMPT_TEMPLATE_SIMPLE = """Clinical history: {history}

Based on the image and clinical history, provide:

DIAGNOSIS: [single primary diagnosis]

FINDINGS: [key imaging findings]"""

# Findings-only prompt — detailed imaging description, no diagnosis
PROMPT_TEMPLATE_FINDINGS_ONLY = """Clinical history: {history}

Analyze the provided medical image. Describe ALL imaging findings in detail:

FINDINGS:
- Location and laterality of abnormalities
- Size, shape, and density/signal characteristics of lesions
- Associated findings (effusions, lymphadenopathy, calcifications)
- Normal structures and their appearance
- Any incidental findings

Be systematic and thorough. Do NOT provide a diagnosis."""

# Diagnosis-only prompt — focused on primary diagnosis
PROMPT_TEMPLATE_DIAGNOSIS_ONLY = """Clinical history: {history}

Based on the image and clinical history, what is the single most likely diagnosis?

DIAGNOSIS: [your primary diagnosis — use standard medical terminology]

DIFFERENTIAL: [top 2-3 alternatives if uncertain]"""

JUDGE_PROMPT_TEMPLATE = """You are an expert medical evaluation judge.

Compare the MODEL PREDICTION against the GOLD STANDARD diagnosis.

GOLD STANDARD: {gold_diagnosis}
MODEL PREDICTION: {predicted_diagnosis}

Score the prediction:
- "correct": The prediction identifies the same disease/condition as the gold standard (synonyms and abbreviations are acceptable)
- "partial": The prediction identifies a related condition or a broader/narrower diagnosis that overlaps significantly
- "incorrect": The prediction identifies a different disease/condition

Respond ONLY with valid JSON:
{{"score": "correct", "explanation": "brief reason"}}"""

FINDINGS_JUDGE_PROMPT_TEMPLATE = """You are a board-certified radiologist evaluating imaging findings quality.

Compare the MODEL FINDINGS against the GOLD STANDARD findings for this case.

CLINICAL HISTORY: {history}
GOLD STANDARD FINDINGS: {gold_findings}
MODEL FINDINGS: {predicted_findings}

Evaluate the model's findings on these dimensions:
1. **Clinically significant findings identified**: Did the model identify the key abnormalities?
2. **Anatomical accuracy**: Are locations, laterality, and descriptions correct?
3. **False positives**: Did the model hallucinate findings not present in the image?
4. **Completeness**: Did the model miss important findings?

Score the findings:
- "good": Identifies the main abnormality correctly with reasonable anatomical accuracy (minor omissions acceptable)
- "partial": Identifies some relevant findings but misses the primary abnormality OR has significant anatomical errors
- "poor": Fails to identify the primary abnormality, OR describes the image as normal when it is not, OR has major hallucinated findings

Respond ONLY with valid JSON:
{{"score": "good", "key_finding_identified": true, "explanation": "brief clinical reason"}}"""


async def run_single_case(
    case: dict,
    medgemma,
    gemini: GeminiAdapter,
    medgemma_prompt: str,
    gemini_prompt: str,
    case_idx: int,
    total: int,
    medgemma_max_tokens: int = 512,
) -> dict:
    """Run both models on a single case and return raw responses."""
    image_path = Path(case["image_path"])
    if not image_path.exists():
        logger.error("image_not_found", path=str(image_path), uid=case["uid"])
        return {"uid": case["uid"], "error": f"Image not found: {image_path}"}

    logger.info(
        "processing_case",
        case=f"{case_idx + 1}/{total}",
        uid=case["uid"],
        image=str(image_path),
    )

    # Run both models — sequential to respect concurrency limits
    medgemma_response = None
    gemini_response = None

    # MedGemma 4B (multimodal)
    try:
        logger.info("calling_medgemma", uid=case["uid"])
        # NOTE: system_message disabled — vLLM container may not support content block format
        # Re-enable after confirming container compatibility
        medgemma_response = await medgemma.generate_with_image(
            prompt=medgemma_prompt, image_path=image_path, max_tokens=medgemma_max_tokens
        )
        logger.info(
            "medgemma_done",
            uid=case["uid"],
            latency_ms=round(medgemma_response.latency_ms),
            output_tokens=medgemma_response.output_tokens,
            text_preview=medgemma_response.text[:120],
        )
    except Exception as e:
        logger.error("medgemma_failed", uid=case["uid"], error=str(e)[:200])

    # Gemini Flash (AI Studio — multimodal)
    try:
        logger.info("calling_gemini", uid=case["uid"])
        gemini_response = await gemini.generate_with_image(
            prompt=gemini_prompt, image_path=image_path, max_tokens=2048
        )
        logger.info(
            "gemini_done",
            uid=case["uid"],
            latency_ms=round(gemini_response.latency_ms),
            output_tokens=gemini_response.output_tokens,
            text_preview=gemini_response.text[:120],
        )
    except Exception as e:
        logger.error("gemini_failed", uid=case["uid"], error=str(e)[:200])

    # Image metadata for trace completeness
    img_size = image_path.stat().st_size
    try:
        from PIL import Image as _Image
        with _Image.open(image_path) as img:
            img_meta = {"width": img.width, "height": img.height, "mode": img.mode}
    except Exception:
        img_meta = {}

    return {
        "uid": case["uid"],
        "title": case.get("title", ""),
        "gold_diagnosis": case["gold_diagnosis"],
        "gold_findings": case["gold_findings"],
        "location_category": case.get("location_category", ""),
        "image_path": str(image_path),
        "image_size_bytes": img_size,
        "image_meta": img_meta,
        "medgemma_prompt": medgemma_prompt,
        "gemini_prompt": gemini_prompt,
        "medgemma_raw": medgemma_response.text if medgemma_response else "",
        "medgemma_latency_ms": medgemma_response.latency_ms if medgemma_response else 0,
        "medgemma_cost": medgemma_response.estimated_cost if medgemma_response else 0,
        "medgemma_input_tokens": medgemma_response.input_tokens if medgemma_response else 0,
        "medgemma_output_tokens": medgemma_response.output_tokens if medgemma_response else 0,
        "gemini_raw": gemini_response.text if gemini_response else "",
        "gemini_latency_ms": gemini_response.latency_ms if gemini_response else 0,
        "gemini_cost": gemini_response.estimated_cost if gemini_response else 0,
        "gemini_input_tokens": gemini_response.input_tokens if gemini_response else 0,
        "gemini_output_tokens": gemini_response.output_tokens if gemini_response else 0,
    }


async def judge_diagnosis(
    judge: GeminiAdapter, gold: str, predicted: str
) -> dict:
    """Use LLM-as-judge to score diagnosis semantic equivalence."""
    if not predicted.strip():
        return {"score": "incorrect", "explanation": "Empty prediction"}

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        gold_diagnosis=gold, predicted_diagnosis=predicted
    )
    try:
        response = await judge.generate(prompt=prompt, max_tokens=256)
        result = json.loads(response.text)
        return result
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("judge_parse_failed", error=str(e)[:120], raw=response.text[:200] if 'response' in dir() else "")
        # Fallback: substring check
        if gold.lower() in predicted.lower() or predicted.lower() in gold.lower():
            return {"score": "partial", "explanation": "Fallback: substring match"}
        return {"score": "incorrect", "explanation": f"Judge failed: {e}"}


async def judge_findings(
    judge: GeminiAdapter, gold: str, predicted: str, history: str = ""
) -> dict:
    """Use LLM-as-judge to score imaging findings clinical quality."""
    if not predicted.strip():
        return {"score": "poor", "key_finding_identified": False, "explanation": "Empty prediction"}

    prompt = FINDINGS_JUDGE_PROMPT_TEMPLATE.format(
        history=history, gold_findings=gold, predicted_findings=predicted
    )
    try:
        response = await judge.generate(prompt=prompt, max_tokens=256)
        result = json.loads(response.text)
        return result
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("findings_judge_parse_failed", error=str(e)[:120])
        return {"score": "poor", "key_finding_identified": False, "explanation": f"Judge failed: {e}"}


async def score_case(
    case_result: dict, judge: GeminiAdapter, mode: str = "combined"
) -> dict:
    """Parse responses, compute metrics for one case based on mode."""
    # Parse model responses
    medgemma_parsed = parse_model_response(case_result["medgemma_raw"])
    gemini_parsed = parse_model_response(case_result["gemini_raw"])

    gold_diag = case_result["gold_diagnosis"]
    gold_find = case_result["gold_findings"]

    # For findings_only mode: use the entire raw response as findings if parsing finds nothing
    if mode == "findings_only":
        if not medgemma_parsed["findings"].strip():
            medgemma_parsed["findings"] = case_result["medgemma_raw"]
        if not gemini_parsed["findings"].strip():
            gemini_parsed["findings"] = case_result["gemini_raw"]

    # For diagnosis_only mode: use the entire raw response as diagnosis if parsing finds nothing
    if mode == "diagnosis_only":
        if not medgemma_parsed["diagnosis"].strip():
            medgemma_parsed["diagnosis"] = case_result["medgemma_raw"].split("\n")[0].strip()
        if not gemini_parsed["diagnosis"].strip():
            gemini_parsed["diagnosis"] = case_result["gemini_raw"].split("\n")[0].strip()

    history = case_result.get("medgemma_prompt", "")  # for findings judge context

    # Score MedGemma — skip irrelevant metrics per mode
    mg_exact = score_diagnosis_exact(gold_diag, medgemma_parsed["diagnosis"]) if mode != "findings_only" else False
    mg_substr = score_diagnosis_substring(gold_diag, medgemma_parsed["diagnosis"]) if mode != "findings_only" else False
    mg_rouge = score_findings_rouge(gold_find, medgemma_parsed["findings"]) if mode != "diagnosis_only" else {"recall": 0, "precision": 0, "fmeasure": 0}
    mg_judge = await judge_diagnosis(judge, gold_diag, medgemma_parsed["diagnosis"]) if mode != "findings_only" else {"score": "skipped", "explanation": "findings_only mode"}
    mg_findings_judge = await judge_findings(judge, gold_find, medgemma_parsed["findings"], history) if mode != "diagnosis_only" else {"score": "skipped", "key_finding_identified": False, "explanation": "diagnosis_only mode"}

    # Score Gemini
    gm_exact = score_diagnosis_exact(gold_diag, gemini_parsed["diagnosis"]) if mode != "findings_only" else False
    gm_substr = score_diagnosis_substring(gold_diag, gemini_parsed["diagnosis"]) if mode != "findings_only" else False
    gm_rouge = score_findings_rouge(gold_find, gemini_parsed["findings"]) if mode != "diagnosis_only" else {"recall": 0, "precision": 0, "fmeasure": 0}
    gm_judge = await judge_diagnosis(judge, gold_diag, gemini_parsed["diagnosis"]) if mode != "findings_only" else {"score": "skipped", "explanation": "findings_only mode"}
    gm_findings_judge = await judge_findings(judge, gold_find, gemini_parsed["findings"], history) if mode != "diagnosis_only" else {"score": "skipped", "key_finding_identified": False, "explanation": "diagnosis_only mode"}

    return {
        "uid": case_result["uid"],
        "title": case_result["title"],
        "gold_diagnosis": gold_diag,
        "gold_findings": gold_find,
        "location_category": case_result.get("location_category", ""),
        "mode": mode,
        "medgemma": {
            "predicted_diagnosis": medgemma_parsed["diagnosis"],
            "predicted_findings": medgemma_parsed["findings"][:1000],
            "exact_match": mg_exact,
            "substring_match": mg_substr,
            "llm_judge_score": mg_judge.get("score", "incorrect"),
            "llm_judge_explanation": mg_judge.get("explanation", ""),
            "findings_judge_score": mg_findings_judge.get("score", "poor"),
            "findings_judge_key_finding": mg_findings_judge.get("key_finding_identified", False),
            "findings_judge_explanation": mg_findings_judge.get("explanation", ""),
            "rouge_recall": mg_rouge["recall"],
            "rouge_precision": mg_rouge["precision"],
            "rouge_fmeasure": mg_rouge["fmeasure"],
            "latency_ms": case_result["medgemma_latency_ms"],
            "cost": case_result["medgemma_cost"],
        },
        "gemini": {
            "predicted_diagnosis": gemini_parsed["diagnosis"],
            "predicted_findings": gemini_parsed["findings"][:1000],
            "exact_match": gm_exact,
            "substring_match": gm_substr,
            "llm_judge_score": gm_judge.get("score", "incorrect"),
            "llm_judge_explanation": gm_judge.get("explanation", ""),
            "findings_judge_score": gm_findings_judge.get("score", "poor"),
            "findings_judge_key_finding": gm_findings_judge.get("key_finding_identified", False),
            "findings_judge_explanation": gm_findings_judge.get("explanation", ""),
            "rouge_recall": gm_rouge["recall"],
            "rouge_precision": gm_rouge["precision"],
            "rouge_fmeasure": gm_rouge["fmeasure"],
            "latency_ms": case_result["gemini_latency_ms"],
            "cost": case_result["gemini_cost"],
        },
    }


def build_summary_table(scored_results: list[dict], mode: str = "combined") -> dict:
    """Compute aggregate metrics for both models based on mode."""
    mg_results = []
    gm_results = []
    for r in scored_results:
        mg_results.append(r["medgemma"])
        gm_results.append(r["gemini"])

    def agg(results: list[dict]) -> dict:
        n = len(results)
        if n == 0:
            return {}
        summary = {
            "n_cases": n,
            "avg_latency_ms": sum(r["latency_ms"] for r in results) / n,
            "total_cost_usd": sum(r["cost"] for r in results),
        }
        if mode != "findings_only":
            summary.update({
                "diagnosis_exact_match": sum(1 for r in results if r["exact_match"]) / n,
                "diagnosis_substring_match": sum(1 for r in results if r["substring_match"]) / n,
                "diagnosis_llm_judge_correct": sum(1 for r in results if r["llm_judge_score"] == "correct") / n,
                "diagnosis_llm_judge_partial": sum(1 for r in results if r["llm_judge_score"] == "partial") / n,
            })
        if mode != "diagnosis_only":
            summary.update({
                "findings_rouge_recall_mean": sum(r["rouge_recall"] for r in results) / n,
                "findings_rouge_precision_mean": sum(r["rouge_precision"] for r in results) / n,
                "findings_rouge_fmeasure_mean": sum(r["rouge_fmeasure"] for r in results) / n,
                "findings_judge_good": sum(1 for r in results if r.get("findings_judge_score") == "good") / n,
                "findings_judge_partial": sum(1 for r in results if r.get("findings_judge_score") == "partial") / n,
                "findings_judge_poor": sum(1 for r in results if r.get("findings_judge_score") == "poor") / n,
                "findings_key_finding_rate": sum(1 for r in results if r.get("findings_judge_key_finding")) / n,
            })
        return summary

    return {
        "mode": mode,
        "medgemma_4b": agg(mg_results),
        "gemini_pro": agg(gm_results),
    }


def print_comparison_table(summary: dict) -> None:
    """Print formatted comparison table."""
    mg = summary["medgemma_4b"]
    gm = summary["gemini_pro"]
    mode = summary.get("mode", "combined")

    mode_label = {"combined": "Combined", "findings_only": "Findings Only", "diagnosis_only": "Diagnosis Only"}.get(mode, mode)

    print("\n" + "=" * 80)
    print(f"  MedPix Multimodal Benchmark — {mode_label} Results")
    print("=" * 80)
    print(f"{'Metric':<40} {'MedGemma 4B':>15} {'Gemini Pro':>15}")
    print("-" * 80)

    rows = []
    if mode != "findings_only":
        rows.extend([
            ("Diagnosis — Exact Match", f"{mg.get('diagnosis_exact_match', 0):.0%}", f"{gm.get('diagnosis_exact_match', 0):.0%}"),
            ("Diagnosis — Substring Match", f"{mg.get('diagnosis_substring_match', 0):.0%}", f"{gm.get('diagnosis_substring_match', 0):.0%}"),
            ("Diagnosis — LLM Judge (correct)", f"{mg.get('diagnosis_llm_judge_correct', 0):.0%}", f"{gm.get('diagnosis_llm_judge_correct', 0):.0%}"),
            ("Diagnosis — LLM Judge (partial)", f"{mg.get('diagnosis_llm_judge_partial', 0):.0%}", f"{gm.get('diagnosis_llm_judge_partial', 0):.0%}"),
        ])
    if mode != "diagnosis_only":
        rows.extend([
            ("Findings — LLM Judge (good)", f"{mg.get('findings_judge_good', 0):.0%}", f"{gm.get('findings_judge_good', 0):.0%}"),
            ("Findings — LLM Judge (partial)", f"{mg.get('findings_judge_partial', 0):.0%}", f"{gm.get('findings_judge_partial', 0):.0%}"),
            ("Findings — Key Finding Identified", f"{mg.get('findings_key_finding_rate', 0):.0%}", f"{gm.get('findings_key_finding_rate', 0):.0%}"),
            ("Findings — ROUGE-L Recall", f"{mg.get('findings_rouge_recall_mean', 0):.3f}", f"{gm.get('findings_rouge_recall_mean', 0):.3f}"),
            ("Findings — ROUGE-L Precision", f"{mg.get('findings_rouge_precision_mean', 0):.3f}", f"{gm.get('findings_rouge_precision_mean', 0):.3f}"),
            ("Findings — ROUGE-L F1", f"{mg.get('findings_rouge_fmeasure_mean', 0):.3f}", f"{gm.get('findings_rouge_fmeasure_mean', 0):.3f}"),
        ])
    rows.extend([
        ("Avg Latency (ms)", f"{mg.get('avg_latency_ms', 0):.0f}", f"{gm.get('avg_latency_ms', 0):.0f}"),
        ("Total Cost ($)", f"${mg.get('total_cost_usd', 0):.4f}", f"${gm.get('total_cost_usd', 0):.4f}"),
    ])

    for label, mg_val, gm_val in rows:
        print(f"{label:<40} {mg_val:>15} {gm_val:>15}")

    print("=" * 80)


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MedPix Multimodal Benchmark")
    parser.add_argument("--config", default="configs/medpix_bench.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without calling models")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load benchmark cases
    benchmark_file = Path(config["data"]["benchmark_file"])
    if not benchmark_file.exists():
        print(f"ERROR: Benchmark file not found: {benchmark_file}")
        print("Run: uv run python scripts/build_medpix_benchmark.py")
        sys.exit(1)

    with open(benchmark_file) as f:
        cases = json.load(f)

    logger.info("benchmark_loaded", n_cases=len(cases), config=args.config)

    # Verify all images exist
    missing = [c["uid"] for c in cases if not Path(c["image_path"]).exists()]
    if missing:
        print(f"ERROR: {len(missing)} images not found: {missing}")
        sys.exit(1)

    if args.dry_run:
        print(f"DRY RUN: {len(cases)} cases loaded, all images verified. Exiting.")
        return

    # Initialize models
    hf_token = os.environ.get("HF_TOKEN", "")
    api_key = os.environ.get("GOOGLE_API_KEY", "")

    medgemma_cfg = config["models"]["medgemma_4b"]
    provider = medgemma_cfg.get("provider", "huggingface")

    if provider == "vertex":
        # Try to get dedicated DNS from config, env, or gcloud
        dedicated_dns = medgemma_cfg.get("dedicated_endpoint_dns") or os.environ.get("VERTEX_DEDICATED_DNS_4B") or ""
        if not dedicated_dns:
            # Auto-discover dedicated DNS from endpoint
            try:
                import subprocess
                result = subprocess.run(
                    ["gcloud", "ai", "endpoints", "describe", medgemma_cfg["endpoint_id"],
                     f"--region={medgemma_cfg['region']}", "--format=value(dedicatedEndpointDns)"],
                    capture_output=True, text=True, timeout=30,
                )
                dedicated_dns = result.stdout.strip()
                if dedicated_dns:
                    logger.info("vertex_dns_discovered", dns=dedicated_dns)
            except Exception as e:
                logger.warning("vertex_dns_discovery_failed", error=str(e)[:100])

        medgemma = VertexMedGemmaAdapter(
            project_id=medgemma_cfg["project_id"],
            region=medgemma_cfg["region"],
            endpoint_id=medgemma_cfg["endpoint_id"],
            model_name=medgemma_cfg.get("model_name", "medgemma-4b-vertex"),
            gpu_hourly_rate=medgemma_cfg.get("gpu_hourly_rate", 1.15),
            dedicated_endpoint_dns=dedicated_dns or None,
        )
    else:
        medgemma = MedGemmaAdapter(
            hf_token=hf_token,
            endpoint_url=medgemma_cfg["endpoint_url"],
            model_name=medgemma_cfg["model_name"],
        )

    medgemma_max_tokens = medgemma_cfg.get("max_tokens", 512)

    gemini_cfg = config["models"]["gemini_pro"]
    gemini = GeminiAdapter(api_key=api_key, model=gemini_cfg["model_id"])

    judge_cfg = config["judge"]
    judge = GeminiAdapter(api_key=api_key, model=judge_cfg["model_id"])

    # Health checks
    logger.info("health_check_start")
    mg_ok = await medgemma.health_check()
    gm_ok = await gemini.health_check()
    logger.info("health_check_done", medgemma=mg_ok, gemini=gm_ok)

    if not mg_ok:
        logger.warning("medgemma_health_check_failed", msg="Will attempt benchmark anyway — HF endpoint may need cold start")
    if not gm_ok:
        print("ERROR: Gemini health check failed. Check GOOGLE_API_KEY.")
        sys.exit(1)

    # Select prompt mode
    prompt_cfg = config.get("prompt", {})
    mode = prompt_cfg.get("mode", "combined")
    use_simple_prompt = prompt_cfg.get("medgemma_simple", False)

    # Run benchmark
    mode_suffix = f"-{mode}" if mode != "combined" else ""
    run_id = f"medpix-multimodal{mode_suffix}-{datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(config["output"]["run_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("benchmark_start", run_id=run_id, n_cases=len(cases))
    start_time = time.perf_counter()

    if mode == "findings_only":
        medgemma_template = PROMPT_TEMPLATE_FINDINGS_ONLY
    elif mode == "diagnosis_only":
        medgemma_template = PROMPT_TEMPLATE_DIAGNOSIS_ONLY
    elif use_simple_prompt:
        medgemma_template = PROMPT_TEMPLATE_SIMPLE
    else:
        medgemma_template = PROMPT_TEMPLATE
    logger.info("prompt_selection", mode=mode, medgemma_simple=use_simple_prompt)

    # Select Gemini prompt template — use same mode-specific template for fair comparison
    if mode == "findings_only":
        gemini_template = PROMPT_TEMPLATE_FINDINGS_ONLY
    elif mode == "diagnosis_only":
        gemini_template = PROMPT_TEMPLATE_DIAGNOSIS_ONLY
    else:
        gemini_template = PROMPT_TEMPLATE

    # Process cases sequentially (respect concurrency limits)
    raw_results = []
    for i, case in enumerate(cases):
        medgemma_prompt = medgemma_template.format(history=case["history"])
        gemini_prompt = gemini_template.format(history=case["history"])
        result = await run_single_case(
            case, medgemma, gemini, medgemma_prompt, gemini_prompt, i, len(cases),
            medgemma_max_tokens=medgemma_max_tokens,
        )
        raw_results.append(result)

    # Save raw responses
    with open(run_dir / "raw_responses.json", "w") as f:
        json.dump(raw_results, f, indent=2, default=str)

    # Score all cases
    logger.info("scoring_start", n_cases=len(raw_results))
    scored_results = []
    for result in raw_results:
        if "error" in result:
            logger.warning("skipping_errored_case", uid=result["uid"])
            continue
        scored = await score_case(result, judge, mode=mode)
        scored_results.append(scored)

    # Save scored results
    with open(run_dir / "scored_results.json", "w") as f:
        json.dump(scored_results, f, indent=2, default=str)

    # Build and save summary
    summary = build_summary_table(scored_results, mode=mode)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save config snapshot
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    elapsed = time.perf_counter() - start_time
    logger.info(
        "benchmark_complete",
        run_id=run_id,
        elapsed_seconds=round(elapsed, 1),
        n_scored=len(scored_results),
    )

    # Print results
    print_comparison_table(summary)
    print(f"\nRun artifacts saved to: {run_dir}")
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
