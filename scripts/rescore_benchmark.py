#!/usr/bin/env python3
"""Re-score a benchmark run from raw_responses.json.

Useful when scoring phase failed/timed out but raw responses were saved.

Usage:
    uv run python scripts/rescore_benchmark.py runs/medpix-multimodal-findings_only-20260223-222130/
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trialmatch.evaluation.multimodal_metrics import (
    parse_model_response,
    score_diagnosis_exact,
    score_diagnosis_substring,
    score_findings_rouge,
)
from trialmatch.models.gemini import GeminiAdapter


FINDINGS_JUDGE_PROMPT = """You are a board-certified radiologist evaluating imaging findings quality.

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


async def judge_findings(judge, gold: str, predicted: str, history: str = "") -> dict:
    if not predicted.strip():
        return {"score": "poor", "key_finding_identified": False, "explanation": "Empty prediction"}
    prompt = FINDINGS_JUDGE_PROMPT.format(
        history=history, gold_findings=gold, predicted_findings=predicted
    )
    try:
        response = await judge.generate(prompt=prompt, max_tokens=256)
        return json.loads(response.text)
    except Exception as e:
        print(f"  Judge error: {e}", file=sys.stderr)
        return {"score": "poor", "key_finding_identified": False, "explanation": f"Judge failed: {e}"}


async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/rescore_benchmark.py <run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    raw_path = run_dir / "raw_responses.json"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found")
        sys.exit(1)

    with open(raw_path) as f:
        raw_results = json.load(f)

    print(f"Loaded {len(raw_results)} cases from {raw_path}")

    # Load config if available
    config_path = run_dir / "config.yaml"
    mode = "findings_only"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        mode = config.get("prompt", {}).get("mode", "combined")

    # Init judge
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    judge = GeminiAdapter(api_key=api_key, model="gemini-3-pro-preview")

    scored_results = []
    for i, result in enumerate(raw_results):
        uid = result["uid"]
        print(f"  Scoring {i+1}/{len(raw_results)}: {uid}")

        mg_parsed = parse_model_response(result["medgemma_raw"])
        gm_parsed = parse_model_response(result["gemini_raw"])

        gold_find = result["gold_findings"]
        gold_diag = result["gold_diagnosis"]

        # For findings_only: if parser found nothing, use raw text
        if mode == "findings_only":
            if not mg_parsed["findings"].strip():
                mg_parsed["findings"] = result["medgemma_raw"]
            if not gm_parsed["findings"].strip():
                gm_parsed["findings"] = result["gemini_raw"]

        # ROUGE scores
        mg_rouge = score_findings_rouge(gold_find, mg_parsed["findings"])
        gm_rouge = score_findings_rouge(gold_find, gm_parsed["findings"])

        # LLM judge for findings
        history = result.get("medgemma_prompt", "")
        mg_fj = await judge_findings(judge, gold_find, mg_parsed["findings"], history)
        gm_fj = await judge_findings(judge, gold_find, gm_parsed["findings"], history)

        scored_results.append({
            "uid": uid,
            "title": result.get("title", ""),
            "gold_diagnosis": gold_diag,
            "gold_findings": gold_find,
            "location_category": result.get("location_category", ""),
            "mode": mode,
            "medgemma": {
                "predicted_findings": mg_parsed["findings"][:1000],
                "findings_judge_score": mg_fj.get("score", "poor"),
                "findings_judge_key_finding": mg_fj.get("key_finding_identified", False),
                "findings_judge_explanation": mg_fj.get("explanation", ""),
                "rouge_recall": mg_rouge["recall"],
                "rouge_precision": mg_rouge["precision"],
                "rouge_fmeasure": mg_rouge["fmeasure"],
                "latency_ms": result.get("medgemma_latency_ms", 0),
                "cost": result.get("medgemma_cost", 0),
            },
            "gemini": {
                "predicted_findings": gm_parsed["findings"][:1000],
                "findings_judge_score": gm_fj.get("score", "poor"),
                "findings_judge_key_finding": gm_fj.get("key_finding_identified", False),
                "findings_judge_explanation": gm_fj.get("explanation", ""),
                "rouge_recall": gm_rouge["recall"],
                "rouge_precision": gm_rouge["precision"],
                "rouge_fmeasure": gm_rouge["fmeasure"],
                "latency_ms": result.get("gemini_latency_ms", 0),
                "cost": result.get("gemini_cost", 0),
            },
        })

    # Save scored results
    scored_path = run_dir / "scored_results.json"
    with open(scored_path, "w") as f:
        json.dump(scored_results, f, indent=2, default=str)
    print(f"\nScored results saved to {scored_path}")

    # Build summary
    n = len(scored_results)

    def agg(key):
        results = [r[key] for r in scored_results]
        return {
            "n_cases": n,
            "avg_latency_ms": sum(r["latency_ms"] for r in results) / n,
            "total_cost_usd": sum(r["cost"] for r in results),
            "findings_rouge_recall_mean": sum(r["rouge_recall"] for r in results) / n,
            "findings_rouge_precision_mean": sum(r["rouge_precision"] for r in results) / n,
            "findings_rouge_fmeasure_mean": sum(r["rouge_fmeasure"] for r in results) / n,
            "findings_judge_good": sum(1 for r in results if r["findings_judge_score"] == "good") / n,
            "findings_judge_partial": sum(1 for r in results if r["findings_judge_score"] == "partial") / n,
            "findings_judge_poor": sum(1 for r in results if r["findings_judge_score"] == "poor") / n,
            "findings_key_finding_rate": sum(1 for r in results if r.get("findings_judge_key_finding")) / n,
        }

    summary = {"mode": mode, "medgemma_4b": agg("medgemma"), "gemini_pro": agg("gemini")}
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")

    # Print table
    mg = summary["medgemma_4b"]
    gm = summary["gemini_pro"]
    print(f"\n{'='*80}")
    print(f"  MedPix Multi-Region 20-case Findings Benchmark")
    print(f"{'='*80}")
    print(f"{'Metric':<40} {'MedGemma 4B':>15} {'Gemini Flash':>15}")
    print(f"{'-'*80}")
    rows = [
        ("Findings — LLM Judge (good)", f"{mg['findings_judge_good']:.0%}", f"{gm['findings_judge_good']:.0%}"),
        ("Findings — LLM Judge (partial)", f"{mg['findings_judge_partial']:.0%}", f"{gm['findings_judge_partial']:.0%}"),
        ("Findings — LLM Judge (poor)", f"{mg['findings_judge_poor']:.0%}", f"{gm['findings_judge_poor']:.0%}"),
        ("Findings — Key Finding Rate", f"{mg['findings_key_finding_rate']:.0%}", f"{gm['findings_key_finding_rate']:.0%}"),
        ("Findings — ROUGE-L Recall", f"{mg['findings_rouge_recall_mean']:.3f}", f"{gm['findings_rouge_recall_mean']:.3f}"),
        ("Findings — ROUGE-L Precision", f"{mg['findings_rouge_precision_mean']:.3f}", f"{gm['findings_rouge_precision_mean']:.3f}"),
        ("Findings — ROUGE-L F1", f"{mg['findings_rouge_fmeasure_mean']:.3f}", f"{gm['findings_rouge_fmeasure_mean']:.3f}"),
        ("Avg Latency (ms)", f"{mg['avg_latency_ms']:.0f}", f"{gm['avg_latency_ms']:.0f}"),
        ("Total Cost ($)", f"${mg['total_cost_usd']:.4f}", f"${gm['total_cost_usd']:.4f}"),
    ]
    for label, mg_val, gm_val in rows:
        print(f"{label:<40} {mg_val:>15} {gm_val:>15}")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
