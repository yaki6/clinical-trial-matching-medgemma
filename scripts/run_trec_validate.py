#!/usr/bin/env python3
"""VALIDATE verification against TREC 2022 ground truth.

Runs criterion-level VALIDATE (two-stage: Gemini Pro for both stages)
on 10 sampled trials, aggregates to trial-level verdicts, and compares
to gold qrel scores.

Usage:
    uv run python scripts/run_trec_validate.py

Requires:
    - data/trec2022_ground_truth/trial_criteria_cache.json (from fetch_trec_trial_criteria.py)
    - GOOGLE_API_KEY in .env
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "trec2022_ground_truth"
CACHE_PATH = DATA_DIR / "trial_criteria_cache.json"

# Label mapping: 方案 A
VERDICT_TO_QREL = {
    "ELIGIBLE": 2,
    "EXCLUDED": 1,
    "UNCERTAIN": 1,
    "NOT_RELEVANT": 0,
}


# ---------------------------------------------------------------------------
# Inline data loading
# ---------------------------------------------------------------------------
def load_patient() -> dict:
    text = (DATA_DIR / "patient.jsonl").read_text().strip()
    return json.loads(text)


def adapt_patient(record: dict) -> str:
    """Return patient note text."""
    return record["text"]


def load_criteria_cache() -> dict:
    if not CACHE_PATH.exists():
        print(f"ERROR: Cache not found at {CACHE_PATH}")
        print("Run: uv run python scripts/fetch_trec_trial_criteria.py")
        sys.exit(1)
    return json.loads(CACHE_PATH.read_text())


# ---------------------------------------------------------------------------
# Run VALIDATE on a single trial
# ---------------------------------------------------------------------------
async def validate_trial(
    patient_note: str,
    nct_id: str,
    trial_data: dict,
    gemini_adapter,
) -> dict:
    """Run criterion-level VALIDATE on all criteria of one trial."""
    from trialmatch.validate.evaluator import evaluate_criterion_two_stage

    inclusion = trial_data.get("inclusion_criteria", [])
    exclusion = trial_data.get("exclusion_criteria", [])

    criterion_results = []

    # Evaluate inclusion criteria
    for i, criterion_text in enumerate(inclusion):
        try:
            result = await evaluate_criterion_two_stage(
                patient_note=patient_note,
                criterion_text=criterion_text,
                criterion_type="inclusion",
                reasoning_adapter=gemini_adapter,
                labeling_adapter=gemini_adapter,
                max_tokens_reasoning=2048,
                max_tokens_labeling=256,
            )
            criterion_results.append({
                "criterion_text": criterion_text,
                "criterion_type": "inclusion",
                "verdict": result.verdict.value,
                "reasoning": result.reasoning[:500],
                "stage1_reasoning": (result.stage1_reasoning or "")[:500],
            })
            print(f"    incl[{i}] {result.verdict.value:8s} | {criterion_text[:60]}")
        except Exception as e:
            print(f"    incl[{i}] ERROR    | {criterion_text[:60]} — {e}")
            criterion_results.append({
                "criterion_text": criterion_text,
                "criterion_type": "inclusion",
                "verdict": "UNKNOWN",
                "reasoning": f"Error: {e}",
                "stage1_reasoning": "",
            })

    # Evaluate exclusion criteria
    for i, criterion_text in enumerate(exclusion):
        try:
            result = await evaluate_criterion_two_stage(
                patient_note=patient_note,
                criterion_text=criterion_text,
                criterion_type="exclusion",
                reasoning_adapter=gemini_adapter,
                labeling_adapter=gemini_adapter,
                max_tokens_reasoning=2048,
                max_tokens_labeling=256,
            )
            criterion_results.append({
                "criterion_text": criterion_text,
                "criterion_type": "exclusion",
                "verdict": result.verdict.value,
                "reasoning": result.reasoning[:500],
                "stage1_reasoning": (result.stage1_reasoning or "")[:500],
            })
            print(f"    excl[{i}] {result.verdict.value:8s} | {criterion_text[:60]}")
        except Exception as e:
            print(f"    excl[{i}] ERROR    | {criterion_text[:60]} — {e}")
            criterion_results.append({
                "criterion_text": criterion_text,
                "criterion_type": "exclusion",
                "verdict": "UNKNOWN",
                "reasoning": f"Error: {e}",
                "stage1_reasoning": "",
            })

    return {
        "nct_id": nct_id,
        "brief_title": trial_data.get("brief_title", ""),
        "gold_qrel": trial_data.get("qrel_score"),
        "num_inclusion": len(inclusion),
        "num_exclusion": len(exclusion),
        "criterion_results": criterion_results,
    }


# ---------------------------------------------------------------------------
# Aggregate and compute metrics
# ---------------------------------------------------------------------------
def aggregate_trial(trial_result: dict) -> dict:
    """Aggregate criterion verdicts → trial verdict → predicted qrel."""
    from trialmatch.evaluation.metrics import TrialVerdict, aggregate_to_trial_verdict
    from trialmatch.models.schema import CriterionVerdict

    pairs = []
    for cr in trial_result["criterion_results"]:
        try:
            verdict = CriterionVerdict(cr["verdict"])
        except ValueError:
            verdict = CriterionVerdict.UNKNOWN
        pairs.append((verdict, cr["criterion_type"]))

    trial_verdict = aggregate_to_trial_verdict(pairs)
    predicted_qrel = VERDICT_TO_QREL.get(trial_verdict.value, 0)

    return {
        **trial_result,
        "trial_verdict": trial_verdict.value,
        "predicted_qrel": predicted_qrel,
        "correct": predicted_qrel == trial_result["gold_qrel"],
    }


def compute_trial_metrics(results: list[dict]) -> dict:
    """Compute accuracy, confusion matrix, per-class breakdown."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    gold = [r["gold_qrel"] for r in results]
    pred = [r["predicted_qrel"] for r in results]

    acc = accuracy_score(gold, pred)
    f1 = f1_score(gold, pred, labels=[0, 1, 2], average="macro", zero_division=0)
    cm = confusion_matrix(gold, pred, labels=[0, 1, 2])

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "confusion_matrix": cm.tolist(),
        "n_trials": len(results),
        "n_correct": sum(1 for r in results if r["correct"]),
        "per_trial": [
            {
                "nct_id": r["nct_id"],
                "gold": r["gold_qrel"],
                "pred": r["predicted_qrel"],
                "verdict": r["trial_verdict"],
                "correct": r["correct"],
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    from trialmatch.models.gemini import GeminiAdapter

    # Load data
    patient = load_patient()
    patient_note = adapt_patient(patient)
    cache = load_criteria_cache()

    print(f"Patient: {patient['_id']} — {patient['structured_profile']['diagnosis']}")
    print(f"Cached trials: {len(cache)}")
    print()

    # Init model
    gemini = GeminiAdapter()
    print(f"Model: {gemini._model} (both stages)")
    print()

    # Run VALIDATE on each cached trial
    all_results = []
    total_start = time.perf_counter()

    for idx, (nct_id, trial_data) in enumerate(sorted(cache.items()), 1):
        qrel = trial_data.get("qrel_score", "?")
        n_criteria = len(trial_data.get("inclusion_criteria", [])) + len(trial_data.get("exclusion_criteria", []))
        print(f"\n[{idx}/{len(cache)}] {nct_id} (gold_qrel={qrel}, criteria={n_criteria})")
        print(f"  Title: {trial_data.get('brief_title', '')[:80]}")

        start = time.perf_counter()
        result = await validate_trial(patient_note, nct_id, trial_data, gemini)
        elapsed = time.perf_counter() - start

        aggregated = aggregate_trial(result)
        all_results.append(aggregated)

        marker = "CORRECT" if aggregated["correct"] else "WRONG"
        print(f"  → verdict={aggregated['trial_verdict']} pred_qrel={aggregated['predicted_qrel']} "
              f"gold_qrel={aggregated['gold_qrel']} [{marker}] ({elapsed:.1f}s)")

    total_elapsed = time.perf_counter() - total_start

    # Compute metrics
    metrics = compute_trial_metrics(all_results)

    print(f"\n{'='*60}")
    print(f"VALIDATE RESULTS")
    print(f"{'='*60}")
    print(f"Trials:    {metrics['n_trials']}")
    print(f"Correct:   {metrics['n_correct']}/{metrics['n_trials']} ({metrics['accuracy']:.0%})")
    print(f"Macro F1:  {metrics['macro_f1']:.3f}")
    print(f"Latency:   {total_elapsed:.1f}s total")
    print(f"\nConfusion matrix (rows=gold, cols=pred, labels=[0,1,2]):")
    for row in metrics["confusion_matrix"]:
        print(f"  {row}")

    print(f"\nPer-trial breakdown:")
    for t in metrics["per_trial"]:
        marker = "OK" if t["correct"] else "XX"
        print(f"  [{marker}] {t['nct_id']} gold={t['gold']} pred={t['pred']} verdict={t['verdict']}")

    # Error analysis for wrong predictions
    wrong = [r for r in all_results if not r["correct"]]
    if wrong:
        print(f"\nError analysis ({len(wrong)} wrong):")
        for r in wrong:
            print(f"\n  {r['nct_id']} — gold={r['gold_qrel']} pred={r['predicted_qrel']} verdict={r['trial_verdict']}")
            print(f"  Title: {r['brief_title'][:80]}")
            # Show which criteria drove the wrong verdict
            for cr in r["criterion_results"]:
                if cr["verdict"] in ("MET", "NOT_MET"):
                    print(f"    {cr['criterion_type']:9s} {cr['verdict']:8s} | {cr['criterion_text'][:70]}")

    # Save results
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "runs" / f"trec_validate_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "results.json").write_text(json.dumps(all_results, indent=2, default=str))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
