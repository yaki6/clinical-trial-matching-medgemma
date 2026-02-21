#!/usr/bin/env python3
"""Zero-cost Proof of Concept: Two-stage criterion evaluation.

Loads MedGemma 27B reasoning from saved results.json, feeds each reasoning
to Gemini 3 Pro with a labeling prompt, and measures accuracy vs expert labels.

Go/No-Go: If 2-stage accuracy >= 80% (fixes 2+ of 6 errors), proceed.
If <= 70%, hypothesis invalidated.

Usage:
    uv run python scripts/two_stage_poc.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trialmatch.models.gemini import GeminiAdapter
from trialmatch.models.schema import CriterionVerdict
from trialmatch.validate.evaluator import (
    build_labeling_prompt,
    parse_criterion_verdict,
)

RESULTS_PATH = Path("runs/phase0-medgemma-27b-vertex-20260221-020334/results.json")

VERDICT_MAP = {
    "MET": CriterionVerdict.MET,
    "NOT_MET": CriterionVerdict.NOT_MET,
    "UNKNOWN": CriterionVerdict.UNKNOWN,
}


async def main():
    # 1. Load saved 27B results
    with open(RESULTS_PATH) as f:
        saved = json.load(f)

    print(f"Loaded {len(saved)} pairs from {RESULTS_PATH}")
    print()

    # 2. Set up Gemini adapter
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)

    gemini = GeminiAdapter(api_key=api_key)

    # 3. Run Stage 2 on each pair
    results = []
    for pair in saved:
        idx = pair["pair_index"]
        reasoning = pair["reasoning"]
        criterion_type = pair["criterion_type"]
        criterion_text = pair["criterion_text"]
        expert_label = VERDICT_MAP[pair["expert_label"]]
        original_verdict = VERDICT_MAP[pair["model_verdict"]]

        # Build Stage 2 prompt
        prompt = build_labeling_prompt(
            stage1_reasoning=reasoning,
            criterion_text=criterion_text,
            criterion_type=criterion_type,
        )

        # Call Gemini
        response = await gemini.generate(prompt, max_tokens=256)
        verdict, label_reasoning, evidence = parse_criterion_verdict(response.text)

        flipped = verdict != original_verdict
        correct = verdict == expert_label
        was_correct = original_verdict == expert_label

        results.append({
            "pair_index": idx,
            "patient_id": pair["patient_id"],
            "trial_id": pair["trial_id"],
            "criterion_type": criterion_type,
            "expert": expert_label.value,
            "original_27b": original_verdict.value,
            "two_stage": verdict.value,
            "flipped": flipped,
            "correct": correct,
            "was_correct": was_correct,
            "gemini_reasoning": label_reasoning[:120],
        })

        status = ""
        if flipped and correct and not was_correct:
            status = " FIXED"
        elif flipped and not correct and was_correct:
            status = " REGRESSED"
        elif flipped:
            status = " flipped"

        print(
            f"  #{idx:2d} [{criterion_type[:4]}] "
            f"expert={expert_label.value:7s}  "
            f"27B={original_verdict.value:7s}  "
            f"2-stage={verdict.value:7s}  "
            f"{'OK' if correct else 'WRONG'}{status}"
        )

    # 4. Summary
    print()
    print("=" * 60)
    original_correct = sum(1 for r in results if r["was_correct"])
    two_stage_correct = sum(1 for r in results if r["correct"])
    total = len(results)
    flipped_count = sum(1 for r in results if r["flipped"])
    fixed = sum(1 for r in results if r["flipped"] and r["correct"] and not r["was_correct"])
    regressed = sum(1 for r in results if r["flipped"] and not r["correct"] and r["was_correct"])

    print(f"Original 27B accuracy:  {original_correct}/{total} = {original_correct/total:.0%}")
    print(f"Two-stage accuracy:     {two_stage_correct}/{total} = {two_stage_correct/total:.0%}")
    print(f"Pairs flipped:          {flipped_count}")
    print(f"  Fixed (wrong→right):  {fixed}")
    print(f"  Regressed (right→wrong): {regressed}")
    print()

    if two_stage_correct / total >= 0.80:
        print("GO: Two-stage hypothesis VALIDATED (>= 80%)")
    elif two_stage_correct / total > original_correct / total:
        print("MARGINAL: Improved but below 80% threshold")
    else:
        print("NO-GO: Two-stage did not improve accuracy")

    print("=" * 60)

    # 5. Error analysis
    print()
    print("--- Error Analysis ---")
    for r in results:
        if not r["correct"]:
            print(
                f"  #{r['pair_index']:2d} [{r['criterion_type'][:4]}] "
                f"expert={r['expert']:7s}  2-stage={r['two_stage']:7s}  "
                f"was_correct={r['was_correct']}"
            )
            print(f"       {r['gemini_reasoning'][:100]}")


if __name__ == "__main__":
    asyncio.run(main())
