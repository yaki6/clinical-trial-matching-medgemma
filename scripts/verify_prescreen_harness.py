#!/usr/bin/env python3
"""Quick e2e verification: run PRESCREEN on mpx1016 patient with real Gemini.

Compares results against the harness (data/sot/prescreen/mpx1016_trial_harness.json).
Prints: search count, trial count, latency, cost, and harness overlap.
"""

import asyncio
import json
import os
import sys
import time

# Ensure repo root on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()


async def main():
    from trialmatch.ingest.profile_adapter import adapt_harness_patient, load_demo_harness
    from trialmatch.models.gemini import GeminiAdapter
    from trialmatch.prescreen.agent import run_prescreen_agent

    # Load patient
    patients = load_demo_harness()
    mpx1016 = next(p for p in patients if p["topic_id"] == "mpx1016")
    patient_note, key_facts = adapt_harness_patient(mpx1016)

    print(f"Patient: mpx1016 — 43yo female, lung adenocarcinoma (signet-ring)")
    print(f"Key facts keys: {list(key_facts.keys())}")
    print(f"  age={key_facts.get('age')}, sex={key_facts.get('sex')}")
    print()

    # Load harness for comparison
    harness_path = os.path.join(
        os.path.dirname(__file__), "..", "data", "sot", "prescreen", "mpx1016_trial_harness.json"
    )
    with open(harness_path) as f:
        harness = json.load(f)
    harness_ncts = {t["nct_id"] for t in harness["top_trials"]}
    print(f"Harness top trials: {harness_ncts}")
    print()

    # Init adapters
    gemini = GeminiAdapter()
    print(f"Gemini model: {gemini._model}")

    # Try to init MedGemma for clinical reasoning
    medgemma = None
    try:
        from trialmatch.models.medgemma import MedGemmaAdapter
        medgemma = MedGemmaAdapter()
        print(f"MedGemma: {medgemma.name} (clinical reasoning enabled)")
    except Exception as exc:
        print(f"MedGemma unavailable ({exc}) — running without clinical reasoning")
    print()

    # Run PRESCREEN
    def on_tool(tc):
        icon = "OK" if not tc.error else "ERR"
        print(f"  [{icon}] {tc.tool_name}({_short_args(tc.args)}) -> {tc.result_count} results, {tc.latency_ms:.0f}ms")

    def on_text(text):
        if text and len(text.strip()) > 20:
            print(f"  [Agent] {text[:120]}...")

    start = time.perf_counter()
    result = await run_prescreen_agent(
        patient_note=patient_note,
        key_facts=key_facts,
        ingest_source="gold",
        gemini_adapter=gemini,
        medgemma_adapter=medgemma,
        topic_id="mpx1016",
        on_tool_call=on_tool,
        on_agent_text=on_text,
    )
    elapsed = time.perf_counter() - start

    # Results
    print()
    print("=" * 60)
    print(f"PRESCREEN RESULTS")
    print(f"=" * 60)
    print(f"Tool calls:      {result.total_api_calls}")
    print(f"Unique trials:   {result.total_unique_nct_ids}")
    print(f"Candidates (after pruning): {len(result.candidates)}")
    print(f"Total latency:   {elapsed:.1f}s")
    print(f"Gemini tokens:   {result.gemini_input_tokens} in / {result.gemini_output_tokens} out")
    print(f"Gemini cost:     ${result.gemini_estimated_cost:.4f}")
    print(f"MedGemma calls:  {result.medgemma_calls}")
    print()

    # Compare against harness
    result_ncts = {c.nct_id for c in result.candidates}
    overlap = harness_ncts & result_ncts
    missed = harness_ncts - result_ncts
    print(f"Harness overlap: {len(overlap)}/{len(harness_ncts)} ({len(overlap)/len(harness_ncts)*100:.0f}%)")
    if overlap:
        print(f"  Found: {overlap}")
    if missed:
        print(f"  Missed: {missed}")
    print()

    # Print top 10 candidates
    print("Top 10 candidates:")
    for i, c in enumerate(result.candidates[:10], 1):
        marker = " *HARNESS*" if c.nct_id in harness_ncts else ""
        print(f"  {i}. {c.nct_id} — {c.brief_title[:70]}... (queries={len(c.found_by_queries)}){marker}")

    print()
    print("Agent reasoning (last):")
    print(result.agent_reasoning[:500] if result.agent_reasoning else "(none)")


def _short_args(args):
    parts = []
    for k, v in args.items():
        if k == "page_size":
            continue
        sv = str(v)[:40]
        parts.append(f"{k}={sv}")
    return ", ".join(parts)


if __name__ == "__main__":
    asyncio.run(main())
