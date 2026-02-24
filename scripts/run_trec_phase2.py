#!/usr/bin/env python3
"""Run TREC Phase 2 — Compare MedGemma 27B vs Gemini Pro for search guidance.

Two runs with Tier 1 quick wins applied:
  Run A: MedGemma 27B generates search terms → Gemini Flash executes searches
  Run B: Gemini Pro fallback generates search terms → Gemini Flash executes searches

Both use: all statuses, page_size=50, max_tool_calls=20, MAX_CANDIDATES=200.
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

ALL_STATUSES = [
    "RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING",
    "COMPLETED", "TERMINATED", "WITHDRAWN", "UNKNOWN",
]


def load_patient() -> dict:
    text = (DATA_DIR / "patient.jsonl").read_text().strip()
    return json.loads(text)


def load_qrels() -> dict[str, int]:
    qrels = {}
    for line in (DATA_DIR / "qrels.tsv").read_text().splitlines():
        if line.startswith("query-id") or not line.strip():
            continue
        parts = line.split("\t")
        qrels[parts[1]] = int(parts[2])
    return qrels


def adapt_patient(record: dict) -> tuple[str, dict[str, str]]:
    sp = record["structured_profile"]
    key_facts = {
        "age": str(sp["age"]),
        "sex": sp["sex"],
        "primary_diagnosis": sp["diagnosis"],
        "comorbidities": "; ".join(sp["comorbidities"]),
        "smoking_history": sp["smoking_history"],
    }
    for k, v in sp.get("key_findings", {}).items():
        key_facts[k] = v
    return record["text"], key_facts


def compute_recall_metrics(returned_ncts: list[str], qrels: dict[str, int]) -> dict:
    eligible_set = {nct for nct, s in qrels.items() if s == 2}
    returned_set = set(returned_ncts)
    found_eligible = returned_set & eligible_set
    in_qrels = returned_set & set(qrels.keys())
    outside_qrels = returned_set - set(qrels.keys())
    top20 = set(returned_ncts[:20])
    top20_eligible = top20 & eligible_set

    return {
        "total_returned": len(returned_ncts),
        "recall_all_eligible": len(found_eligible) / len(eligible_set) if eligible_set else 0,
        "recall_at_20_eligible": len(top20_eligible) / len(eligible_set) if eligible_set else 0,
        "found_eligible": len(found_eligible),
        "outside_qrels": len(outside_qrels),
        "eligible_total": len(eligible_set),
        "found_eligible_ncts": sorted(found_eligible),
        "missed_eligible_ncts": sorted(eligible_set - returned_set),
    }


async def run_single(
    label: str,
    patient_note: str,
    key_facts: dict,
    qrels: dict[str, int],
    gemini_adapter,
    medgemma_adapter,
    out_dir: Path,
) -> dict:
    """Run one PRESCREEN agent and save artifacts."""
    from trialmatch.prescreen.agent import run_prescreen_agent
    from trialmatch.prescreen.tools import ToolExecutor
    from run_trec_prescreen import make_enhanced_search_patch

    print(f"\n{'='*60}")
    print(f"RUN: {label}")
    print(f"  MedGemma: {'YES' if medgemma_adapter else 'NO (Gemini fallback)'}")
    print(f"{'='*60}")

    trace_events: list[dict] = []
    tool_calls_log: list[dict] = []

    def trace_cb(event_name: str, payload: dict):
        trace_events.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_name,
            "payload": payload,
        })
        if event_name in ("medgemma_guidance_response", "gemini_fallback_guidance_response"):
            print(f"  [TRACE] Guidance response ({payload.get('latency_ms', 0):.0f}ms)")
        elif event_name == "medgemma_guidance_parsed":
            terms = payload.get("condition_terms", [])
            print(f"  [TRACE] Parsed {len(terms)} condition terms: {terms}")
        elif event_name == "tool_call_result":
            cond = payload.get("args", {}).get("condition", "")
            kw = payload.get("args", {}).get("eligibility_keywords", "")
            print(f"  [TRACE] {payload.get('tool_name')}({cond or kw}) -> {payload.get('result_count', '?')} results")
        elif event_name == "prescreen_complete":
            print(f"  [TRACE] Complete: {payload.get('candidate_count')} candidates, "
                  f"{payload.get('tool_calls')} tool calls")

    def on_tool(tc):
        icon = "OK" if not tc.error else "ERR"
        short = ", ".join(f"{k}={str(v)[:40]}" for k, v in tc.args.items() if k != "page_size")
        print(f"  [{icon}] {tc.tool_name}({short}) -> {tc.result_count} results, {tc.latency_ms:.0f}ms")
        tool_calls_log.append({
            "tool": tc.tool_name, "args": tc.args,
            "result_count": tc.result_count, "latency_ms": tc.latency_ms,
            "error": tc.error,
        })

    def on_text(text):
        if text and len(text.strip()) > 20:
            print(f"  [Agent] {text[:200]}...")

    # TREC 2022 ground truth trials were posted before mid-2022.
    # Date cutoff prevents recent (2023+) trials from crowding out relevant older ones.
    patched = make_enhanced_search_patch(
        use_all_statuses=True, max_page_size=100, date_cutoff="2022-06-01",
    )
    start = time.perf_counter()

    from unittest.mock import patch as mock_patch
    with mock_patch.object(ToolExecutor, "_search_trials", patched):
        result = await run_prescreen_agent(
            patient_note=patient_note,
            key_facts=key_facts,
            ingest_source="gold",
            gemini_adapter=gemini_adapter,
            medgemma_adapter=medgemma_adapter,
            topic_id="trec-20226",
            on_tool_call=on_tool,
            on_agent_text=on_text,
            trace_callback=trace_cb,
            max_tool_calls=20,
        )

    elapsed = time.perf_counter() - start
    returned_ncts = [c.nct_id for c in result.candidates]
    metrics = compute_recall_metrics(returned_ncts, qrels)

    # Print results
    print(f"\n--- {label} RESULTS ---")
    print(f"Tool calls:       {result.total_api_calls}")
    print(f"Unique trials:    {result.total_unique_nct_ids}")
    print(f"Candidates:       {len(result.candidates)}")
    print(f"Latency:          {elapsed:.1f}s")
    print(f"Gemini cost:      ${result.gemini_estimated_cost:.4f}")
    if result.medgemma_calls:
        print(f"MedGemma calls:   {result.medgemma_calls}")
        print(f"MedGemma cost:    ${result.medgemma_estimated_cost:.4f}")

    guidance_terms = result.medgemma_condition_terms or []
    if guidance_terms:
        suggested = set(guidance_terms)
        searched = set(result.condition_terms_searched)
        coverage = len(suggested & searched) / len(suggested) if suggested else 0
        print(f"\nGuidance coverage:")
        print(f"  Terms suggested ({len(suggested)}): {sorted(suggested)}")
        print(f"  Terms searched  ({len(searched)}): {sorted(searched)}")
        print(f"  Coverage:         {coverage:.0%}")

    print(f"\nRecall (eligible, n={metrics['eligible_total']}):")
    print(f"  Recall@all:     {metrics['recall_all_eligible']:.1%} ({metrics['found_eligible']}/{metrics['eligible_total']})")
    print(f"  Recall@20:      {metrics['recall_at_20_eligible']:.1%}")
    print(f"  Found eligible: {metrics['found_eligible_ncts'][:20]}")
    if len(metrics['found_eligible_ncts']) > 20:
        print(f"                  ...+{len(metrics['found_eligible_ncts']) - 20} more")
    print(f"  Missed (first 15): {metrics['missed_eligible_ncts'][:15]}")

    # Save artifacts
    run_dir = out_dir / label.lower().replace(" ", "_").replace(":", "_")
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "result_summary.json").write_text(json.dumps({
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_s": elapsed,
        "tool_calls": result.total_api_calls,
        "candidates": len(result.candidates),
        "unique_ncts": result.total_unique_nct_ids,
        "gemini_cost": result.gemini_estimated_cost,
        "gemini_tokens_in": result.gemini_input_tokens,
        "gemini_tokens_out": result.gemini_output_tokens,
        "medgemma_calls": result.medgemma_calls,
        "medgemma_cost": result.medgemma_estimated_cost,
        "metrics": metrics,
        "guidance_condition_terms": guidance_terms,
        "guidance_eligibility_keywords": result.medgemma_eligibility_keywords,
        "condition_terms_searched": result.condition_terms_searched,
        "returned_ncts": returned_ncts,
        "agent_reasoning": result.agent_reasoning[:3000],
    }, indent=2))
    (run_dir / "trace_events.json").write_text(json.dumps(trace_events, indent=2))
    (run_dir / "tool_calls.json").write_text(json.dumps(tool_calls_log, indent=2))
    (run_dir / "guidance_raw.txt").write_text(result.medgemma_guidance_raw)
    (run_dir / "tool_call_trace.json").write_text(json.dumps(
        [tc.model_dump() for tc in result.tool_call_trace], indent=2
    ))

    print(f"  Artifacts -> {run_dir}/")

    return {
        "label": label,
        "recall": metrics["recall_all_eligible"],
        "found_eligible": metrics["found_eligible"],
        "eligible_total": metrics["eligible_total"],
        "candidates": len(result.candidates),
        "unique_ncts": result.total_unique_nct_ids,
        "tool_calls": result.total_api_calls,
        "latency_s": elapsed,
        "gemini_cost": result.gemini_estimated_cost,
        "medgemma_cost": result.medgemma_estimated_cost,
        "guidance_terms": guidance_terms,
        "found_eligible_ncts": metrics["found_eligible_ncts"],
    }


async def main():
    from trialmatch.models.gemini import GeminiAdapter
    from trialmatch.models.vertex_medgemma import VertexMedGemmaAdapter

    patient = load_patient()
    qrels = load_qrels()
    patient_note, key_facts = adapt_patient(patient)

    eligible_count = sum(1 for s in qrels.values() if s == 2)
    print(f"Patient: {patient['_id']} — {patient['structured_profile']['diagnosis']}")
    print(f"Qrels: {len(qrels)} trials — eligible={eligible_count}")
    print(f"\nTier 1 changes: study_type=None, page_size=50, MAX_CANDIDATES=200, improved prompt")

    gemini = GeminiAdapter()
    print(f"Gemini model: {gemini._model}")

    # Create MedGemma adapter
    project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    endpoint_id = os.environ.get("VERTEX_ENDPOINT_ID_27B") or os.environ.get("VERTEX_MEDGEMMA_ENDPOINT_ID")
    region = os.environ.get("GCP_REGION") or os.environ.get("VERTEX_REGION", "us-central1")
    dedicated_dns = os.environ.get("VERTEX_DEDICATED_DNS_27B")

    medgemma = VertexMedGemmaAdapter(
        project_id=project_id, region=region, endpoint_id=endpoint_id,
        model_name="medgemma-27b-vertex", dedicated_endpoint_dns=dedicated_dns,
        gpu_hourly_rate=2.30,
    )
    print(f"MedGemma 27B: endpoint={endpoint_id}")
    print("Health check...")
    healthy = await medgemma.health_check()
    if not healthy:
        print("FATAL: MedGemma 27B not healthy — skipping Run A")
        medgemma = None

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "runs" / f"trec_phase2_compare_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Run A: MedGemma 27B guidance
    if medgemma:
        print("\nMedGemma 27B: HEALTHY")
        run_a = await run_single(
            label="A_medgemma_27b",
            patient_note=patient_note, key_facts=key_facts, qrels=qrels,
            gemini_adapter=gemini, medgemma_adapter=medgemma,
            out_dir=out_dir,
        )
        results.append(run_a)

    # Run B: Gemini Pro fallback guidance (no MedGemma)
    run_b = await run_single(
        label="B_gemini_pro_fallback",
        patient_note=patient_note, key_facts=key_facts, qrels=qrels,
        gemini_adapter=gemini, medgemma_adapter=None,
        out_dir=out_dir,
    )
    results.append(run_b)

    # Comparison table
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<30} ", end="")
    for r in results:
        print(f"{r['label']:>25}", end="")
    print()
    print("-" * (30 + 25 * len(results)))

    for metric, key, fmt in [
        ("Recall (eligible)", "recall", "{:.1%}"),
        ("Found eligible", "found_eligible", "{}"),
        ("Candidates", "candidates", "{}"),
        ("Unique NCTs seen", "unique_ncts", "{}"),
        ("Tool calls", "tool_calls", "{}"),
        ("Latency (s)", "latency_s", "{:.1f}"),
        ("Gemini cost", "gemini_cost", "${:.4f}"),
        ("MedGemma cost", "medgemma_cost", "${:.4f}"),
        ("Guidance terms", "guidance_terms", "{}"),
    ]:
        print(f"{metric:<30} ", end="")
        for r in results:
            val = r[key]
            if key == "guidance_terms":
                val = len(val)
            print(f"{fmt.format(val):>25}", end="")
        print()

    # Find trials unique to each run
    if len(results) == 2:
        set_a = set(results[0]["found_eligible_ncts"])
        set_b = set(results[1]["found_eligible_ncts"])
        only_a = set_a - set_b
        only_b = set_b - set_a
        both = set_a & set_b
        print(f"\nOverlap analysis:")
        print(f"  Found by both:     {len(both)} — {sorted(both)[:10]}")
        print(f"  Only MedGemma:     {len(only_a)} — {sorted(only_a)[:10]}")
        print(f"  Only Gemini Pro:   {len(only_b)} — {sorted(only_b)[:10]}")
        print(f"  Union:             {len(set_a | set_b)}")

    # Save comparison
    (out_dir / "comparison.json").write_text(json.dumps(results, indent=2))
    print(f"\nAll artifacts saved to {out_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
