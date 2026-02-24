#!/usr/bin/env python3
"""PRESCREEN verification against TREC 2022 ground truth.

Three-phase evaluation:
  Phase 1 — API Ceiling: Direct NCT ID lookup of ALL 647 qrels trials.
            Confirms existence in CT.gov API and captures status/date range.
            Establishes the theoretical maximum recall.

  Phase 2 — Enhanced Agent: Run adaptive PRESCREEN agent with expanded capacity:
            page_size=100, max_tool_calls=25, all statuses.
            Measures: can the agent's search strategy find gold trials?

  Phase 3 — Default Agent: Run PRESCREEN as production would (RECRUITING only,
            page_size capped at 50, max_tool_calls=25).
            Measures: real-world baseline.

Usage:
    uv run python scripts/run_trec_prescreen.py                  # Full run (Phase 1+2+3)
    uv run python scripts/run_trec_prescreen.py --skip-ceiling   # Skip Phase 1 (faster)
    uv run python scripts/run_trec_prescreen.py --phase2-only    # Only Phase 2
    uv run python scripts/run_trec_prescreen.py --phase3-only    # Only Phase 3
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "trec2022_ground_truth"

# All statuses for historical search
ALL_STATUSES = [
    "RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING",
    "COMPLETED", "TERMINATED", "WITHDRAWN", "UNKNOWN",
]


# ---------------------------------------------------------------------------
# Trace logger — saves every agent event to a JSON-lines file for auditing
# ---------------------------------------------------------------------------
class TraceLogger:
    """Writes structured trace events to a JSONL file for full audit trail."""

    def __init__(self, out_dir: Path):
        self.trace_file = out_dir / "agent_trace.jsonl"
        self._fh = open(self.trace_file, "a")
        self._event_count = 0

    def __call__(self, event_name: str, payload: dict[str, Any]) -> None:
        self._event_count += 1
        record = {
            "seq": self._event_count,
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_name,
            **payload,
        }
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()

    @property
    def event_count(self) -> int:
        return self._event_count


# ---------------------------------------------------------------------------
# Inline data loading
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Recall metrics
# ---------------------------------------------------------------------------
def compute_recall_metrics(returned_ncts: list[str], qrels: dict[str, int]) -> dict:
    eligible_set = {nct for nct, s in qrels.items() if s == 2}
    partial_set = {nct for nct, s in qrels.items() if s == 1}
    excluded_set = {nct for nct, s in qrels.items() if s == 0}
    all_qrel_set = set(qrels.keys())
    returned_set = set(returned_ncts)

    found_eligible = returned_set & eligible_set
    found_partial = returned_set & partial_set
    found_excluded = returned_set & excluded_set
    in_qrels = returned_set & all_qrel_set
    outside_qrels = returned_set - all_qrel_set

    top20 = set(returned_ncts[:20])
    top20_eligible = top20 & eligible_set

    return {
        "total_returned": len(returned_ncts),
        "recall_all_eligible": len(found_eligible) / len(eligible_set) if eligible_set else 0,
        "recall_at_20_eligible": len(top20_eligible) / len(eligible_set) if eligible_set else 0,
        "precision_in_qrels": len(in_qrels) / len(returned_set) if returned_set else 0,
        "found_eligible": len(found_eligible),
        "found_partial": len(found_partial),
        "found_excluded": len(found_excluded),
        "outside_qrels": len(outside_qrels),
        "eligible_total": len(eligible_set),
        "found_eligible_ncts": sorted(found_eligible),
        "missed_eligible_ncts": sorted(eligible_set - returned_set),
        "top20_eligible_ncts": sorted(top20_eligible),
        "in_qrels_ncts": sorted(in_qrels),
    }


# ---------------------------------------------------------------------------
# Phase 1: Exhaustive API ceiling
# ---------------------------------------------------------------------------
async def run_api_ceiling(qrels: dict[str, int]) -> dict:
    """Direct NCT ID lookup of ALL qrels trials to establish API ceiling."""
    from trialmatch.prescreen.ctgov_client import CTGovClient

    print(f"\n{'='*60}")
    print("PHASE 1: API CEILING (direct NCT ID lookup)")
    print(f"{'='*60}")

    eligible_set = {nct for nct, s in qrels.items() if s == 2}
    partial_set = {nct for nct, s in qrels.items() if s == 1}
    excluded_set = {nct for nct, s in qrels.items() if s == 0}
    all_ncts = sorted(qrels.keys())
    print(f"  Checking {len(all_ncts)} qrels trials "
          f"(eligible={len(eligible_set)}, partial={len(partial_set)}, excluded={len(excluded_set)})")

    ctgov = CTGovClient()
    found: dict[str, dict] = {}
    not_found: list[str] = []

    try:
        for i, nct in enumerate(all_ncts):
            try:
                raw = await ctgov.get_details(nct)
                proto = raw.get("protocolSection", {})
                status_mod = proto.get("statusModule", {})
                status = status_mod.get("overallStatus", "?")
                start_date = status_mod.get("startDateStruct", {}).get("date", "")
                completion_date = status_mod.get("primaryCompletionDateStruct", {}).get("date", "")
                found[nct] = {
                    "status": status,
                    "start_date": start_date,
                    "completion_date": completion_date,
                }
            except Exception as e:
                not_found.append(nct)
                print(f"    {nct}: NOT FOUND - {e}")

            if (i + 1) % 50 == 0 or (i + 1) == len(all_ncts):
                print(f"    ... {i+1}/{len(all_ncts)} checked, "
                      f"{len(found)} found, {len(not_found)} missing")
    finally:
        await ctgov.aclose()

    found_set = set(found.keys())
    found_eligible = found_set & eligible_set
    found_partial = found_set & partial_set
    found_excluded = found_set & excluded_set

    status_counts: dict[str, int] = {}
    for info in found.values():
        s = info["status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    start_dates = [info["start_date"] for info in found.values() if info["start_date"]]
    completion_dates = [info["completion_date"] for info in found.values() if info["completion_date"]]

    metrics = compute_recall_metrics(sorted(found_set), qrels)

    print(f"\n--- Phase 1 Results ---")
    print(f"Total qrels trials:     {len(all_ncts)}")
    print(f"Found in CT.gov API:    {len(found)}/{len(all_ncts)} ({len(found)/len(all_ncts):.1%})")
    print(f"  Eligible found:       {len(found_eligible)}/{len(eligible_set)}")
    print(f"  Partial found:        {len(found_partial)}/{len(partial_set)}")
    print(f"  Excluded found:       {len(found_excluded)}/{len(excluded_set)}")
    print(f"Ceiling recall (eligible): {metrics['recall_all_eligible']:.1%}")

    print(f"\nStatus distribution:")
    for s, c in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}")

    if start_dates:
        print(f"\nDate range:")
        print(f"  Start dates:      {min(start_dates)} .. {max(start_dates)}")
    if completion_dates:
        print(f"  Completion dates: {min(completion_dates)} .. {max(completion_dates)}")

    if not_found:
        print(f"\n  NOT in CT.gov API ({len(not_found)}): {not_found[:20]}")

    return {
        "phase": "api_ceiling",
        "total_qrels": len(all_ncts),
        "total_found": len(found),
        "metrics": metrics,
        "found_eligible": len(found_eligible),
        "found_partial": len(found_partial),
        "found_excluded": len(found_excluded),
        "status_distribution": status_counts,
        "not_found": not_found,
        "start_date_range": [min(start_dates), max(start_dates)] if start_dates else [],
        "completion_date_range": [min(completion_dates), max(completion_dates)] if completion_dates else [],
    }


# ---------------------------------------------------------------------------
# Patched ToolExecutor for historical + enhanced search
# ---------------------------------------------------------------------------
def make_enhanced_search_patch(
    use_all_statuses: bool = True,
    max_page_size: int = 100,
    date_cutoff: str | None = None,
):
    """Create a patched _search_trials with expanded capacity."""
    from trialmatch.prescreen.ctgov_client import parse_search_results, parse_study_summary

    async def patched_search(
        self,
        condition=None,
        intervention=None,
        eligibility_keywords=None,
        status=None,
        phase=None,
        location=None,
        min_age=None,
        max_age=None,
        sex=None,
        study_type=None,
        page_size=50,
        **_ignored,
    ):
        import structlog
        logger = structlog.get_logger()

        patched_status = ALL_STATUSES if use_all_statuses else (status or ["RECRUITING"])
        patched_page_size = max(min(int(page_size), max_page_size), 50)

        date_filter = None
        if date_cutoff:
            date_filter = f"AREA[StudyFirstPostDate]RANGE[MIN, {date_cutoff}]"

        start = time.perf_counter()
        raw = await self._ctgov.search(
            condition=condition,
            intervention=intervention,
            eligibility_keywords=eligibility_keywords,
            status=patched_status,
            phase=phase or None,
            location=location,
            sex=sex,
            min_age=min_age,
            max_age=max_age,
            study_type=study_type or None,
            advanced_query=date_filter,
            page_size=patched_page_size,
        )
        latency = (time.perf_counter() - start) * 1000

        studies = parse_search_results(raw)
        summaries = [parse_study_summary(s) for s in studies]

        compact = [
            {
                "nct_id": s["nct_id"],
                "brief_title": s["brief_title"],
                "phase": s["phase"],
                "conditions": s["conditions"][:3],
                "interventions": s["interventions"][:4],
                "status": s["status"],
                "sponsor": s["sponsor"],
                "enrollment": s["enrollment"],
            }
            for s in summaries
        ]

        total = raw.get("totalCount") or len(studies)
        summary = (
            f"Found {len(studies)} trials (API total: {total}), latency {latency:.0f}ms. "
            f"NCT IDs: {[s['nct_id'] for s in summaries]}"
        )
        logger.info("ctgov_search_complete", count=len(studies), query_condition=condition)
        return {"count": len(compact), "total_available": total, "trials": compact}, summary

    return patched_search


# ---------------------------------------------------------------------------
# Single PRESCREEN agent run
# ---------------------------------------------------------------------------
async def _create_medgemma_adapter():
    """Create Vertex MedGemma 27B adapter with health check."""
    try:
        from trialmatch.models.vertex_medgemma import VertexMedGemmaAdapter

        project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
        endpoint_id = os.environ.get("VERTEX_ENDPOINT_ID_27B") or os.environ.get("VERTEX_MEDGEMMA_ENDPOINT_ID")
        region = os.environ.get("GCP_REGION") or os.environ.get("VERTEX_REGION", "us-central1")
        dedicated_dns = os.environ.get("VERTEX_DEDICATED_DNS_27B")

        if not project_id or not endpoint_id:
            print("  [WARN] GCP_PROJECT_ID/VERTEX_ENDPOINT_ID_27B not set — MedGemma unavailable")
            return None

        adapter = VertexMedGemmaAdapter(
            project_id=project_id,
            region=region,
            endpoint_id=endpoint_id,
            model_name="medgemma-27b-vertex",
            dedicated_endpoint_dns=dedicated_dns,
            gpu_hourly_rate=2.30,
        )
        print(f"  MedGemma 27B: Vertex AI ({region}, endpoint={endpoint_id[:12]}...)")
        print("  Running health check...")
        healthy = await adapter.health_check()
        if healthy:
            print("  MedGemma 27B: HEALTHY")
            return adapter
        else:
            print("  [WARN] MedGemma 27B health check failed — agent proceeds without expert")
            return None
    except Exception as e:
        print(f"  [WARN] MedGemma adapter creation failed: {e}")
        return None


async def run_agent(
    patient_note: str,
    key_facts: dict,
    qrels: dict[str, int],
    run_label: str,
    trace_logger: TraceLogger | None = None,
    use_all_statuses: bool = False,
    max_tool_calls: int = 25,
    max_page_size: int = 50,
    use_medgemma: bool = True,
    date_cutoff: str | None = None,
) -> dict:
    from trialmatch.models.gemini import GeminiAdapter
    from trialmatch.prescreen.agent import run_prescreen_agent
    from trialmatch.prescreen.tools import ToolExecutor

    # MUST use Gemini Pro for PRESCREEN — full reasoning capability required
    gemini = GeminiAdapter(model="gemini-3-pro-preview")
    print(f"\n{'='*60}")
    print(f"PRESCREEN AGENT RUN: {run_label}")
    print(f"{'='*60}")
    print(f"Architecture: Adaptive Gemini Pro + MedGemma-as-tool")
    print(f"Model: {gemini._model}")
    print(f"All statuses: {use_all_statuses}")
    print(f"Max tool calls: {max_tool_calls}")
    print(f"Max page size: {max_page_size}")
    print(f"Use MedGemma: {use_medgemma}")
    print(f"Date cutoff: {date_cutoff or 'None (current)'}")

    # Log run config
    if trace_logger:
        trace_logger("run_config", {
            "run_label": run_label,
            "model": gemini._model,
            "use_all_statuses": use_all_statuses,
            "max_tool_calls": max_tool_calls,
            "max_page_size": max_page_size,
            "use_medgemma": use_medgemma,
            "date_cutoff": date_cutoff,
        })

    # Create MedGemma adapter if requested
    medgemma = None
    if use_medgemma:
        medgemma = await _create_medgemma_adapter()
    if medgemma is None:
        print("  [INFO] Running WITHOUT MedGemma — consult_medical_expert will return error")

    tool_calls_log = []

    def on_tool(tc):
        icon = "OK" if not tc.error else "ERR"
        short = ", ".join(f"{k}={str(v)[:40]}" for k, v in tc.args.items() if k != "page_size")
        line = f"  [{icon}] #{tc.call_index} {tc.tool_name}({short}) -> {tc.result_count} results, {tc.latency_ms:.0f}ms"
        print(line)
        entry = {
            "call_index": tc.call_index,
            "tool": tc.tool_name,
            "args": {k: v for k, v in tc.args.items()},
            "result_count": tc.result_count,
            "latency_ms": tc.latency_ms,
            "result_summary": tc.result_summary,
            "error": tc.error,
        }
        tool_calls_log.append(entry)

    def on_text(text):
        if text and len(text.strip()) > 20:
            preview = text[:200].replace('\n', ' ')
            print(f"  [Agent] {preview}...")

    start = time.perf_counter()

    # Always use patched search when date_cutoff or custom statuses/page_size needed
    needs_patch = use_all_statuses or max_page_size > 50 or date_cutoff is not None
    if needs_patch:
        patched = make_enhanced_search_patch(
            use_all_statuses=use_all_statuses,
            max_page_size=max_page_size,
            date_cutoff=date_cutoff,
        )
        with patch.object(ToolExecutor, "_search_trials", patched):
            result = await run_prescreen_agent(
                patient_note=patient_note,
                key_facts=key_facts,
                ingest_source="gold",
                gemini_adapter=gemini,
                medgemma_adapter=medgemma,
                topic_id="trec-20226",
                on_tool_call=on_tool,
                on_agent_text=on_text,
                max_tool_calls=max_tool_calls,
                trace_callback=trace_logger,
            )
    else:
        result = await run_prescreen_agent(
            patient_note=patient_note,
            key_facts=key_facts,
            ingest_source="gold",
            gemini_adapter=gemini,
            medgemma_adapter=medgemma,
            topic_id="trec-20226",
            on_tool_call=on_tool,
            on_agent_text=on_text,
            max_tool_calls=max_tool_calls,
            trace_callback=trace_logger,
        )
    elapsed = time.perf_counter() - start

    returned_ncts = [c.nct_id for c in result.candidates]
    metrics = compute_recall_metrics(returned_ncts, qrels)

    print(f"\n--- Results ({run_label}) ---")
    print(f"Tool calls:       {result.total_api_calls}")
    print(f"Unique NCTs seen: {result.total_unique_nct_ids}")
    print(f"Candidates:       {len(result.candidates)}")
    print(f"Latency:          {elapsed:.1f}s")
    print(f"Gemini cost:      ${result.gemini_estimated_cost:.4f}")
    print(f"Gemini tokens:    {result.gemini_input_tokens:,} in / {result.gemini_output_tokens:,} out")
    print(f"MedGemma calls:   {result.medgemma_calls}")
    print(f"MedGemma cost:    ${result.medgemma_estimated_cost:.4f}")

    # Condition terms diversity analysis
    searched_terms = result.condition_terms_searched
    unique_terms = list(dict.fromkeys(searched_terms))  # preserve order, dedupe
    print(f"\nSearch strategy:")
    print(f"  Condition terms searched: {len(searched_terms)} total, {len(unique_terms)} unique")
    for i, term in enumerate(unique_terms):
        count = searched_terms.count(term)
        dup = f" (x{count})" if count > 1 else ""
        print(f"    {i+1}. {term}{dup}")

    print(f"\nRecall (eligible, n={metrics['eligible_total']}):")
    print(f"  Recall@all:     {metrics['recall_all_eligible']:.1%} ({metrics['found_eligible']}/{metrics['eligible_total']})")
    print(f"  Recall@20:      {metrics['recall_at_20_eligible']:.1%}")
    print(f"  Precision(qrel):{metrics['precision_in_qrels']:.1%}")
    print(f"  Outside qrels:  {metrics['outside_qrels']}")
    print(f"  In qrels:       {len(metrics['in_qrels_ncts'])}")

    if metrics["found_eligible_ncts"]:
        print(f"\n  Found eligible:  {metrics['found_eligible_ncts'][:15]}{'...' if len(metrics['found_eligible_ncts']) > 15 else ''}")
    if metrics["in_qrels_ncts"]:
        print(f"  In qrels (all):  {metrics['in_qrels_ncts'][:15]}{'...' if len(metrics['in_qrels_ncts']) > 15 else ''}")
    if metrics["missed_eligible_ncts"]:
        print(f"  Missed eligible (first 10): {metrics['missed_eligible_ncts'][:10]}")

    run_data = {
        "run_label": run_label,
        "architecture": "adaptive_gemini_pro_medgemma_as_tool",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_s": elapsed,
        "total_tool_calls": result.total_api_calls,
        "total_candidates": len(result.candidates),
        "total_unique_ncts_seen": result.total_unique_nct_ids,
        "gemini_cost": result.gemini_estimated_cost,
        "gemini_tokens_in": result.gemini_input_tokens,
        "gemini_tokens_out": result.gemini_output_tokens,
        "medgemma_calls": result.medgemma_calls,
        "medgemma_cost": result.medgemma_estimated_cost,
        "condition_terms_searched": result.condition_terms_searched,
        "condition_terms_unique": unique_terms,
        "metrics": metrics,
        "tool_calls_log": tool_calls_log,
        "returned_ncts": returned_ncts,
        "agent_reasoning": result.agent_reasoning[:5000] if result.agent_reasoning else "",
        "use_all_statuses": use_all_statuses,
        "use_medgemma": use_medgemma,
        "max_tool_calls": max_tool_calls,
        "max_page_size": max_page_size,
        "medgemma_available": medgemma is not None,
    }

    return run_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    # Parse CLI args
    skip_ceiling = "--skip-ceiling" in sys.argv
    phase2_only = "--phase2-only" in sys.argv
    phase3_only = "--phase3-only" in sys.argv

    patient = load_patient()
    qrels = load_qrels()
    patient_note, key_facts = adapt_patient(patient)

    eligible_count = sum(1 for s in qrels.values() if s == 2)
    partial_count = sum(1 for s in qrels.values() if s == 1)
    excluded_count = sum(1 for s in qrels.values() if s == 0)

    print(f"Patient: {patient['_id']} — {patient['structured_profile']['diagnosis']}")
    print(f"Key facts: {list(key_facts.keys())}")
    print(f"Qrels: {len(qrels)} trials — eligible={eligible_count}, "
          f"partial={partial_count}, excluded={excluded_count}")
    print(f"Flags: skip_ceiling={skip_ceiling}, phase2_only={phase2_only}, phase3_only={phase3_only}")

    # Create output directory early for trace logging
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "runs" / f"trec_prescreen_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging for structlog output
    file_handler = logging.FileHandler(out_dir / "structlog.log")
    file_handler.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)
    logging.root.setLevel(logging.DEBUG)

    # Create trace logger for agent events
    trace_logger = TraceLogger(out_dir)
    trace_logger("session_start", {
        "patient_id": patient["_id"],
        "qrels_total": len(qrels),
        "eligible": eligible_count,
        "partial": partial_count,
        "excluded": excluded_count,
        "skip_ceiling": skip_ceiling,
        "phase2_only": phase2_only,
        "phase3_only": phase3_only,
    })

    results: dict[str, Any] = {"patient_id": patient["_id"]}

    # Phase 1: API ceiling
    if not skip_ceiling and not phase2_only and not phase3_only:
        ceiling = await run_api_ceiling(qrels)
        results["ceiling"] = ceiling
        (out_dir / "phase1_ceiling.json").write_text(json.dumps(ceiling, indent=2))
    else:
        print("\n[SKIP] Phase 1 (API ceiling)")
        results["ceiling"] = None

    agent_runs = []

    # TREC 2022 corpus snapshot: April 27, 2021
    # Only return trials first posted on or before this date
    trec_date_cutoff = "2021-04-27"
    print(f"Date cutoff: {trec_date_cutoff} (TREC 2022 corpus snapshot)")

    # Phase 2: Enhanced agent (all statuses, page_size=100, 25 tool calls)
    if not phase3_only:
        run_enhanced = await run_agent(
            patient_note, key_facts, qrels,
            run_label="enhanced_all_statuses",
            trace_logger=trace_logger,
            use_all_statuses=True,
            max_tool_calls=25,
            max_page_size=100,
            use_medgemma=True,
            date_cutoff=trec_date_cutoff,
        )
        agent_runs.append(run_enhanced)
        (out_dir / "phase2_enhanced.json").write_text(json.dumps(run_enhanced, indent=2))

    # Phase 3: Default agent (RECRUITING only, standard limits)
    if not phase2_only:
        run_default = await run_agent(
            patient_note, key_facts, qrels,
            run_label="default_recruiting",
            trace_logger=trace_logger,
            use_all_statuses=False,
            max_tool_calls=25,
            max_page_size=50,
            use_medgemma=True,
            date_cutoff=trec_date_cutoff,
        )
        agent_runs.append(run_default)
        (out_dir / "phase3_default.json").write_text(json.dumps(run_default, indent=2))

    results["agent_runs"] = agent_runs

    # Save combined results
    (out_dir / "prescreen_results.json").write_text(json.dumps(results, indent=2))

    # Summary comparison (if both phases ran)
    if len(agent_runs) >= 2:
        e_metrics = agent_runs[0]["metrics"]
        d_metrics = agent_runs[1]["metrics"]

        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")

        header = f"{'Metric':<35} {'Enhanced':>12} {'Default':>12}"
        if results.get("ceiling"):
            header = f"{'Metric':<35} {'Ceiling':>10} {'Enhanced':>12} {'Default':>12}"

        print(header)
        print("-" * len(header))

        def row(label, c_val, e_val, d_val):
            if results.get("ceiling"):
                print(f"{label:<35} {c_val:>10} {e_val:>12} {d_val:>12}")
            else:
                print(f"{label:<35} {e_val:>12} {d_val:>12}")

        c = results.get("ceiling", {})
        row("Candidates returned", "—",
            str(agent_runs[0]["total_candidates"]),
            str(agent_runs[1]["total_candidates"]))
        row("Tool calls", "—",
            str(agent_runs[0]["total_tool_calls"]),
            str(agent_runs[1]["total_tool_calls"]))
        row("Recall@all (eligible)",
            f"{c.get('metrics', {}).get('recall_all_eligible', 0):.1%}" if c else "—",
            f"{e_metrics['recall_all_eligible']:.1%}",
            f"{d_metrics['recall_all_eligible']:.1%}")
        row("Found eligible",
            str(c.get("found_eligible", "—")) if c else "—",
            str(e_metrics["found_eligible"]),
            str(d_metrics["found_eligible"]))
        row("Gemini cost", "$0.00",
            f"${agent_runs[0]['gemini_cost']:.4f}",
            f"${agent_runs[1]['gemini_cost']:.4f}")

    elif len(agent_runs) == 1:
        r = agent_runs[0]
        m = r["metrics"]
        print(f"\n{'='*60}")
        print(f"SINGLE RUN SUMMARY: {r['run_label']}")
        print(f"{'='*60}")
        print(f"Recall@all: {m['recall_all_eligible']:.1%} ({m['found_eligible']}/{m['eligible_total']})")
        print(f"Candidates: {r['total_candidates']}, Tool calls: {r['total_tool_calls']}")
        print(f"Gemini cost: ${r['gemini_cost']:.4f}")

    trace_logger.close()
    print(f"\nResults saved to {out_dir}/")
    print(f"  prescreen_results.json  — combined results")
    print(f"  agent_trace.jsonl       — {trace_logger.event_count} trace events (full audit)")
    print(f"  structlog.log           — structured log output")
    if not phase3_only and agent_runs:
        print(f"  phase2_enhanced.json    — enhanced run details")
    if not phase2_only and len(agent_runs) > (0 if phase3_only else 1):
        print(f"  phase3_default.json     — default run details")


if __name__ == "__main__":
    asyncio.run(main())
