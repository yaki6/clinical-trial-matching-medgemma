#!/usr/bin/env python3
"""Run the full e2e pipeline (INGEST -> PRESCREEN -> VALIDATE) and write demo cache files.

This script generates cached data for the Streamlit demo, supporting two patients:
  - mpx1016: 43F, lung adenocarcinoma, multimodal (has CT image)
  - trec-20226: 61M, mesothelioma, text-only, with TREC 2022 ground truth

Usage:
    uv run python scripts/run_demo_cache.py --patient mpx1016
    uv run python scripts/run_demo_cache.py --patient trec-20226
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — must happen before local imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "demo"))

import dotenv  # noqa: E402

dotenv.load_dotenv(ROOT / ".env")

from trialmatch.ingest.profile_adapter import (  # noqa: E402
    adapt_harness_patient,
    get_image_path,
    load_demo_harness,
    merge_image_findings,
)
from trialmatch.prescreen.agent import run_prescreen_agent  # noqa: E402
from trialmatch.prescreen.ctgov_client import CTGovClient  # noqa: E402
from trialmatch.validate.evaluator import evaluate_criterion_two_stage  # noqa: E402
from trialmatch.evaluation.metrics import aggregate_to_trial_verdict  # noqa: E402
from trialmatch.live_runtime import (  # noqa: E402
    create_imaging_adapter,
    create_prescreen_adapters,
    create_validate_adapters,
)
from cache_manager import (  # noqa: E402
    save_ingest_result,
    save_prescreen_result,
    save_validate_results,
    save_cached_manifest,
    validate_cached_run,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_PATIENTS = ("mpx1016", "trec-20226")
HARNESS_PATH = ROOT / "data" / "harness schema" / "ingest" / "nsclc_demo_harness.json"
TREC_PATIENT_JSONL = ROOT / "data" / "trec2022_ground_truth" / "patient.jsonl"
TREC_QRELS_PATH = ROOT / "data" / "trec2022_ground_truth" / "qrels.tsv"

RADIOLOGY_PROMPT = (
    "Analyze this chest CT scan. Describe the key findings, any masses, "
    "nodules, effusions, lymphadenopathy, or other abnormalities. "
    "Be specific about location and size."
)
RADIOLOGY_SYSTEM_MSG = "You are an expert radiologist analyzing medical images."

logger = logging.getLogger("run_demo_cache")


# ---------------------------------------------------------------------------
# Eligibility criteria parser (copied from demo/pages/1_Pipeline_Demo.py)
# ---------------------------------------------------------------------------
def parse_eligibility_criteria(criteria_text: str) -> list[dict]:
    """Split raw eligibility criteria text into individual criteria with types."""
    criteria: list[dict] = []
    current_type = "inclusion"
    for line in criteria_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if "inclusion" in line.lower() and "criteria" in line.lower():
            current_type = "inclusion"
            continue
        if "exclusion" in line.lower() and "criteria" in line.lower():
            current_type = "exclusion"
            continue
        if line.startswith(("*", "-", "~")):
            line = line[1:].strip()
        if len(line) > 10:
            criteria.append({"text": line, "type": current_type})
    return criteria


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(log_file: str) -> None:
    """Configure logging to both console and file."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="w"))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)


def _find_harness_patient(topic_id: str) -> dict | None:
    """Look up a patient by topic_id in the demo harness file."""
    if not HARNESS_PATH.exists():
        logger.warning("Demo harness file not found: %s", HARNESS_PATH)
        return None
    patients = load_demo_harness(HARNESS_PATH)
    for p in patients:
        if p.get("topic_id") == topic_id:
            return p
    return None


def _load_trec_patient(topic_id: str) -> dict | None:
    """Load a TREC patient from the ground truth JSONL and convert to harness format.

    The TREC patient JSONL has a different schema than the demo harness, so we
    adapt it into a compatible format that adapt_harness_patient() can process.
    """
    if not TREC_PATIENT_JSONL.exists():
        logger.error("TREC patient JSONL not found: %s", TREC_PATIENT_JSONL)
        return None

    with open(TREC_PATIENT_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            patient = json.loads(line)
            if patient.get("_id") == topic_id:
                # Convert TREC format -> harness format
                profile = patient.get("structured_profile", {})
                key_facts = []

                # Primary diagnosis
                if profile.get("diagnosis"):
                    key_facts.append({
                        "field": "primary_diagnosis",
                        "value": profile["diagnosis"],
                        "evidence_span": profile.get("diagnosis_evidence", ""),
                        "required": True,
                    })

                # Demographics
                key_facts.append({
                    "field": "demographics",
                    "value": {
                        "age": str(profile.get("age", "unknown")),
                        "sex": profile.get("sex", "unknown"),
                    },
                    "evidence_span": None,
                    "required": True,
                })

                # Key findings
                findings = profile.get("key_findings", {})
                if findings:
                    key_facts.append({
                        "field": "key_findings",
                        "value": [
                            f"{k}: {v}" for k, v in findings.items()
                        ],
                        "evidence_span": None,
                        "required": True,
                    })

                # Comorbidities
                if profile.get("comorbidities"):
                    key_facts.append({
                        "field": "comorbidities",
                        "value": profile["comorbidities"],
                        "evidence_span": None,
                        "required": False,
                    })

                # Smoking history
                if profile.get("smoking_history"):
                    key_facts.append({
                        "field": "smoking_history",
                        "value": profile["smoking_history"],
                        "evidence_span": None,
                        "required": False,
                    })

                return {
                    "topic_id": topic_id,
                    "source_dataset": "TREC 2022",
                    "ingest_mode": "text",
                    "ehr_text": patient.get("text", ""),
                    "profile_text": patient.get("text", ""),
                    "key_facts": key_facts,
                    "image": None,
                }

    return None


def _load_patient(topic_id: str) -> dict:
    """Load patient profile from harness or TREC data."""
    # Try harness first
    patient = _find_harness_patient(topic_id)
    if patient:
        logger.info("Loaded patient '%s' from demo harness", topic_id)
        return patient

    # Fallback: TREC data
    if topic_id.startswith("trec-"):
        patient = _load_trec_patient(topic_id)
        if patient:
            logger.info("Loaded patient '%s' from TREC ground truth", topic_id)
            return patient

    logger.error(
        "Patient '%s' not found in demo harness (%s) or TREC data (%s)",
        topic_id, HARNESS_PATH, TREC_PATIENT_JSONL,
    )
    sys.exit(1)


def _load_qrels(topic_id: str) -> dict[str, int]:
    """Load TREC 2022 qrels for a specific topic. Returns {nct_id: score}."""
    if not TREC_QRELS_PATH.exists():
        logger.warning("TREC qrels file not found: %s", TREC_QRELS_PATH)
        return {}

    qrels: dict[str, int] = {}
    with open(TREC_QRELS_PATH) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header
            parts = line.strip().split("\t")
            if len(parts) >= 3 and parts[0] == topic_id:
                qrels[parts[1]] = int(parts[2])
    return qrels


async def _fetch_trial_criteria(nct_id: str) -> str:
    """Fetch eligibility criteria text for a trial from CT.gov."""
    client = CTGovClient()
    try:
        details = await client.get_details(nct_id)
        protocol = details.get("protocolSection", {})
        eligibility = protocol.get("eligibilityModule", {})
        return eligibility.get("eligibilityCriteria", "")
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Phase 1: INGEST
# ---------------------------------------------------------------------------
async def run_ingest(patient_id: str, profile: dict) -> tuple[str, dict]:
    """Run INGEST phase: adapt patient profile, optionally process image."""
    logger.info("=" * 60)
    logger.info("PHASE 1: INGEST for %s", patient_id)
    logger.info("=" * 60)

    start = time.perf_counter()

    # Adapt profile to get patient_note + key_facts
    patient_note, key_facts = adapt_harness_patient(profile)
    logger.info("Adapted patient profile: %d chars note, %d key_facts fields",
                len(patient_note), len(key_facts))

    # Multimodal: process CT image with MedGemma 1.5 4B
    image_path = get_image_path(profile, base_dir=ROOT)
    if image_path and image_path.exists():
        logger.info("Processing CT image: %s", image_path)
        hf_token = os.environ.get("HF_TOKEN", "")
        imaging_adapter = create_imaging_adapter(hf_token=hf_token)
        response = await imaging_adapter.generate_with_image(
            prompt=RADIOLOGY_PROMPT,
            image_path=image_path,
            max_tokens=512,
            system_message=RADIOLOGY_SYSTEM_MSG,
        )
        image_findings = {
            "extracted_text": response.text,
            "latency_seconds": response.latency_ms / 1000,
            "model": "medgemma-4b-vertex",
            "prompt": RADIOLOGY_PROMPT,
        }
        key_facts = merge_image_findings(key_facts, image_findings)
        logger.info("Image analysis complete (%.1fs): %d chars",
                     response.latency_ms / 1000, len(response.text))
    elif image_path:
        logger.warning("Image path specified but file not found: %s", image_path)
    else:
        logger.info("Text-only patient — no image processing")

    # Save INGEST result
    save_ingest_result(patient_id, patient_note, key_facts)

    elapsed = time.perf_counter() - start
    logger.info("INGEST complete in %.1fs", elapsed)

    return patient_note, key_facts


# ---------------------------------------------------------------------------
# Phase 2: PRESCREEN
# ---------------------------------------------------------------------------
async def run_prescreen(
    patient_id: str,
    patient_note: str,
    key_facts: dict,
    is_historical: bool = False,
) -> object:
    """Run PRESCREEN phase: agentic trial search via Gemini + MedGemma."""
    logger.info("=" * 60)
    logger.info("PHASE 2: PRESCREEN for %s", patient_id)
    logger.info("=" * 60)

    start = time.perf_counter()

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    if not api_key:
        logger.error("GOOGLE_API_KEY not set. Cannot run PRESCREEN.")
        sys.exit(1)

    gemini_adapter, medgemma_27b = create_prescreen_adapters(api_key, hf_token)

    # For historical TREC cases, search across all statuses
    note = patient_note
    if is_historical:
        note += (
            "\n\n**Note: Search across all trial statuses including COMPLETED, "
            "TERMINATED, and ACTIVE_NOT_RECRUITING, as this is a historical "
            "validation case.**"
        )

    def on_tool_call(record):
        logger.info(
            "  Tool[%d] %s -> %s (%.0fms)%s",
            record.call_index,
            record.tool_name,
            record.result_summary[:80] if record.result_summary else "ok",
            record.latency_ms,
            f" ERROR: {record.error}" if record.error else "",
        )

    def on_agent_text(text: str):
        # Log first 120 chars of agent reasoning
        preview = text.replace("\n", " ")[:120]
        logger.info("  Agent: %s%s", preview, "..." if len(text) > 120 else "")

    prescreen_result = await run_prescreen_agent(
        patient_note=note,
        key_facts=key_facts,
        ingest_source="gold",
        gemini_adapter=gemini_adapter,
        medgemma_adapter=medgemma_27b,
        max_tool_calls=25,
        topic_id=patient_id,
        on_tool_call=on_tool_call,
        on_agent_text=on_agent_text,
    )

    # Save PRESCREEN result
    save_prescreen_result(patient_id, prescreen_result)

    elapsed = time.perf_counter() - start
    logger.info(
        "PRESCREEN complete in %.1fs: %d candidates, %d tool calls, "
        "cost=$%.4f (Gemini) + $%.4f (MedGemma)",
        elapsed,
        len(prescreen_result.candidates),
        prescreen_result.total_api_calls,
        prescreen_result.gemini_estimated_cost,
        prescreen_result.medgemma_estimated_cost,
    )

    return prescreen_result


# ---------------------------------------------------------------------------
# Phase 3: VALIDATE
# ---------------------------------------------------------------------------
async def run_validate(
    patient_id: str,
    patient_note: str,
    prescreen_result: object,
    max_validate_trials: int,
    max_criteria: int,
    trial_ids: list[str] | None = None,
) -> dict:
    """Run VALIDATE phase: criterion-level evaluation for top candidates."""
    logger.info("=" * 60)
    logger.info("PHASE 3: VALIDATE for %s (max %d trials, %d criteria each)",
                patient_id, max_validate_trials, max_criteria)
    logger.info("=" * 60)

    start = time.perf_counter()

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")

    reasoning_adapter, labeling_adapter = create_validate_adapters(
        "Two-Stage (MedGemma \u2192 Gemini)",
        api_key=api_key,
        hf_token=hf_token,
    )

    # Select candidates: use explicit trial_ids if provided, else top-N by query count
    if trial_ids:
        id_set = set(trial_ids)
        candidates = [c for c in prescreen_result.candidates if c.nct_id in id_set]
        # Preserve the order of trial_ids
        order = {nct: i for i, nct in enumerate(trial_ids)}
        candidates.sort(key=lambda c: order.get(c.nct_id, 999))
        logger.info("Using explicit trial IDs: %s", trial_ids)
    else:
        candidates = sorted(
            prescreen_result.candidates,
            key=lambda c: len(c.found_by_queries),
            reverse=True,
        )[:max_validate_trials]

    logger.info("Validating %d trials: %s",
                len(candidates),
                [c.nct_id for c in candidates])

    validate_data: dict = {}
    total_criteria_evaluated = 0

    for trial_idx, candidate in enumerate(candidates):
        nct_id = candidate.nct_id
        logger.info("-" * 40)
        logger.info("Trial %d/%d: %s — %s",
                     trial_idx + 1, len(candidates), nct_id,
                     candidate.title[:60])

        # Fetch full eligibility criteria from CT.gov
        try:
            criteria_text = await _fetch_trial_criteria(nct_id)
        except Exception as exc:
            logger.error("Failed to fetch criteria for %s: %s", nct_id, exc)
            validate_data[nct_id] = {
                "verdict": "UNCERTAIN",
                "mode": "two_stage",
                "criteria": [],
                "error": str(exc),
            }
            continue

        if not criteria_text:
            logger.warning("No eligibility criteria found for %s", nct_id)
            validate_data[nct_id] = {
                "verdict": "UNCERTAIN",
                "mode": "two_stage",
                "criteria": [],
                "error": "No eligibility criteria in CT.gov record",
            }
            continue

        # Parse criteria into individual items
        parsed_criteria = parse_eligibility_criteria(criteria_text)
        logger.info("Parsed %d criteria (%d inclusion, %d exclusion)",
                     len(parsed_criteria),
                     sum(1 for c in parsed_criteria if c["type"] == "inclusion"),
                     sum(1 for c in parsed_criteria if c["type"] == "exclusion"))

        # Limit criteria count
        criteria_to_evaluate = parsed_criteria[:max_criteria]
        criteria_results: list[dict] = []
        verdict_tuples: list[tuple] = []

        for crit_idx, criterion in enumerate(criteria_to_evaluate):
            crit_text = criterion["text"]
            crit_type = criterion["type"]

            logger.info("  Criterion %d/%d [%s]: %s",
                         crit_idx + 1, len(criteria_to_evaluate),
                         crit_type, crit_text[:80])

            try:
                result = await evaluate_criterion_two_stage(
                    patient_note=patient_note,
                    criterion_text=crit_text,
                    criterion_type=crit_type,
                    reasoning_adapter=reasoning_adapter,
                    labeling_adapter=labeling_adapter,
                )
                criteria_results.append({
                    "text": crit_text,
                    "type": crit_type,
                    "verdict": result.verdict.name,
                    "reasoning": result.reasoning or "",
                    "evidence_sentences": result.evidence_sentences or [],
                    "stage1_reasoning": result.stage1_reasoning or "",
                })
                verdict_tuples.append((result.verdict, crit_type))
                total_criteria_evaluated += 1

                logger.info("    -> %s (stage1: %.0fms, stage2: %.0fms)",
                             result.verdict.name,
                             result.stage1_response.latency_ms if result.stage1_response else 0,
                             result.model_response.latency_ms if result.model_response else 0)

            except Exception as exc:
                logger.error("    -> ERROR evaluating criterion: %s", exc)
                from trialmatch.models.schema import CriterionVerdict
                criteria_results.append({
                    "text": crit_text,
                    "type": crit_type,
                    "verdict": "UNKNOWN",
                    "reasoning": f"Error: {exc}",
                    "evidence_sentences": [],
                    "stage1_reasoning": "",
                })
                verdict_tuples.append((CriterionVerdict.UNKNOWN, crit_type))
                total_criteria_evaluated += 1

        # Aggregate criterion verdicts to trial verdict
        trial_verdict = aggregate_to_trial_verdict(verdict_tuples)
        logger.info("Trial %s verdict: %s", nct_id, trial_verdict.name)

        validate_data[nct_id] = {
            "verdict": trial_verdict.name,
            "mode": "two_stage",
            "criteria": criteria_results,
        }

    # Save VALIDATE results
    prescreen_trial_ids = [c.nct_id for c in prescreen_result.candidates]
    validated_trial_ids = list(validate_data.keys())

    save_validate_results(patient_id, validate_data)
    save_cached_manifest(patient_id, prescreen_trial_ids, validated_trial_ids, "two_stage")

    # Validate cache consistency
    report = validate_cached_run(patient_id)
    if report.valid:
        logger.info("Cache validation PASSED for %s", patient_id)
    else:
        logger.warning("Cache validation FAILED for %s: %s", patient_id, report.errors)

    elapsed = time.perf_counter() - start
    logger.info(
        "VALIDATE complete in %.1fs: %d trials, %d criteria evaluated",
        elapsed, len(validate_data), total_criteria_evaluated,
    )

    return validate_data


# ---------------------------------------------------------------------------
# Phase 4: Ground Truth Comparison (trec-20226 only)
# ---------------------------------------------------------------------------
def run_ground_truth_comparison(
    patient_id: str,
    prescreen_result: object,
    validate_data: dict,
) -> None:
    """Compare pipeline results against TREC 2022 ground truth."""
    if not patient_id.startswith("trec-"):
        return

    logger.info("=" * 60)
    logger.info("PHASE 4: GROUND TRUTH COMPARISON for %s", patient_id)
    logger.info("=" * 60)

    qrels = _load_qrels(patient_id)
    if not qrels:
        logger.warning("No qrels found for %s — skipping ground truth comparison", patient_id)
        return

    # Categorize by score
    eligible_ncts = {nct for nct, score in qrels.items() if score == 2}
    excluded_ncts = {nct for nct, score in qrels.items() if score == 1}
    not_relevant_ncts = {nct for nct, score in qrels.items() if score == 0}

    logger.info("TREC qrels: %d eligible, %d excluded, %d not_relevant",
                 len(eligible_ncts), len(excluded_ncts), len(not_relevant_ncts))

    # PRESCREEN recall: how many eligible (score=2) trials were found
    prescreen_ncts = {c.nct_id for c in prescreen_result.candidates}
    eligible_found = eligible_ncts & prescreen_ncts
    eligible_missed = eligible_ncts - prescreen_ncts

    prescreen_recall = len(eligible_found) / len(eligible_ncts) if eligible_ncts else 0.0

    logger.info("PRESCREEN recall for eligible trials: %.1f%% (%d/%d)",
                 prescreen_recall * 100, len(eligible_found), len(eligible_ncts))
    if eligible_missed:
        logger.info("Missed eligible trials: %s", sorted(eligible_missed))

    # VALIDATE comparison: compare predicted vs gold for evaluated trials
    validate_comparison = []
    for nct_id, trial_data in validate_data.items():
        if nct_id not in qrels:
            validate_comparison.append({
                "nct_id": nct_id,
                "predicted": trial_data["verdict"],
                "gold": None,
                "gold_score": None,
                "match": None,
            })
            continue

        gold_score = qrels[nct_id]
        # Map gold score: 2->ELIGIBLE, 1->EXCLUDED, 0->NOT_RELEVANT
        gold_label_map = {2: "ELIGIBLE", 1: "EXCLUDED", 0: "NOT_RELEVANT"}
        gold_label = gold_label_map.get(gold_score, "UNKNOWN")
        predicted = trial_data["verdict"]

        match = predicted == gold_label
        validate_comparison.append({
            "nct_id": nct_id,
            "predicted": predicted,
            "gold": gold_label,
            "gold_score": gold_score,
            "match": match,
        })
        logger.info("  %s: predicted=%s, gold=%s -> %s",
                     nct_id, predicted, gold_label,
                     "MATCH" if match else "MISMATCH")

    # Build comparison report
    compared_trials = [v for v in validate_comparison if v["gold"] is not None]
    matches = sum(1 for v in compared_trials if v["match"])
    accuracy = matches / len(compared_trials) if compared_trials else 0.0

    comparison_report = {
        "patient_id": patient_id,
        "qrels_summary": {
            "eligible_count": len(eligible_ncts),
            "excluded_count": len(excluded_ncts),
            "not_relevant_count": len(not_relevant_ncts),
        },
        "prescreen_recall": {
            "eligible_found": sorted(eligible_found),
            "eligible_missed": sorted(eligible_missed),
            "recall": prescreen_recall,
        },
        "validate_comparison": validate_comparison,
        "validate_accuracy": accuracy,
        "validate_total_compared": len(compared_trials),
        "validate_matches": matches,
    }

    # Save comparison report
    from cache_manager import _topic_dir
    out_dir = _topic_dir(patient_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ground_truth_comparison.json"
    out_path.write_text(json.dumps(comparison_report, indent=2))
    logger.info("Saved ground truth comparison: %s", out_path)

    # Print summary
    print("\n" + "=" * 60)
    print(f"GROUND TRUTH SUMMARY for {patient_id}")
    print("=" * 60)
    print(f"  PRESCREEN recall (eligible): {prescreen_recall:.1%} "
          f"({len(eligible_found)}/{len(eligible_ncts)})")
    if eligible_missed:
        print(f"  Missed eligible trials: {sorted(eligible_missed)}")
    if compared_trials:
        print(f"  VALIDATE accuracy: {accuracy:.1%} ({matches}/{len(compared_trials)})")
    else:
        print("  VALIDATE accuracy: N/A (no overlapping trials evaluated)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def async_main(args: argparse.Namespace) -> None:
    """Run the full pipeline asynchronously."""
    patient_id = args.patient
    is_historical = patient_id.startswith("trec-")
    overall_start = time.perf_counter()

    print(f"\n{'=' * 60}")
    print(f"  Demo Cache Builder: {patient_id}")
    print(f"  Max validate trials: {args.max_validate_trials}")
    print(f"  Max criteria per trial: {args.max_criteria}")
    print(f"{'=' * 60}\n")

    # Load patient data
    profile = _load_patient(patient_id)

    if getattr(args, "validate_only", False):
        # Re-run VALIDATE only — load INGEST + PRESCREEN from cache
        logger.info("VALIDATE-ONLY mode: loading INGEST + PRESCREEN from cache")
        from cache_manager import load_ingest_result, load_prescreen_result
        cached_ingest = load_ingest_result(patient_id)
        if not cached_ingest:
            raise RuntimeError(f"No cached INGEST for {patient_id}. Run full pipeline first.")
        patient_note = cached_ingest["patient_note"]
        key_facts = cached_ingest["key_facts"]
        prescreen_result = load_prescreen_result(patient_id)
        if not prescreen_result:
            raise RuntimeError(f"No cached PRESCREEN for {patient_id}. Run full pipeline first.")
        logger.info("Loaded cached INGEST (%d key_facts) + PRESCREEN (%d candidates)",
                     len(key_facts), len(prescreen_result.candidates))
    else:
        # Phase 1: INGEST
        patient_note, key_facts = await run_ingest(patient_id, profile)

        # Phase 2: PRESCREEN
        prescreen_result = await run_prescreen(
            patient_id, patient_note, key_facts, is_historical=is_historical,
        )

    # Phase 3: VALIDATE
    trial_ids = [t.strip() for t in args.trial_ids.split(",") if t.strip()] if args.trial_ids else None
    validate_data = await run_validate(
        patient_id,
        patient_note,
        prescreen_result,
        max_validate_trials=args.max_validate_trials,
        max_criteria=args.max_criteria,
        trial_ids=trial_ids,
    )

    # Phase 4: Ground Truth (TREC only)
    if is_historical:
        run_ground_truth_comparison(patient_id, prescreen_result, validate_data)

    # Final summary
    overall_elapsed = time.perf_counter() - overall_start
    total_cost = (
        prescreen_result.gemini_estimated_cost
        + prescreen_result.medgemma_estimated_cost
    )

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE for {patient_id}")
    print(f"{'=' * 60}")
    print(f"  Total time: {overall_elapsed:.1f}s")
    print(f"  PRESCREEN: {len(prescreen_result.candidates)} candidates found")
    print(f"  PRESCREEN: {prescreen_result.total_api_calls} tool calls")
    print(f"  VALIDATE: {len(validate_data)} trials evaluated")
    for nct_id, trial_data in validate_data.items():
        verdict = trial_data["verdict"]
        n_criteria = len(trial_data.get("criteria", []))
        print(f"    {nct_id}: {verdict} ({n_criteria} criteria)")
    print(f"  Estimated cost: ~${total_cost:.4f} (PRESCREEN only, "
          f"VALIDATE cost not tracked)")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full e2e pipeline and generate demo cache files.",
    )
    parser.add_argument(
        "--patient",
        required=True,
        choices=VALID_PATIENTS,
        help="Patient ID to process",
    )
    parser.add_argument(
        "--max-validate-trials",
        type=int,
        default=3,
        help="Maximum number of top trials to validate (default: 3)",
    )
    parser.add_argument(
        "--max-criteria",
        type=int,
        default=10,
        help="Maximum criteria to evaluate per trial (default: 10)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Re-run only VALIDATE using existing INGEST+PRESCREEN cache",
    )
    parser.add_argument(
        "--trial-ids",
        type=str,
        default="",
        help="Comma-separated NCT IDs to validate (overrides top-N selection)",
    )
    parser.add_argument(
        "--log-file",
        default="/tmp/demo_cache.log",
        help="Log file path (default: /tmp/demo_cache.log)",
    )

    args = parser.parse_args()
    _setup_logging(args.log_file)

    logger.info("Starting demo cache pipeline for patient=%s", args.patient)
    logger.info("Config: max_validate_trials=%d, max_criteria=%d, log_file=%s",
                 args.max_validate_trials, args.max_criteria, args.log_file)

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
