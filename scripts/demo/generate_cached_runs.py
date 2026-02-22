"""Generate cached demo artifacts via real live E2E runs.

This script intentionally performs real API calls:
- MedGemma (PRESCREEN guidance + VALIDATE stage-1 reasoning)
- Gemini (PRESCREEN orchestration + VALIDATE stage-2 labeling)
- ClinicalTrials.gov API (PRESCREEN search + trial details in VALIDATE)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DEMO_ROOT = REPO_ROOT / "demo"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(DEMO_ROOT))

from cache_manager import (  # noqa: E402
    save_cached_manifest,
    save_ingest_result,
    save_prescreen_result,
    save_validate_results,
    validate_cached_run,
)
from trialmatch.ingest.profile_adapter import adapt_harness_patient, load_demo_harness  # noqa: E402
from trialmatch.live_runtime import (  # noqa: E402
    VALIDATE_MODE_TWO_STAGE,
    create_imaging_adapter,
    create_prescreen_adapters,
    create_validate_adapters,
    failed_preflight_checks,
    run_live_preflight,
)
from trialmatch.prescreen.agent import run_prescreen_agent  # noqa: E402
from trialmatch.prescreen.ctgov_client import CTGovClient  # noqa: E402
from trialmatch.validate.evaluator import evaluate_criterion_two_stage  # noqa: E402

CURATED_TOPIC_IDS = ["mpx1016", "mpx1575", "mpx1875"]
IMAGE_CACHE_DIR = REPO_ROOT / "data" / "sot" / "ingest" / "medgemma_image_cache"

PATIENT_IMAGE_PROMPTS = {
    "mpx1016": (
        "What abnormalities do you see in this chest CT scan? "
        "Focus on lung parenchyma, pleural space, and any masses or nodules. "
        "Describe location, size, and characteristics."
    ),
    "mpx1575": (
        "Analyze this CT image. Identify any pulmonary masses, fibrotic changes, "
        "honeycombing, pleural effusions, or lymphadenopathy. "
        "Describe findings systematically."
    ),
    "mpx1875": (
        "Is there a lung mass visible in this image? Describe its location, size, "
        "margins, and density. Note any mediastinal or hilar abnormalities."
    ),
}
FALLBACK_IMAGE_PROMPT = (
    "Describe the key abnormalities visible in this medical image. "
    "Focus on lung findings, masses, and any pathology."
)


def parse_eligibility_criteria(criteria_text: str) -> list[dict]:
    """Split eligibility criteria text into criterion dicts."""
    criteria = []
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
            line = line.lstrip("*-~ ").strip()
        if len(line) > 10:
            criteria.append({"text": line, "type": current_type})

    return criteria


def compute_trial_verdict(results: list[tuple[dict, Any]]) -> str:
    """Compute trial-level verdict from criterion results."""
    has_unknown = False
    for criterion, cr in results:
        verdict_val = cr.verdict.value
        ctype = criterion["type"]

        if ctype == "exclusion" and verdict_val == "NOT_MET":
            continue
        if ctype == "exclusion" and verdict_val == "MET":
            return "EXCLUDED"
        if ctype == "inclusion" and verdict_val == "NOT_MET":
            return "EXCLUDED"
        if verdict_val == "UNKNOWN":
            has_unknown = True

    if has_unknown:
        return "UNCERTAIN"
    return "ELIGIBLE"


def _save_image_cache(topic_id: str, payload: dict) -> Path:
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = IMAGE_CACHE_DIR / f"{topic_id}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


async def _extract_image_findings(topic_id: str, patient: dict, imaging_adapter: Any) -> dict | None:
    image = patient.get("image")
    if not image:
        return None

    image_rel = image.get("file_path")
    if not image_rel:
        return None

    image_path = REPO_ROOT / image_rel
    if not image_path.exists():
        raise RuntimeError(f"Image file missing for {topic_id}: {image_path}")

    prompt = PATIENT_IMAGE_PROMPTS.get(topic_id, FALLBACK_IMAGE_PROMPT)
    response = await imaging_adapter.generate_with_image(prompt, image_path, max_tokens=512)
    findings = {
        "topic_id": topic_id,
        "image_path": image_rel,
        "prompt": prompt,
        "raw_response": response.text,
        "extracted_text": response.text,
        "latency_seconds": round(response.latency_ms / 1000, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": imaging_adapter.name,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "estimated_cost": response.estimated_cost,
    }
    _save_image_cache(topic_id, findings)
    return findings


def _build_validate_cache_entry(criterion: dict, result: Any) -> dict:
    entry = {
        "text": criterion["text"],
        "type": criterion["type"],
        "verdict": result.verdict.value,
        "reasoning": result.reasoning,
        "evidence_sentences": result.evidence_sentences,
    }
    if result.stage1_reasoning:
        entry["stage1_reasoning"] = result.stage1_reasoning
    return entry


async def _run_live_for_patient(
    topic_id: str,
    patient: dict,
    imaging_adapter: Any,
    gemini_adapter: Any,
    medgemma_adapter: Any,
    reasoning_adapter: Any,
    labeling_adapter: Any,
    max_trials: int,
    max_criteria: int,
) -> tuple[int, int]:
    image_findings = await _extract_image_findings(topic_id, patient, imaging_adapter)
    patient_note, key_facts = adapt_harness_patient(patient, image_findings=image_findings)
    save_ingest_result(topic_id, patient_note, key_facts)

    prescreen = await run_prescreen_agent(
        patient_note=patient_note,
        key_facts=key_facts,
        ingest_source="gold",
        gemini_adapter=gemini_adapter,
        medgemma_adapter=medgemma_adapter,
        require_clinical_guidance=True,
        topic_id=topic_id,
    )
    save_prescreen_result(topic_id, prescreen)

    validate_cache: dict[str, dict] = {}
    ctgov = CTGovClient()
    try:
        for trial in prescreen.candidates[:max_trials]:
            raw_details = await ctgov.get_details(trial.nct_id)
            protocol = raw_details.get("protocolSection", {})
            eligibility = protocol.get("eligibilityModule", {})
            criteria_text = eligibility.get("eligibilityCriteria", "")
            if not criteria_text:
                continue

            criteria = parse_eligibility_criteria(criteria_text)
            if not criteria:
                continue

            criterion_results = []
            for criterion in criteria[:max_criteria]:
                result = await evaluate_criterion_two_stage(
                    patient_note=patient_note,
                    criterion_text=criterion["text"],
                    criterion_type=criterion["type"],
                    reasoning_adapter=reasoning_adapter,
                    labeling_adapter=labeling_adapter,
                )
                criterion_results.append((criterion, result))

            if not criterion_results:
                continue

            validate_cache[trial.nct_id] = {
                "verdict": compute_trial_verdict(criterion_results),
                "mode": "two_stage",
                "criteria": [
                    _build_validate_cache_entry(criterion, result)
                    for criterion, result in criterion_results
                ],
            }
    finally:
        await ctgov.aclose()

    save_validate_results(topic_id, validate_cache)
    save_cached_manifest(
        topic_id=topic_id,
        prescreen_trial_ids=[c.nct_id for c in prescreen.candidates],
        validated_trial_ids=list(validate_cache.keys()),
        validate_mode="two_stage",
    )

    report = validate_cached_run(topic_id)
    if not report.valid:
        msg = f"Cache invalid after live run for {topic_id}: {'; '.join(report.errors)}"
        raise RuntimeError(msg)

    return report.prescreen_candidate_count, report.validated_trial_count


async def main_async(topic_ids: list[str], max_trials: int, max_criteria: int) -> None:
    load_dotenv(REPO_ROOT / ".env")
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    if not google_api_key:
        raise RuntimeError("GOOGLE_API_KEY must be set (or present in .env).")

    patients = load_demo_harness()
    by_topic = {p["topic_id"]: p for p in patients}
    missing = [tid for tid in topic_ids if tid not in by_topic]
    if missing:
        raise RuntimeError(f"Missing patients in demo harness: {', '.join(missing)}")

    gemini_adapter, medgemma_adapter = create_prescreen_adapters(
        api_key=google_api_key,
        hf_token=hf_token,
    )
    imaging_adapter = create_imaging_adapter(hf_token=hf_token)
    imaging_ok = await imaging_adapter.health_check()
    if not imaging_ok:
        raise RuntimeError(
            "Live preflight failed: MedGemma imaging adapter is not reachable. "
            "Check VERTEX_ENDPOINT_ID deployment or HF endpoint health."
        )
    preflight = await run_live_preflight(
        gemini_adapter=gemini_adapter,
        medgemma_adapter=medgemma_adapter,
        include_ctgov=True,
    )
    failed = failed_preflight_checks(preflight)
    if failed:
        details = ", ".join(f"{r.name}: {r.detail}" for r in failed)
        raise RuntimeError(f"Live preflight failed: {details}")

    reasoning_adapter, labeling_adapter = create_validate_adapters(
        VALIDATE_MODE_TWO_STAGE,
        api_key=google_api_key,
        hf_token=hf_token,
    )
    print(
        "[live] adapters: "
        f"imaging={imaging_adapter.name}, "
        f"prescreen_medgemma={medgemma_adapter.name}, "
        f"validate_reasoning={reasoning_adapter.name}, "
        f"validate_labeling={labeling_adapter.name if labeling_adapter else 'none'}"
    )

    for topic_id in topic_ids:
        print(f"[live] running full E2E for {topic_id}...")
        prescreen_count, validated_count = await _run_live_for_patient(
            topic_id=topic_id,
            patient=by_topic[topic_id],
            imaging_adapter=imaging_adapter,
            gemini_adapter=gemini_adapter,
            medgemma_adapter=medgemma_adapter,
            reasoning_adapter=reasoning_adapter,
            labeling_adapter=labeling_adapter,
            max_trials=max_trials,
            max_criteria=max_criteria,
        )
        print(
            f"[live] {topic_id}: prescreen_candidates={prescreen_count} "
            f"validated_trials={validated_count}"
        )

    print("Live cache generation complete for curated patients.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topics",
        nargs="+",
        default=CURATED_TOPIC_IDS,
        help="topic_ids to run live E2E for",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=3,
        help="max PRESCREEN candidates to validate per patient",
    )
    parser.add_argument(
        "--max-criteria",
        type=int,
        default=10,
        help="max criteria per trial during VALIDATE",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(
        main_async(
            topic_ids=args.topics,
            max_trials=args.max_trials,
            max_criteria=args.max_criteria,
        )
    )


if __name__ == "__main__":
    main()
