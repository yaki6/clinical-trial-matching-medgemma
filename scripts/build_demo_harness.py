#!/usr/bin/env python3
"""Build the NSCLC demo harness JSON with 5 curated patients.

Reads:
  - ingest_design/nsclc-dataset/nsclc_dataset.jsonl  (37 MedPix + TREC patients)
  - ingest_design/nsclc-dataset/nsclc_trial_profiles.json  (Gemini-structured profiles)

Writes:
  - data/sot/ingest/nsclc_demo_harness.json

Run from repo root:
  uv run python scripts/build_demo_harness.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration: 5 curated patients
# ---------------------------------------------------------------------------

MULTIMODAL_PATIENTS = [
    {
        "topic_id": "mpx1016",
        "jsonl_uid": "MPX1016",
        "source_dataset": "MedPix",
        "image_file": "ingest_design/MedPix-2-0/images/MPX1016_synpic34317.png",
    },
    {
        "topic_id": "mpx1575",
        "jsonl_uid": "MPX1575",
        "source_dataset": "MedPix",
        "image_file": "ingest_design/MedPix-2-0/images/MPX1575_synpic39398.png",
    },
    {
        "topic_id": "mpx1875",
        "jsonl_uid": "MPX1875",
        "source_dataset": "MedPix",
        "image_file": "ingest_design/MedPix-2-0/images/MPX1875_synpic23054.png",
    },
]

TEXT_ONLY_PATIENTS = [
    {
        "topic_id": "6031552-1",
        "jsonl_uid": "6031552-1",
        "source_dataset": "PMC-Patients",
    },
    {
        "topic_id": "6000873-1",
        "jsonl_uid": "6000873-1",
        "source_dataset": "PMC-Patients",
    },
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
JSONL_PATH = REPO_ROOT / "ingest_design" / "nsclc-dataset" / "nsclc_dataset.jsonl"
PROFILES_PATH = REPO_ROOT / "ingest_design" / "nsclc-dataset" / "nsclc_trial_profiles.json"
OUTPUT_PATH = REPO_ROOT / "data" / "sot" / "ingest" / "nsclc_demo_harness.json"


def load_jsonl_index(path: Path) -> dict[str, dict]:
    """Load JSONL file and index rows by uid."""
    index: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = row.get("uid", "")
            if uid in index:
                print(f"  WARNING: duplicate uid '{uid}' at line {line_no}")
            index[uid] = row
    return index


def load_profiles_index(path: Path) -> dict[str, dict]:
    """Load profiles JSON and index by topic_id."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    profiles = data.get("profiles", [])
    return {p["topic_id"]: p for p in profiles}


def build_ehr_text_medpix(row: dict) -> str:
    """Build EHR text for MedPix patients: history + exam + findings."""
    parts = []
    if row.get("history"):
        parts.append(row["history"])
    if row.get("exam"):
        parts.append(row["exam"])
    if row.get("findings"):
        parts.append(row["findings"])
    return "\n\n".join(parts)


def build_ehr_text_text_only(row: dict) -> str:
    """Build EHR text for text-only (PMC-Patients) patients.

    The 'findings' field in PMC-Patients rows contains metadata (related
    articles, similar patients), not clinical findings.  We use only
    'history' and 'exam' for the EHR text.  The 'history' field for these
    cases is the full case report text with all clinical detail.
    """
    parts = []
    if row.get("history"):
        parts.append(row["history"])
    if row.get("exam"):
        # exam field for TREC cases is minimal metadata (age, gender, PMID)
        # Include it for completeness but it adds little clinical content
        parts.append(row["exam"])
    return "\n\n".join(parts)


def build_image_entry(row: dict, expected_file: str) -> dict | None:
    """Extract image metadata for the first image from a MedPix JSONL row."""
    images = row.get("images", [])
    if not images:
        return None

    # Find the image matching the expected file path
    img = None
    for candidate in images:
        candidate_path = candidate.get("file_path", "")
        # JSONL stores paths like "MedPix-2-0/images/MPX1016_synpic34317.png"
        # expected_file is "ingest_design/MedPix-2-0/images/MPX1016_synpic34317.png"
        if expected_file.endswith(candidate_path):
            img = candidate
            break

    if img is None:
        # Fall back to first image
        img = images[0]
        print(f"  WARNING: expected image '{expected_file}' not found, using first image")

    return {
        "file_path": f"ingest_design/{img['file_path']}",
        "modality": img.get("modality", ""),
        "plane": img.get("plane", ""),
        "location": img.get("location", ""),
        "caption": img.get("caption", ""),
    }


def build_patient_entry(
    *,
    topic_id: str,
    source_dataset: str,
    ingest_mode: str,
    jsonl_row: dict,
    profile: dict,
    image_file: str | None = None,
) -> dict:
    """Assemble a single patient entry for the demo harness."""
    # Build EHR text
    if source_dataset == "MedPix":
        ehr_text = build_ehr_text_medpix(jsonl_row)
    else:
        ehr_text = build_ehr_text_text_only(jsonl_row)

    # Image
    image_entry = None
    if ingest_mode == "multimodal" and image_file:
        image_entry = build_image_entry(jsonl_row, image_file)

    # Diagnosis: prefer JSONL diagnosis, fall back to title
    diagnosis = jsonl_row.get("diagnosis", "") or jsonl_row.get("title", "")

    return {
        "topic_id": topic_id,
        "source_dataset": source_dataset,
        "ingest_mode": ingest_mode,
        "ehr_text": ehr_text,
        "profile_text": profile.get("profile_text", ""),
        "key_facts": profile.get("key_facts", []),
        "image": image_entry,
        "medgemma_image_findings": None,
        "diagnosis": diagnosis,
    }


def main() -> None:
    print("=== Building NSCLC Demo Harness ===\n")

    # Validate source files exist
    for path, label in [(JSONL_PATH, "JSONL"), (PROFILES_PATH, "Profiles")]:
        if not path.exists():
            print(f"ERROR: {label} file not found: {path}")
            sys.exit(1)
        print(f"  {label}: {path}")

    # Load data
    print("\nLoading source data...")
    jsonl_index = load_jsonl_index(JSONL_PATH)
    print(f"  JSONL: {len(jsonl_index)} patients loaded")

    profiles_index = load_profiles_index(PROFILES_PATH)
    print(f"  Profiles: {len(profiles_index)} profiles loaded")

    # Assemble patients
    patients = []
    errors = []

    print("\nAssembling multimodal patients:")
    for spec in MULTIMODAL_PATIENTS:
        tid = spec["topic_id"]
        uid = spec["jsonl_uid"]

        if uid not in jsonl_index:
            errors.append(f"  MISSING in JSONL: uid='{uid}' for topic_id='{tid}'")
            continue
        if tid not in profiles_index:
            errors.append(f"  MISSING in profiles: topic_id='{tid}'")
            continue

        patient = build_patient_entry(
            topic_id=tid,
            source_dataset=spec["source_dataset"],
            ingest_mode="multimodal",
            jsonl_row=jsonl_index[uid],
            profile=profiles_index[tid],
            image_file=spec["image_file"],
        )
        patients.append(patient)

        diag_short = patient["diagnosis"][:60]
        img_status = "with image" if patient["image"] else "NO IMAGE"
        print(f"  [{tid}] {diag_short}... ({img_status})")

    print("\nAssembling text-only patients:")
    for spec in TEXT_ONLY_PATIENTS:
        tid = spec["topic_id"]
        uid = spec["jsonl_uid"]

        if uid not in jsonl_index:
            errors.append(f"  MISSING in JSONL: uid='{uid}' for topic_id='{tid}'")
            continue
        if tid not in profiles_index:
            errors.append(f"  MISSING in profiles: topic_id='{tid}'")
            continue

        patient = build_patient_entry(
            topic_id=tid,
            source_dataset=spec["source_dataset"],
            ingest_mode="text",
            jsonl_row=jsonl_index[uid],
            profile=profiles_index[tid],
        )
        patients.append(patient)

        diag_short = patient["diagnosis"][:60] if patient["diagnosis"] else "(no diagnosis)"
        print(f"  [{tid}] {diag_short}")

    # Report errors
    if errors:
        print("\nERRORS:")
        for err in errors:
            print(err)
        sys.exit(1)

    # Build output
    output = {
        "version": "2.0",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": "5 curated NSCLC patients for MedGemma Impact Challenge demo",
        "patients": patients,
    }

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nOutput written to: {OUTPUT_PATH}")
    print(f"  {len(patients)} patients ({sum(1 for p in patients if p['ingest_mode'] == 'multimodal')} multimodal, {sum(1 for p in patients if p['ingest_mode'] == 'text')} text-only)")

    # Summary table
    print("\n--- Summary ---")
    print(f"{'topic_id':<15} {'mode':<12} {'source':<15} {'ehr_len':>8} {'key_facts':>10} {'image':>6}")
    print("-" * 72)
    for p in patients:
        print(
            f"{p['topic_id']:<15} "
            f"{p['ingest_mode']:<12} "
            f"{p['source_dataset']:<15} "
            f"{len(p['ehr_text']):>8} "
            f"{len(p['key_facts']):>10} "
            f"{'yes' if p['image'] else 'no':>6}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
