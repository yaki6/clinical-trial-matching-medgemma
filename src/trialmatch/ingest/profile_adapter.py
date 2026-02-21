"""Adapter: nsclc_trial_profiles.json -> PRESCREEN input format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Default path relative to repo root
_DEFAULT_PROFILES_PATH = Path(__file__).resolve().parents[3] / "nsclc_trial_profiles.json"


def adapt_profile_for_prescreen(profile: dict) -> tuple[str, dict[str, Any]]:
    """Convert a single profile from nsclc_trial_profiles.json to PRESCREEN input.

    Returns (patient_note, key_facts_dict) where key_facts_dict is a flat dict
    compatible with run_prescreen_agent()'s key_facts parameter.
    """
    patient_note = profile["profile_text"]
    raw_facts = profile.get("key_facts", [])

    key_facts: dict[str, Any] = {}
    for kf in raw_facts:
        field = kf["field"]
        value = kf["value"]

        if isinstance(value, dict):
            # Flatten nested dict to readable string
            parts = []
            for k, v in value.items():
                if isinstance(v, list):
                    parts.append(f"{k}: {', '.join(str(i) for i in v)}")
                elif v is not None:
                    parts.append(f"{k}: {v}")
            key_facts[field] = "; ".join(parts)
        else:
            # String, list, or other -- pass through as-is
            key_facts[field] = value

    return patient_note, key_facts


def load_profiles(path: Path | str | None = None) -> list[dict]:
    """Load nsclc_trial_profiles.json, return list of profile dicts."""
    p = Path(path) if path else _DEFAULT_PROFILES_PATH
    with open(p) as f:
        data = json.load(f)
    return data.get("profiles", data if isinstance(data, list) else [])


# Default path for demo harness
_DEFAULT_HARNESS_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "sot" / "ingest" / "nsclc_demo_harness.json"
)


def load_demo_harness(path: Path | str | None = None) -> list[dict]:
    """Load nsclc_demo_harness.json — 5 curated patients."""
    p = Path(path) if path else _DEFAULT_HARNESS_PATH
    with open(p) as f:
        data = json.load(f)
    return data.get("patients", [])


def get_image_path(patient: dict, base_dir: Path | str | None = None) -> Path | None:
    """Return resolved image path for multimodal patients, None for text-only.

    base_dir defaults to repo root (3 levels up from this file).
    """
    if not patient.get("image"):
        return None
    base = Path(base_dir) if base_dir else Path(__file__).resolve().parents[3]
    return base / patient["image"]["file_path"]


def merge_image_findings(key_facts: dict[str, Any], image_findings: dict) -> dict[str, Any]:
    """Merge MedGemma image extraction results into key_facts dict.

    Adds 'medgemma_imaging' key with findings, impression, modality.
    Does not overwrite existing facts.
    """
    merged = dict(key_facts)
    merged["medgemma_imaging"] = image_findings
    return merged


def adapt_harness_patient(
    patient: dict, image_findings: dict | None = None
) -> tuple[str, dict[str, Any]]:
    """Adapt harness patient for PRESCREEN.

    Returns (patient_note, key_facts).
    Reuses adapt_profile_for_prescreen() internally.
    If image_findings provided, merges into key_facts.
    """
    patient_note, key_facts = adapt_profile_for_prescreen(patient)
    if image_findings:
        key_facts = merge_image_findings(key_facts, image_findings)
    return patient_note, key_facts
