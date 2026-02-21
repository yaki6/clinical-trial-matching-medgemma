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
