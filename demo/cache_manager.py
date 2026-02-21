"""Cache manager for pipeline results -- enables reliable demo recording.

Saves and loads PRESCREEN and VALIDATE results so the demo can replay
from cached data without hitting live APIs. This ensures demo reliability
for video recording and presentations.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHED_RUNS_DIR = Path(__file__).resolve().parent / "data" / "cached_runs"


def save_prescreen_result(topic_id: str, result: Any) -> Path:
    """Save PresearchResult to cached_runs/{topic_id}/prescreen_result.json."""
    out_dir = CACHED_RUNS_DIR / topic_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prescreen_result.json"
    out_path.write_text(result.model_dump_json(indent=2))
    logger.info("Saved PRESCREEN cache: %s", out_path)
    return out_path


def load_prescreen_result(topic_id: str) -> Any | None:
    """Load PresearchResult from cache. Returns None if not found."""
    cached_path = CACHED_RUNS_DIR / topic_id / "prescreen_result.json"
    if not cached_path.exists():
        return None
    from trialmatch.prescreen.schema import PresearchResult

    return PresearchResult.model_validate_json(cached_path.read_text())


def save_validate_results(topic_id: str, validate_data: dict) -> Path:
    """Save VALIDATE results to cached_runs/{topic_id}/validate_results.json.

    validate_data format:
    {
        "nct_id": {
            "verdict": "ELIGIBLE|EXCLUDED|UNCERTAIN",
            "criteria": [
                {
                    "text": "criterion text",
                    "type": "inclusion|exclusion",
                    "verdict": "MET|NOT_MET|UNKNOWN",
                    "reasoning": "...",
                    "evidence_sentences": [0, 1]
                }
            ]
        }
    }
    """
    out_dir = CACHED_RUNS_DIR / topic_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validate_results.json"
    out_path.write_text(json.dumps(validate_data, indent=2))
    logger.info("Saved VALIDATE cache: %s", out_path)
    return out_path


def load_validate_results(topic_id: str) -> dict | None:
    """Load VALIDATE results from cache. Returns None if not found."""
    cached_path = CACHED_RUNS_DIR / topic_id / "validate_results.json"
    if not cached_path.exists():
        return None
    return json.loads(cached_path.read_text())


def save_ingest_result(topic_id: str, patient_note: str, key_facts: dict) -> Path:
    """Save INGEST (adapted key_facts) to cache."""
    out_dir = CACHED_RUNS_DIR / topic_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ingest.json"
    out_path.write_text(
        json.dumps({"patient_note": patient_note, "key_facts": key_facts}, indent=2)
    )
    logger.info("Saved INGEST cache: %s", out_path)
    return out_path


def list_cached_patients() -> list[str]:
    """List topic_ids that have cached results."""
    if not CACHED_RUNS_DIR.exists():
        return []
    return [d.name for d in sorted(CACHED_RUNS_DIR.iterdir()) if d.is_dir()]
