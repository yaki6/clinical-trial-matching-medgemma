"""Cache manager for pipeline results -- enables reliable demo recording.

Saves and loads PRESCREEN and VALIDATE results so the demo can replay
from cached data without hitting live APIs. This ensures demo reliability
for video recording and presentations.
"""

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHED_RUNS_DIR = Path(__file__).resolve().parent / "data" / "cached_runs"
CACHE_MANIFEST_FILENAME = "manifest.json"
CACHE_SCHEMA_VERSION = "1"


@dataclass(slots=True)
class CachedRunManifest:
    """Versioned cache manifest for one topic_id."""

    topic_id: str
    prescreen_trial_ids: list[str]
    validated_trial_ids: list[str]
    validate_mode: str
    generated_at: str
    schema_version: str = CACHE_SCHEMA_VERSION


@dataclass(slots=True)
class CacheValidationReport:
    """Validation report for cached run artifacts."""

    topic_id: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    prescreen_candidate_count: int = 0
    validated_trial_count: int = 0
    manifest: CachedRunManifest | None = None


def _topic_dir(topic_id: str) -> Path:
    return CACHED_RUNS_DIR / topic_id


def _manifest_path(topic_id: str) -> Path:
    return _topic_dir(topic_id) / CACHE_MANIFEST_FILENAME


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def save_prescreen_result(topic_id: str, result: Any) -> Path:
    """Save PresearchResult to cached_runs/{topic_id}/prescreen_result.json."""
    out_dir = _topic_dir(topic_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prescreen_result.json"
    out_path.write_text(result.model_dump_json(indent=2))
    logger.info("Saved PRESCREEN cache: %s", out_path)
    return out_path


def load_prescreen_result(topic_id: str) -> Any | None:
    """Load PresearchResult from cache. Returns None if not found."""
    cached_path = _topic_dir(topic_id) / "prescreen_result.json"
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
    out_dir = _topic_dir(topic_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validate_results.json"
    out_path.write_text(json.dumps(validate_data, indent=2))
    logger.info("Saved VALIDATE cache: %s", out_path)
    return out_path


def load_validate_results(topic_id: str) -> dict | None:
    """Load VALIDATE results from cache. Returns None if not found."""
    cached_path = _topic_dir(topic_id) / "validate_results.json"
    if not cached_path.exists():
        return None
    return json.loads(cached_path.read_text())


def save_ingest_result(topic_id: str, patient_note: str, key_facts: dict) -> Path:
    """Save INGEST (adapted key_facts) to cache."""
    out_dir = _topic_dir(topic_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ingest.json"
    out_path.write_text(
        json.dumps({"patient_note": patient_note, "key_facts": key_facts}, indent=2)
    )
    logger.info("Saved INGEST cache: %s", out_path)
    return out_path


def load_ingest_result(topic_id: str) -> dict | None:
    """Load INGEST (patient_note + key_facts) from cache. Returns None if not found."""
    cached_path = _topic_dir(topic_id) / "ingest.json"
    if not cached_path.exists():
        return None
    return json.loads(cached_path.read_text())


def save_cached_manifest(
    topic_id: str,
    prescreen_trial_ids: list[str],
    validated_trial_ids: list[str],
    validate_mode: str,
    generated_at: str | None = None,
) -> Path:
    """Save cache manifest for one topic_id."""
    manifest = CachedRunManifest(
        topic_id=topic_id,
        prescreen_trial_ids=sorted(set(prescreen_trial_ids)),
        validated_trial_ids=sorted(set(validated_trial_ids)),
        validate_mode=validate_mode,
        generated_at=generated_at or _utc_now_iso(),
    )
    out_dir = _topic_dir(topic_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / CACHE_MANIFEST_FILENAME
    out_path.write_text(json.dumps(asdict(manifest), indent=2))
    logger.info("Saved cache manifest: %s", out_path)
    return out_path


def load_cached_manifest(topic_id: str) -> CachedRunManifest | None:
    """Load cache manifest for one topic_id. Returns None if not found."""
    path = _manifest_path(topic_id)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return CachedRunManifest(**raw)


def validate_cached_run(topic_id: str) -> CacheValidationReport:
    """Validate cached artifacts and trial-id consistency for one topic_id."""
    report = CacheValidationReport(topic_id=topic_id, valid=False)
    topic_dir = _topic_dir(topic_id)
    ingest_path = topic_dir / "ingest.json"
    prescreen_path = topic_dir / "prescreen_result.json"
    validate_path = topic_dir / "validate_results.json"
    manifest_path = topic_dir / CACHE_MANIFEST_FILENAME

    if not topic_dir.exists():
        report.errors.append(f"Cache directory does not exist: {topic_dir}")
        return report

    for p in (ingest_path, prescreen_path, validate_path, manifest_path):
        if not p.exists():
            report.errors.append(f"Missing cache artifact: {p.name}")

    prescreen_ids: set[str] = set()
    validate_ids: set[str] = set()

    if prescreen_path.exists():
        try:
            from trialmatch.prescreen.schema import PresearchResult

            prescreen = PresearchResult.model_validate_json(prescreen_path.read_text())
            prescreen_ids = {c.nct_id for c in prescreen.candidates}
            report.prescreen_candidate_count = len(prescreen_ids)
        except Exception as exc:
            report.errors.append(f"Invalid prescreen_result.json: {exc}")

    if validate_path.exists():
        try:
            validate_data = json.loads(validate_path.read_text())
            if not isinstance(validate_data, dict):
                raise TypeError("validate_results.json must be a dict keyed by nct_id")
            validate_ids = set(validate_data.keys())
            report.validated_trial_count = len(validate_ids)
        except Exception as exc:
            report.errors.append(f"Invalid validate_results.json: {exc}")

    orphan_validate_ids = validate_ids - prescreen_ids
    if orphan_validate_ids:
        report.errors.append(
            "validate_results.json contains trial ids not present in prescreen_result.json: "
            + ", ".join(sorted(orphan_validate_ids))
        )

    if manifest_path.exists():
        try:
            manifest = load_cached_manifest(topic_id)
            if manifest is None:
                raise ValueError("manifest file is empty")
            report.manifest = manifest
            if manifest.schema_version != CACHE_SCHEMA_VERSION:
                report.errors.append(
                    "Unsupported manifest schema_version: "
                    f"{manifest.schema_version} (expected {CACHE_SCHEMA_VERSION})"
                )
            manifest_prescreen = set(manifest.prescreen_trial_ids)
            manifest_validate = set(manifest.validated_trial_ids)
            if manifest_prescreen != prescreen_ids:
                report.errors.append(
                    "Manifest prescreen_trial_ids do not match prescreen_result.json"
                )
            if manifest_validate != validate_ids:
                report.errors.append(
                    "Manifest validated_trial_ids do not match validate_results.json"
                )
            if not manifest.validate_mode:
                report.warnings.append("Manifest validate_mode is empty.")
        except Exception as exc:
            report.errors.append(f"Invalid {CACHE_MANIFEST_FILENAME}: {exc}")

    report.valid = not report.errors
    return report


def list_cached_patients() -> list[str]:
    """List topic_ids that have cached results."""
    if not CACHED_RUNS_DIR.exists():
        return []
    return [d.name for d in sorted(CACHED_RUNS_DIR.iterdir()) if d.is_dir()]
