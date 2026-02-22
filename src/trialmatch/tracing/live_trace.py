"""Local trace recorder for Streamlit live runs.

Writes structured JSONL events so live executions can be audited and replayed
for model/prompt/tooling improvements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import structlog

logger = structlog.get_logger()

DEFAULT_LIVE_TRACES_DIR = (
    Path(__file__).resolve().parents[3] / "demo" / "data" / "live_traces"
)


def _json_safe(value: Any) -> Any:
    """Convert nested values into JSON-serializable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            return str(value)
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            return str(value)
    return str(value)


@dataclass(slots=True)
class LiveTraceRecorder:
    """Append-only JSONL trace writer."""

    run_id: str
    topic_id: str
    path: Path
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, event: str, payload: dict[str, Any] | None = None) -> None:
        """Append one event to the trace file. Never raises to caller."""
        entry = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "run_id": self.run_id,
            "topic_id": self.topic_id,
            "event": event,
            "payload": _json_safe(payload or {}),
        }
        line = json.dumps(entry, ensure_ascii=True)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            logger.warning(
                "live_trace_write_failed",
                run_id=self.run_id,
                topic_id=self.topic_id,
                path=str(self.path),
                event=event,
                exc_info=True,
            )


def create_live_trace_recorder(
    topic_id: str,
    pipeline_mode: str,
    validate_mode: str,
    dev_mode: bool,
    base_dir: Path | None = None,
) -> LiveTraceRecorder:
    """Create recorder and emit the initial run-start event."""
    root = base_dir or DEFAULT_LIVE_TRACES_DIR
    ts = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    run_id = f"live-{topic_id}-{ts}-{uuid4().hex[:8]}"
    path = root / topic_id / f"{run_id}.jsonl"

    recorder = LiveTraceRecorder(run_id=run_id, topic_id=topic_id, path=path)
    recorder.record(
        "run_started",
        {
            "pipeline_mode": pipeline_mode,
            "validate_mode": validate_mode,
            "dev_mode": dev_mode,
            "trace_path": str(path),
        },
    )
    logger.info(
        "live_trace_started",
        run_id=run_id,
        topic_id=topic_id,
        path=str(path),
        pipeline_mode=pipeline_mode,
        validate_mode=validate_mode,
        dev_mode=dev_mode,
    )
    return recorder
