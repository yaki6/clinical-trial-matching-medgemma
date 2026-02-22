"""Unit tests for live trace recorder."""

from __future__ import annotations

import json

from trialmatch.tracing.live_trace import create_live_trace_recorder


def test_create_live_trace_recorder_writes_run_started(tmp_path):
    recorder = create_live_trace_recorder(
        topic_id="mpx1016",
        pipeline_mode="live",
        validate_mode="Two-Stage (MedGemma -> Gemini)",
        dev_mode=True,
        base_dir=tmp_path,
    )

    assert recorder.path.exists()
    lines = recorder.path.read_text().strip().splitlines()
    assert len(lines) == 1

    event = json.loads(lines[0])
    assert event["event"] == "run_started"
    assert event["topic_id"] == "mpx1016"
    assert event["payload"]["pipeline_mode"] == "live"
    assert "trace_path" in event["payload"]


def test_record_appends_jsonl_entries(tmp_path):
    recorder = create_live_trace_recorder(
        topic_id="topic-1",
        pipeline_mode="live",
        validate_mode="Gemini 3 Pro (single)",
        dev_mode=False,
        base_dir=tmp_path,
    )
    recorder.record("tool_call_result", {"tool_name": "search_trials", "result": {"count": 2}})

    lines = recorder.path.read_text().strip().splitlines()
    assert len(lines) == 2

    event = json.loads(lines[1])
    assert event["event"] == "tool_call_result"
    assert event["payload"]["tool_name"] == "search_trials"
    assert event["payload"]["result"]["count"] == 2
