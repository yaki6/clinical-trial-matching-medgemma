"""Unit tests for demo/benchmark_loader.py pinned run validation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_LOADER_PATH = REPO_ROOT / "demo" / "benchmark_loader.py"


def _load_benchmark_loader_module():
    module_name = "demo_benchmark_loader"
    spec = importlib.util.spec_from_file_location(module_name, BENCHMARK_LOADER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_run(
    runs_dir: Path,
    run_id: str,
    seed: int,
    pairs: list[tuple[str, str, int]],
) -> None:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": 0.75,
                "f1_macro": 0.70,
                "f1_met_not_met": 0.80,
                "cohens_kappa": 0.60,
                "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "confusion_matrix_labels": ["MET", "NOT_MET", "UNKNOWN"],
            },
            indent=2,
        )
    )
    (run_dir / "cost_summary.json").write_text(
        json.dumps(
            {
                "total_pairs": len(pairs),
                "total_cost_usd": 0.01,
                "total_input_tokens": 100,
                "total_output_tokens": 10,
                "avg_latency_ms": 1000,
            },
            indent=2,
        )
    )
    (run_dir / "config.json").write_text(json.dumps({"data": {"seed": seed}}, indent=2))
    (run_dir / "results.json").write_text(
        json.dumps(
            [
                {"patient_id": p, "trial_id": t, "pair_index": idx}
                for p, t, idx in pairs
            ],
            indent=2,
        )
    )


def _write_pinned_config(path: Path, run_ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "runs": [{"run_id": run_id, "label": run_id} for run_id in run_ids],
            },
            indent=2,
        )
    )


def test_load_pinned_runs_success(tmp_path):
    module = _load_benchmark_loader_module()
    runs_dir = tmp_path / "runs"
    pairs = [("p1", "t1", 0), ("p2", "t2", 1)]
    run_ids = ["phase0-a", "phase0-b", "phase0-c"]
    for run_id in run_ids:
        _write_run(runs_dir, run_id, seed=42, pairs=pairs)
    pinned = tmp_path / "pinned_runs.json"
    _write_pinned_config(pinned, run_ids)

    loaded = module.load_pinned_benchmark_runs(runs_dir, pinned)
    assert len(loaded) == 3
    assert len({r.pair_set_hash for r in loaded}) == 1
    assert all(r.seed == 42 for r in loaded)


def test_load_pinned_runs_rejects_seed_mismatch(tmp_path):
    module = _load_benchmark_loader_module()
    runs_dir = tmp_path / "runs"
    pairs = [("p1", "t1", 0), ("p2", "t2", 1)]
    _write_run(runs_dir, "phase0-a", seed=42, pairs=pairs)
    _write_run(runs_dir, "phase0-b", seed=7, pairs=pairs)
    pinned = tmp_path / "pinned_runs.json"
    _write_pinned_config(pinned, ["phase0-a", "phase0-b"])

    with pytest.raises(ValueError, match="seed="):
        module.load_pinned_benchmark_runs(runs_dir, pinned)


def test_load_pinned_runs_rejects_pair_set_mismatch(tmp_path):
    module = _load_benchmark_loader_module()
    runs_dir = tmp_path / "runs"
    _write_run(runs_dir, "phase0-a", seed=42, pairs=[("p1", "t1", 0), ("p2", "t2", 1)])
    _write_run(runs_dir, "phase0-b", seed=42, pairs=[("p1", "t1", 0), ("p9", "t9", 1)])
    pinned = tmp_path / "pinned_runs.json"
    _write_pinned_config(pinned, ["phase0-a", "phase0-b"])

    with pytest.raises(ValueError, match="pair_set_hash"):
        module.load_pinned_benchmark_runs(runs_dir, pinned)
