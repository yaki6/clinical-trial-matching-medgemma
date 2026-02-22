"""Pinned benchmark loader with comparability checks for demo safety."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class BenchmarkRunData:
    """Loaded benchmark run artifacts for one pinned run."""

    run_id: str
    label: str
    path: Path
    seed: int | None
    pair_set_hash: str
    metrics: dict
    cost: dict
    config: dict
    audit_table: str | None = None


def _stable_pair_set_hash(results: list[dict]) -> str:
    keys = [
        f"{row.get('patient_id', '')}||{row.get('trial_id', '')}||{row.get('pair_index', '')}"
        for row in results
    ]
    payload = "\n".join(sorted(keys)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()  # nosec B324 - deterministic non-security checksum


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_pinned_benchmark_runs(
    runs_dir: Path,
    pinned_config_path: Path,
) -> list[BenchmarkRunData]:
    """Load pinned runs and enforce strict comparability invariants."""
    cfg = _load_json(pinned_config_path)
    if cfg.get("schema_version") != "1":
        msg = (
            f"Unsupported pinned_runs schema_version={cfg.get('schema_version')!r}; "
            "expected '1'."
        )
        raise ValueError(msg)

    entries = cfg.get("runs", [])
    if not entries:
        raise ValueError("Pinned benchmark config has no runs.")

    loaded: list[BenchmarkRunData] = []
    expected_seed: int | None = None
    expected_pair_hash: str | None = None

    for entry in entries:
        run_id = entry["run_id"]
        label = entry.get("label", run_id)
        run_path = runs_dir / run_id
        if not run_path.exists():
            raise FileNotFoundError(f"Pinned run not found: {run_id}")

        metrics_path = run_path / "metrics.json"
        config_path = run_path / "config.json"
        results_path = run_path / "results.json"
        cost_path = run_path / "cost_summary.json"

        for required_path in (metrics_path, config_path, results_path, cost_path):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Run {run_id} missing required artifact: {required_path.name}"
                )

        metrics = _load_json(metrics_path)
        config = _load_json(config_path)
        results = _load_json(results_path)
        cost = _load_json(cost_path)

        if not isinstance(results, list):
            raise ValueError(f"Run {run_id} has invalid results.json; expected a list.")

        seed = config.get("data", {}).get("seed")
        pair_hash = _stable_pair_set_hash(results)

        if expected_seed is None:
            expected_seed = seed
        elif seed != expected_seed:
            raise ValueError(
                f"Run {run_id} has seed={seed}; expected seed={expected_seed} for comparability."
            )

        if expected_pair_hash is None:
            expected_pair_hash = pair_hash
        elif pair_hash != expected_pair_hash:
            raise ValueError(
                f"Run {run_id} has pair_set_hash={pair_hash}; "
                f"expected {expected_pair_hash} for comparability."
            )

        audit_path = run_path / "audit_table.md"
        audit_table = audit_path.read_text() if audit_path.exists() else None

        loaded.append(
            BenchmarkRunData(
                run_id=run_id,
                label=label,
                path=run_path,
                seed=seed,
                pair_set_hash=pair_hash,
                metrics=metrics,
                cost=cost,
                config=config,
                audit_table=audit_table,
            )
        )

    return loaded
