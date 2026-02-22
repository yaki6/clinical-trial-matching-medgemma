"""Pinned benchmark loader with comparability checks for demo safety."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    results: list[dict] = field(default_factory=list, repr=False)
    audit_table: str | None = None


@dataclass(slots=True)
class AggregateStats:
    """Aggregate statistics across multiple multi-seed runs."""

    n: int
    v4_accuracy: float
    v4_f1_macro: float
    v4_kappa: float
    v4_confusion: list[list[int]]
    gpt4_accuracy: float
    gpt4_f1_macro: float
    gpt4_kappa: float
    gpt4_confusion: list[list[int]]
    labels: list[str]
    per_seed: list[dict]
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int


def _stable_pair_set_hash(results: list[dict]) -> str:
    keys = [
        f"{row.get('patient_id', '')}||{row.get('trial_id', '')}||{row.get('pair_index', '')}"
        for row in results
    ]
    payload = "\n".join(sorted(keys)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()  # nosec B324 - deterministic non-security checksum


def _load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def _compute_metrics_from_pairs(
    pairs: list[dict],
    pred_key: str,
    label_key: str = "expert_label",
) -> tuple[float, float, float, list[list[int]]]:
    """Compute accuracy, macro-F1, Cohen's kappa, and confusion matrix from pairs."""
    label_order = ["MET", "NOT_MET", "UNKNOWN"]
    label_to_idx = {lbl: i for i, lbl in enumerate(label_order)}

    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0, [[0] * 3 for _ in range(3)]

    cm = [[0] * 3 for _ in range(3)]
    correct = 0
    for pair in pairs:
        true = pair.get(label_key, "UNKNOWN")
        pred = pair.get(pred_key, "UNKNOWN")
        ti = label_to_idx.get(true, 2)
        pi = label_to_idx.get(pred, 2)
        cm[ti][pi] += 1
        if true == pred:
            correct += 1

    accuracy = correct / n

    # Per-class F1
    f1s = []
    for i in range(3):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(3)) - tp
        fn = sum(cm[i][j] for j in range(3)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    f1_macro = sum(f1s) / len(f1s)

    # Cohen's kappa
    total = sum(sum(row) for row in cm)
    if total == 0:
        kappa = 0.0
    else:
        po = correct / total
        pe = sum((sum(cm[i]) * sum(cm[j][i] for j in range(3))) for i in range(3)) / (total * total)
        kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    return accuracy, f1_macro, kappa, cm


def compute_aggregate_stats(runs: list[BenchmarkRunData]) -> AggregateStats:
    """Compute aggregate statistics across multi-seed runs."""
    all_pairs = []
    per_seed = []
    total_cost = 0.0
    total_input = 0
    total_output = 0

    for run in runs:
        all_pairs.extend(run.results)
        m = run.metrics
        seed = run.seed
        per_seed.append(
            {
                "seed": seed,
                "run_id": run.run_id,
                "v4_accuracy": m.get("accuracy", 0.0),
                "v4_f1_macro": m.get("f1_macro", 0.0),
                "v4_kappa": m.get("cohens_kappa", 0.0),
                "gpt4_accuracy": m.get("gpt4_baseline_accuracy", 0.0),
            }
        )
        c = run.cost
        total_cost += c.get("total_cost_usd", 0.0)
        total_input += c.get("total_input_tokens", 0)
        total_output += c.get("total_output_tokens", 0)

    n = len(all_pairs)
    labels = ["MET", "NOT_MET", "UNKNOWN"]

    v4_acc, v4_f1, v4_kappa, v4_cm = _compute_metrics_from_pairs(
        all_pairs, pred_key="model_verdict"
    )
    gpt4_acc, gpt4_f1, gpt4_kappa, gpt4_cm = _compute_metrics_from_pairs(
        all_pairs, pred_key="gpt4_label"
    )

    return AggregateStats(
        n=n,
        v4_accuracy=v4_acc,
        v4_f1_macro=v4_f1,
        v4_kappa=v4_kappa,
        v4_confusion=v4_cm,
        gpt4_accuracy=gpt4_acc,
        gpt4_f1_macro=gpt4_f1,
        gpt4_kappa=gpt4_kappa,
        gpt4_confusion=gpt4_cm,
        labels=labels,
        per_seed=per_seed,
        total_cost_usd=total_cost,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
    )


def load_pinned_benchmark_runs(
    runs_dir: Path,
    pinned_config_path: Path,
) -> list[BenchmarkRunData]:
    """Load pinned runs and enforce strict comparability invariants."""
    cfg = _load_json(pinned_config_path)
    schema_version = cfg.get("schema_version")
    if schema_version not in ("1", "2"):
        msg = f"Unsupported pinned_runs schema_version={schema_version!r}; expected '1' or '2'."
        raise ValueError(msg)

    multi_seed = cfg.get("multi_seed", False) and schema_version == "2"

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

        if not multi_seed:
            if expected_seed is None:
                expected_seed = seed
            elif seed != expected_seed:
                raise ValueError(
                    f"Run {run_id} has seed={seed}; "
                    f"expected seed={expected_seed} for comparability."
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
                results=results,
                audit_table=audit_table,
            )
        )

    return loaded
