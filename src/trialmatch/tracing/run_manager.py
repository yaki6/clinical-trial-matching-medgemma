"""Run manager for saving benchmark results and cost tracking.

Each run saves to runs/<run_id>/ with:
  config.json, metrics.json, results.json, cost_summary.json, audit_table.md
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from trialmatch.models.schema import CriterionAnnotation, RunResult

logger = structlog.get_logger()

RUNS_DIR = Path("runs")


class RunManager:
    """Manages benchmark run artifacts."""

    def __init__(self, runs_dir: Path = RUNS_DIR):
        self._runs_dir = runs_dir

    def generate_run_id(self, model_name: str) -> str:
        """Generate a unique run ID."""
        ts = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
        return f"phase0-{model_name}-{ts}"

    def save_run(
        self,
        run_result: RunResult,
        config: dict | None = None,
        annotations: list[CriterionAnnotation] | None = None,
    ) -> Path:
        """Save run artifacts to runs/<run_id>/.

        Args:
            run_result: Model predictions and metrics.
            config: Pipeline config (saved as config.json).
            annotations: Original CriterionAnnotation objects aligned with
                run_result.results. When provided, each result row is enriched
                with patient_id, trial_id, criterion metadata, expert labels,
                and a 'correct' boolean for auditing.
        """
        run_dir = self._runs_dir / run_result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        if config:
            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2, default=str)

        # Save metrics
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(run_result.metrics, f, indent=2, default=str)

        # Build per-pair results with full identity + audit fields
        results_data = []
        for i, r in enumerate(run_result.results):
            ann = annotations[i] if annotations and i < len(annotations) else None
            row: dict = {
                "pair_index": i,
            }
            # Identity fields (from annotation if available)
            if ann is not None:
                row["patient_id"] = ann.patient_id
                row["trial_id"] = ann.trial_id
                row["criterion_type"] = ann.criterion_type
                row["criterion_text"] = ann.criterion_text
                row["expert_label"] = ann.expert_label.value
                row["gpt4_label"] = ann.gpt4_label.value
            # Model prediction
            row["model_verdict"] = r.verdict.value
            if ann is not None:
                row["correct"] = r.verdict == ann.expert_label
            row["reasoning"] = r.reasoning
            row["evidence_sentences"] = r.evidence_sentences
            # Cost/latency
            row["input_tokens"] = r.model_response.input_tokens
            row["output_tokens"] = r.model_response.output_tokens
            row["latency_ms"] = r.model_response.latency_ms
            row["estimated_cost"] = r.model_response.estimated_cost
            row["token_count_estimated"] = r.model_response.token_count_estimated
            results_data.append(row)

        with open(run_dir / "results.json", "w") as f:
            json.dump(results_data, f, indent=2)

        # Save cost summary
        total_cost = sum(r.model_response.estimated_cost for r in run_result.results)
        total_input = sum(r.model_response.input_tokens for r in run_result.results)
        total_output = sum(r.model_response.output_tokens for r in run_result.results)
        avg_latency = sum(r.model_response.latency_ms for r in run_result.results) / max(
            len(run_result.results), 1
        )
        any_estimated = any(r.model_response.token_count_estimated for r in run_result.results)
        cost_summary = {
            "model": run_result.model_name,
            "total_pairs": len(run_result.results),
            "total_cost_usd": total_cost,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "avg_latency_ms": avg_latency,
            "token_counts_estimated": any_estimated,
        }
        with open(run_dir / "cost_summary.json", "w") as f:
            json.dump(cost_summary, f, indent=2)

        # Save human-readable audit table
        if annotations:
            self._save_audit_table(run_dir, results_data, run_result.model_name)

        logger.info("run_saved", run_id=run_result.run_id, run_dir=str(run_dir))
        return run_dir

    def _save_audit_table(
        self, run_dir: Path, results_data: list[dict], model_name: str
    ) -> None:
        """Write audit_table.md — human-readable per-pair verdict comparison."""
        lines = [
            f"# Audit Table — {model_name}",
            "",
            "| # | Patient | Trial | Type | Criterion (60) | Expert | GPT-4 | Model | ✓/✗ | Reasoning (80) |",
            "|---|---------|-------|------|----------------|--------|-------|-------|-----|----------------|",
        ]
        for row in results_data:
            i = row["pair_index"] + 1
            patient = row.get("patient_id", "—")
            trial = row.get("trial_id", "—")
            ctype = row.get("criterion_type", "—")[:4]  # incl/excl
            criterion = row.get("criterion_text", "—")[:60].replace("|", "/")
            expert = row.get("expert_label", "—")
            gpt4 = row.get("gpt4_label", "—")
            model = row.get("model_verdict", "—")
            correct = row.get("correct")
            tick = "✓" if correct is True else ("✗" if correct is False else "—")
            reasoning = (row.get("reasoning") or "")[:80].replace("\n", " ").replace("|", "/")
            lines.append(
                f"| {i} | {patient} | {trial} | {ctype} | {criterion} "
                f"| {expert} | {gpt4} | {model} | {tick} | {reasoning} |"
            )

        # Summary line
        correct_count = sum(1 for r in results_data if r.get("correct") is True)
        total = len(results_data)
        lines += [
            "",
            f"**Correct: {correct_count}/{total} ({correct_count/total:.0%})**",
        ]

        with open(run_dir / "audit_table.md", "w") as f:
            f.write("\n".join(lines) + "\n")
