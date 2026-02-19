"""Run manager for saving benchmark results and cost tracking.

Each run saves to runs/<run_id>/ with config, results, and metrics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import structlog

from trialmatch.models.schema import RunResult

logger = structlog.get_logger()

RUNS_DIR = Path("runs")


class RunManager:
    """Manages benchmark run artifacts."""

    def __init__(self, runs_dir: Path = RUNS_DIR):
        self._runs_dir = runs_dir

    def generate_run_id(self, model_name: str) -> str:
        """Generate a unique run ID."""
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"phase0-{model_name}-{ts}"

    def save_run(self, run_result: RunResult, config: dict | None = None) -> Path:
        """Save run artifacts to runs/<run_id>/."""
        run_dir = self._runs_dir / run_result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        if config:
            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2, default=str)

        # Save metrics
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(run_result.metrics, f, indent=2, default=str)

        # Save per-pair results
        results_data = []
        for r in run_result.results:
            results_data.append({
                "verdict": r.verdict.value,
                "reasoning": r.reasoning,
                "evidence_sentences": r.evidence_sentences,
                "input_tokens": r.model_response.input_tokens,
                "output_tokens": r.model_response.output_tokens,
                "latency_ms": r.model_response.latency_ms,
                "estimated_cost": r.model_response.estimated_cost,
            })

        with open(run_dir / "results.json", "w") as f:
            json.dump(results_data, f, indent=2)

        # Save cost summary
        total_cost = sum(r.model_response.estimated_cost for r in run_result.results)
        total_input = sum(r.model_response.input_tokens for r in run_result.results)
        total_output = sum(r.model_response.output_tokens for r in run_result.results)
        avg_latency = (
            sum(r.model_response.latency_ms for r in run_result.results)
            / max(len(run_result.results), 1)
        )

        cost_summary = {
            "model": run_result.model_name,
            "total_pairs": len(run_result.results),
            "total_cost_usd": total_cost,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "avg_latency_ms": avg_latency,
        }
        with open(run_dir / "cost_summary.json", "w") as f:
            json.dump(cost_summary, f, indent=2)

        logger.info("run_saved", run_id=run_result.run_id, run_dir=str(run_dir))
        return run_dir
