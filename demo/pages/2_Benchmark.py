"""Benchmark Dashboard -- pinned Phase 0 demo results."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow importing demo/benchmark_loader.py when running from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_loader import BenchmarkRunData, load_pinned_benchmark_runs

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
PINNED_RUNS_PATH = Path(__file__).resolve().parents[1] / "data" / "benchmark" / "pinned_runs.json"


def create_confusion_matrix_figure(
    matrix: list[list[int]],
    labels: list[str],
    title: str,
) -> go.Figure:
    """Create a plotly heatmap for a confusion matrix."""
    arr = np.array(matrix)
    fig = go.Figure(
        data=go.Heatmap(
            z=arr,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=False,
            text=arr,
            texttemplate="%{text}",
            textfont={"size": 16},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=400,
        height=400,
        yaxis={"autorange": "reversed"},
    )
    return fig


def _format_pct(value: float) -> str:
    return f"{value:.0%}"


def _format_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.1f}pp"


def _find_run(runs: list[BenchmarkRunData], token: str) -> BenchmarkRunData | None:
    token_lower = token.lower()
    for run in runs:
        if token_lower in run.run_id.lower() or token_lower in run.label.lower():
            return run
    return None


def _render_key_findings(runs: list[BenchmarkRunData]) -> None:
    """Render data-derived findings to avoid stale hardcoded narrative."""
    st.header("Key Findings")

    if not runs:
        st.info("No pinned runs loaded.")
        return

    by_acc = sorted(runs, key=lambda r: r.metrics.get("accuracy", 0.0), reverse=True)
    best = by_acc[0]
    gpt4_baseline = best.metrics.get("gpt4_baseline_accuracy")

    med27 = _find_run(runs, "27b")
    med4 = _find_run(runs, "4b")
    gemini = _find_run(runs, "gemini-3-pro-preview")

    findings: list[str] = []
    findings.append(
        f"**Top pinned run:** {best.label} at "
        f"{_format_pct(best.metrics.get('accuracy', 0.0))} accuracy "
        f"(F1 macro {best.metrics.get('f1_macro', 0.0):.3f}, "
        f"kappa {best.metrics.get('cohens_kappa', 0.0):.3f})."
    )

    if gpt4_baseline is not None:
        delta = best.metrics.get("accuracy", 0.0) - float(gpt4_baseline)
        relation = "above" if delta >= 0 else "below"
        findings.append(
            f"**Best run vs GPT-4 baseline:** {_format_delta(delta)} {relation} "
            f"(baseline {_format_pct(float(gpt4_baseline))})."
        )

    if med27 and med4:
        delta = med27.metrics.get("accuracy", 0.0) - med4.metrics.get("accuracy", 0.0)
        relation = "higher" if delta >= 0 else "lower"
        findings.append(
            f"**27B vs 4B:** {med27.label} is {_format_delta(delta)} {relation} "
            f"in this pinned comparison."
        )

    if med27 and gemini:
        delta = med27.metrics.get("accuracy", 0.0) - gemini.metrics.get("accuracy", 0.0)
        relation = "higher" if delta >= 0 else "lower"
        findings.append(
            f"**MedGemma 27B two-stage vs Gemini single-stage:** "
            f"{_format_delta(delta)} {relation} on the same sampled pair set."
        )

    for bullet in findings:
        st.markdown(f"- {bullet}")


st.title("Benchmark Dashboard")
st.caption("Phase 0 criterion-level evaluation -- pinned and reproducible demo runs")

try:
    run_data = load_pinned_benchmark_runs(
        runs_dir=RUNS_DIR,
        pinned_config_path=PINNED_RUNS_PATH,
    )
except Exception as exc:
    st.error(f"Benchmark data is blocked: {exc}")
    st.stop()

st.caption(
    f"Pinned config: `{PINNED_RUNS_PATH.name}` | "
    f"Seed: {run_data[0].seed} | "
    f"Pair set hash: `{run_data[0].pair_set_hash}`"
)

st.header("Model Comparison")
comparison_rows = []
gpt4_row_added = False

for run in run_data:
    m = run.metrics
    comparison_rows.append(
        {
            "Model": run.label,
            "Run ID": run.run_id,
            "Accuracy": _format_pct(m.get("accuracy", 0.0)),
            "F1 Macro": f"{m.get('f1_macro', 0.0):.3f}",
            "F1 MET/NOT_MET": f"{m.get('f1_met_not_met', 0.0):.3f}",
            "Cohen's Kappa": f"{m.get('cohens_kappa', 0.0):.3f}",
        }
    )
    if not gpt4_row_added and "gpt4_baseline_accuracy" in m:
        comparison_rows.append(
            {
                "Model": "GPT-4 (baseline)",
                "Run ID": "HF dataset baseline",
                "Accuracy": _format_pct(float(m["gpt4_baseline_accuracy"])),
                "F1 Macro": f"{float(m.get('gpt4_baseline_f1_macro', 0.0)):.3f}",
                "F1 MET/NOT_MET": "--",
                "Cohen's Kappa": "--",
            }
        )
        gpt4_row_added = True

comparison_df = pd.DataFrame(comparison_rows)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

st.header("Confusion Matrices")
with_matrix = [
    run for run in run_data if "confusion_matrix" in run.metrics and "confusion_matrix_labels" in run.metrics
]
if with_matrix:
    cols = st.columns(len(with_matrix))
    for col, run in zip(cols, with_matrix, strict=True):
        fig = create_confusion_matrix_figure(
            run.metrics["confusion_matrix"],
            run.metrics["confusion_matrix_labels"],
            run.label,
        )
        col.plotly_chart(fig, use_container_width=True)

st.header("Per-Class F1 Scores")
f1_rows = []
for run in run_data:
    f1_per_class = run.metrics.get("f1_per_class", {})
    for cls, score in f1_per_class.items():
        f1_rows.append({"Model": run.label, "Class": cls, "F1": score})

if f1_rows:
    f1_df = pd.DataFrame(f1_rows)
    fig = px.bar(
        f1_df,
        x="Class",
        y="F1",
        color="Model",
        barmode="group",
        title="F1 Score by Class and Model",
        range_y=[0, 1],
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.header("Cost & Latency")
cost_rows = []
for run in run_data:
    c = run.cost
    cost_rows.append(
        {
            "Model": run.label,
            "Run ID": run.run_id,
            "Pairs": c.get("total_pairs", "--"),
            "Cost ($)": f"${c.get('total_cost_usd', 0):.4f}",
            "Input Tokens": f"{c.get('total_input_tokens', 0):,}",
            "Output Tokens": f"{c.get('total_output_tokens', 0):,}",
            "Avg Latency (ms)": f"{c.get('avg_latency_ms', 0):,.0f}",
        }
    )

cost_df = pd.DataFrame(cost_rows)
st.dataframe(cost_df, use_container_width=True, hide_index=True)

_render_key_findings(run_data)

st.header("Audit Tables")
for run in run_data:
    if run.audit_table:
        with st.expander(f"{run.label} -- Audit Table", expanded=False):
            st.markdown(run.audit_table)
