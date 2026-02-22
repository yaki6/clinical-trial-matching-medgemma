"""Benchmark Dashboard -- Multi-seed Phase 0 v4 results (MedGemma 27B + Gemini Pro vs GPT-4)."""

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

from benchmark_loader import (
    compute_aggregate_stats,
    load_pinned_benchmark_runs,
)

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
    return f"{value:.1%}"


def _format_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.1f}pp"


def _per_class_f1_from_cm(cm: list[list[int]], labels: list[str]) -> dict[str, float]:
    """Compute per-class F1 from a confusion matrix."""
    result = {}
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(len(labels))) - tp
        fn = sum(cm[i][j] for j in range(len(labels))) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        result[label] = f1
    return result


# ── Page Header ──────────────────────────────────────────────────────────────

st.title("Benchmark Dashboard")
st.caption("Phase 0 Criterion-Level Evaluation -- Multi-Seed Analysis (n=80)")

try:
    run_data = load_pinned_benchmark_runs(
        runs_dir=RUNS_DIR,
        pinned_config_path=PINNED_RUNS_PATH,
    )
except Exception as exc:
    st.error(f"Benchmark data is blocked: {exc}")
    st.stop()

agg = compute_aggregate_stats(run_data)

# ── Headline Metrics ─────────────────────────────────────────────────────────

st.header("Aggregate Results (n=80, 4 seeds)")

col1, col2, col3 = st.columns(3)
col1.metric(
    "v4 (27B+Pro) Accuracy",
    _format_pct(agg.v4_accuracy),
)
col2.metric(
    "GPT-4 Accuracy",
    _format_pct(agg.gpt4_accuracy),
)
col3.metric(
    "Delta",
    _format_delta(agg.v4_accuracy - agg.gpt4_accuracy),
)

subcol1, subcol2 = st.columns(2)
subcol1.metric("v4 Macro-F1", f"{agg.v4_f1_macro:.4f}")
subcol2.metric("v4 Cohen's kappa", f"{agg.v4_kappa:.4f}")

# ── Per-Seed Comparison Table ────────────────────────────────────────────────

st.header("Per-Seed Breakdown")

seed_rows = []
for ps in sorted(agg.per_seed, key=lambda x: x["seed"] or 0):
    v4_acc = ps["v4_accuracy"]
    gpt4_acc = ps["gpt4_accuracy"]
    delta = v4_acc - gpt4_acc
    seed_rows.append(
        {
            "Seed": ps["seed"],
            "v4 Acc": _format_pct(v4_acc),
            "GPT-4 Acc": _format_pct(gpt4_acc),
            "Delta": _format_delta(delta),
            "v4 F1": f"{ps['v4_f1_macro']:.3f}",
            "v4 kappa": f"{ps['v4_kappa']:.3f}",
        }
    )

seed_df = pd.DataFrame(seed_rows)
st.dataframe(seed_df, use_container_width=True, hide_index=True)

st.caption(
    "Each seed selects a different random 20-pair subset from the TrialGPT HF dataset (n=1015). "
    "Variance across seeds (80-95%) illustrates the instability of small-sample evaluation."
)

# ── Aggregate Confusion Matrices ─────────────────────────────────────────────

st.header("Aggregate Confusion Matrices")

cm_col1, cm_col2 = st.columns(2)
with cm_col1:
    fig_v4 = create_confusion_matrix_figure(agg.v4_confusion, agg.labels, "v4 (27B+Pro)")
    st.plotly_chart(fig_v4, use_container_width=True)

with cm_col2:
    fig_gpt4 = create_confusion_matrix_figure(agg.gpt4_confusion, agg.labels, "GPT-4 (baseline)")
    st.plotly_chart(fig_gpt4, use_container_width=True)

# ── Per-Class F1 Bar Chart ───────────────────────────────────────────────────

st.header("Per-Class F1 Scores (Aggregate)")

v4_f1 = _per_class_f1_from_cm(agg.v4_confusion, agg.labels)
gpt4_f1 = _per_class_f1_from_cm(agg.gpt4_confusion, agg.labels)

f1_rows = []
for cls in agg.labels:
    f1_rows.append({"Model": "v4 (27B+Pro)", "Class": cls, "F1": v4_f1[cls]})
    f1_rows.append({"Model": "GPT-4", "Class": cls, "F1": gpt4_f1[cls]})

f1_df = pd.DataFrame(f1_rows)
fig_f1 = px.bar(
    f1_df,
    x="Class",
    y="F1",
    color="Model",
    barmode="group",
    title="F1 Score by Class (v4 vs GPT-4, n=80)",
    range_y=[0, 1],
)
fig_f1.update_layout(height=400)
st.plotly_chart(fig_f1, use_container_width=True)

# ── Cost & Latency ───────────────────────────────────────────────────────────

st.header("Cost & Latency")

cost_rows = []
for run in run_data:
    c = run.cost
    cost_rows.append(
        {
            "Seed": str(run.seed),
            "Run ID": run.run_id,
            "Pairs": c.get("total_pairs", "--"),
            "Cost ($)": f"${c.get('total_cost_usd', 0):.4f}",
            "Input Tokens": f"{c.get('total_input_tokens', 0):,}",
            "Output Tokens": f"{c.get('total_output_tokens', 0):,}",
            "Avg Latency (ms)": f"{c.get('avg_latency_ms', 0):,.0f}",
        }
    )

cost_rows.append(
    {
        "Seed": "ALL",
        "Run ID": "Aggregate (4 seeds)",
        "Pairs": agg.n,
        "Cost ($)": f"${agg.total_cost_usd:.4f}",
        "Input Tokens": f"{agg.total_input_tokens:,}",
        "Output Tokens": f"{agg.total_output_tokens:,}",
        "Avg Latency (ms)": "--",
    }
)

cost_df = pd.DataFrame(cost_rows)
st.dataframe(cost_df, use_container_width=True, hide_index=True)

# ── Statistical Caveat ───────────────────────────────────────────────────────

st.header("Statistical Caveat")
st.warning(
    "All results are on n=80 pairs (4 seeds x 20 pairs). "
    "95% CI for 87.5% accuracy at n=80 is approximately [78%, 94%]. "
    "Single-seed results range from 80% to 95%, illustrating high variance. "
    "Tier A evaluation (n=1024) is needed for statistical significance."
)

# ── Audit Tables ─────────────────────────────────────────────────────────────

st.header("Audit Tables")
for run in run_data:
    if run.audit_table:
        with st.expander(f"Seed {run.seed} -- Audit Table", expanded=False):
            st.markdown(run.audit_table)
