"""Benchmark Dashboard -- Phase 0 Results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def discover_runs(runs_dir: Path) -> dict[str, Path]:
    """Scan runs/ for phase0-* dirs and return {model_name: run_path} (latest per model)."""
    runs: dict[str, Path] = {}
    if not runs_dir.exists():
        return runs

    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("phase0-"):
            continue
        # Extract model name: phase0-<model>-<timestamp>
        parts = d.name.split("-")
        # Timestamp is the last two parts (date-time)
        model_name = "-".join(parts[1:-2])
        # Keep last (latest by sorted order)
        runs[model_name] = d

    return runs


def load_run_data(run_path: Path) -> dict:
    """Load metrics.json and cost_summary.json from a run directory."""
    data: dict = {"path": run_path, "name": run_path.name}

    metrics_path = run_path / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data["metrics"] = json.load(f)

    cost_path = run_path / "cost_summary.json"
    if cost_path.exists():
        with open(cost_path) as f:
            data["cost"] = json.load(f)

    audit_path = run_path / "audit_table.md"
    if audit_path.exists():
        data["audit_table"] = audit_path.read_text()

    return data


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


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.title("Benchmark Dashboard")
st.caption("Phase 0 criterion-level evaluation -- 20 pairs from TrialGPT HF dataset")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

runs = discover_runs(RUNS_DIR)
if not runs:
    st.error("No Phase 0 runs found in runs/ directory.")
    st.stop()

run_data: dict[str, dict] = {}
for model_name, run_path in runs.items():
    run_data[model_name] = load_run_data(run_path)

# ---------------------------------------------------------------------------
# Section 1: Model Comparison Table
# ---------------------------------------------------------------------------

st.header("Model Comparison")

comparison_rows = []
gpt4_row_added = False

for model_name, data in run_data.items():
    m = data.get("metrics", {})
    display_name = model_name.replace("-", " ").title()
    comparison_rows.append(
        {
            "Model": display_name,
            "Accuracy": f"{m.get('accuracy', 0):.0%}",
            "F1 Macro": f"{m.get('f1_macro', 0):.3f}",
            "F1 MET/NOT_MET": f"{m.get('f1_met_not_met', 0):.3f}",
            "Cohen's Kappa": f"{m.get('cohens_kappa', 0):.3f}",
        }
    )
    if not gpt4_row_added and "gpt4_baseline_accuracy" in m:
        comparison_rows.append(
            {
                "Model": "GPT-4 (baseline)",
                "Accuracy": f"{m['gpt4_baseline_accuracy']:.0%}",
                "F1 Macro": f"{m['gpt4_baseline_f1_macro']:.3f}",
                "F1 MET/NOT_MET": "--",
                "Cohen's Kappa": "--",
            }
        )
        gpt4_row_added = True

comparison_df = pd.DataFrame(comparison_rows)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Section 2: Confusion Matrices (side-by-side)
# ---------------------------------------------------------------------------

st.header("Confusion Matrices")

model_names_with_metrics = [
    (name, data)
    for name, data in run_data.items()
    if "metrics" in data and "confusion_matrix" in data["metrics"]
]

if model_names_with_metrics:
    cols = st.columns(len(model_names_with_metrics))
    for col, (model_name, data) in zip(cols, model_names_with_metrics, strict=True):
        m = data["metrics"]
        display_name = model_name.replace("-", " ").title()
        fig = create_confusion_matrix_figure(
            m["confusion_matrix"],
            m["confusion_matrix_labels"],
            display_name,
        )
        col.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 3: Per-class F1 Bar Chart
# ---------------------------------------------------------------------------

st.header("Per-Class F1 Scores")

f1_rows = []
for model_name, data in run_data.items():
    m = data.get("metrics", {})
    f1_per_class = m.get("f1_per_class", {})
    display_name = model_name.replace("-", " ").title()
    for cls, score in f1_per_class.items():
        f1_rows.append({"Model": display_name, "Class": cls, "F1": score})

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

# ---------------------------------------------------------------------------
# Section 4: Cost & Latency
# ---------------------------------------------------------------------------

st.header("Cost & Latency")

cost_rows = []
for model_name, data in run_data.items():
    c = data.get("cost", {})
    if c:
        display_name = model_name.replace("-", " ").title()
        cost_rows.append(
            {
                "Model": display_name,
                "Pairs": c.get("total_pairs", "--"),
                "Cost ($)": f"${c.get('total_cost_usd', 0):.4f}",
                "Input Tokens": f"{c.get('total_input_tokens', 0):,}",
                "Output Tokens": f"{c.get('total_output_tokens', 0):,}",
                "Avg Latency (ms)": f"{c.get('avg_latency_ms', 0):,.0f}",
            }
        )

if cost_rows:
    cost_df = pd.DataFrame(cost_rows)
    st.dataframe(cost_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Section 5: Key Findings
# ---------------------------------------------------------------------------

st.header("Key Findings")

st.markdown(
    """
- **MedGemma 27B is competitive with GPT-4** on criterion-level matching
  (70% accuracy, F1 macro 0.722, Cohen's kappa 0.538 vs GPT-4 baseline 75%)
- **27B dramatically outperforms 4B** (70% vs 35% accuracy) -- the larger model
  has much better instruction-following for structured JSON output
- **MedGemma 4B has systematic MET bias** on exclusion criteria -- model reasoning
  is often correct but the JSON label contradicts it (instruction-following failure,
  not a reasoning failure). Degraded further by max_tokens=512 TGI CUDA bug workaround
- **Gemini 3 Pro matches GPT-4 baseline** at 75% accuracy, confirming it as a
  strong general-purpose option for criterion evaluation
- **Multi-model approach validated**: domain-specialized models (MedGemma 27B)
  approach general model (GPT-4) accuracy, supporting the orchestration narrative
"""
)

# ---------------------------------------------------------------------------
# Section 6: Audit Tables
# ---------------------------------------------------------------------------

st.header("Audit Tables")

for model_name, data in run_data.items():
    audit = data.get("audit_table")
    if audit:
        display_name = model_name.replace("-", " ").title()
        with st.expander(f"{display_name} -- Audit Table", expanded=False):
            st.markdown(audit)
