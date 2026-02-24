"""Story-first benchmark dashboard for MedGemma clinical trial criterion matching."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Allow importing demo/benchmark_loader.py when running from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_loader import (
    compute_aggregate_stats,
    load_pinned_benchmark_runs,
)

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
PINNED_RUNS_PATH = Path(__file__).resolve().parents[1] / "data" / "benchmark" / "pinned_runs.json"

SEED42_MEDGEMMA_4B_RUN_ID = "phase0-medgemma-4b-vertex-20260221-005453"
SEED42_MEDGEMMA_27B_RUN_ID = "phase0-medgemma-27b-vertex-20260221-020334"
SEED42_TWO_STAGE_V4_RUN_ID = "phase0-medgemma-27b+gemini-pro-two-stage-v4-20260222-124953"


def _read_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def _format_pct(value: float) -> str:
    return f"{value:.1%}"


def _format_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.1f}pp"


def _truncate(text: str, max_len: int = 90) -> str:
    if len(text) <= max_len:
        return text
    return f"{text[:max_len].rstrip()}..."


def _sanitize_audit_markdown(raw_text: str) -> str:
    """Ensure historical artifacts do not display legacy baseline naming."""
    return re.sub(
        r"gpt\s*-\s*4|gpt4",
        "TrialGPT baseline (labels in dataset)",
        raw_text,
        flags=re.IGNORECASE,
    )


def _accuracy(rows: list[dict], pred_key: str, label_key: str = "expert_label") -> float:
    if not rows:
        return 0.0
    correct = sum(1 for row in rows if row.get(pred_key) == row.get(label_key))
    return correct / len(rows)


def _accuracy_by_criterion_type(rows: list[dict], pred_key: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for criterion_type in ("inclusion", "exclusion"):
        subset = [row for row in rows if row.get("criterion_type") == criterion_type]
        result[criterion_type] = _accuracy(subset, pred_key) if subset else 0.0
    return result


def _pair_key(row: dict) -> tuple[str, str, int]:
    return (
        str(row.get("patient_id", "")),
        str(row.get("trial_id", "")),
        int(row.get("pair_index", -1)),
    )


def _compare_same_sample(base_rows: list[dict], candidate_rows: list[dict]) -> dict[str, int]:
    """Compare two runs on overlapping criterion pairs."""
    base_by_pair = {_pair_key(row): row for row in base_rows}
    candidate_by_pair = {_pair_key(row): row for row in candidate_rows}
    common_pairs = sorted(set(base_by_pair.keys()) & set(candidate_by_pair.keys()))

    summary = {
        "n_common": len(common_pairs),
        "base_errors": 0,
        "candidate_errors": 0,
        "rescued_errors": 0,
        "regressions": 0,
        "both_wrong": 0,
        "label_flips": 0,
    }

    for key in common_pairs:
        base_row = base_by_pair[key]
        candidate_row = candidate_by_pair[key]
        expert_label = str(candidate_row.get("expert_label", "UNKNOWN"))
        base_label = str(base_row.get("model_verdict", "UNKNOWN"))
        candidate_label = str(candidate_row.get("model_verdict", "UNKNOWN"))

        base_correct = base_label == expert_label
        candidate_correct = candidate_label == expert_label

        if not base_correct:
            summary["base_errors"] += 1
        if not candidate_correct:
            summary["candidate_errors"] += 1
        if not base_correct and candidate_correct:
            summary["rescued_errors"] += 1
        elif base_correct and not candidate_correct:
            summary["regressions"] += 1
        elif not base_correct and not candidate_correct:
            summary["both_wrong"] += 1
        if base_label != candidate_label:
            summary["label_flips"] += 1

    return summary


@st.cache_data(show_spinner=False)
def _load_pinned_data():
    run_data = load_pinned_benchmark_runs(
        runs_dir=RUNS_DIR,
        pinned_config_path=PINNED_RUNS_PATH,
    )
    agg = compute_aggregate_stats(run_data)
    return run_data, agg


@st.cache_data(show_spinner=False)
def _load_run_artifacts(run_id: str) -> dict:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run: {run_id}")

    config = _read_json(run_dir / "config.json")
    metrics = _read_json(run_dir / "metrics.json")
    results = _read_json(run_dir / "results.json")
    cost = _read_json(run_dir / "cost_summary.json")

    audit_path = run_dir / "audit_table.md"
    audit_table = audit_path.read_text() if audit_path.exists() else None

    if not isinstance(results, list):
        raise ValueError(f"Run {run_id} has invalid results.json (expected list).")

    return {
        "run_id": run_id,
        "config": config,
        "metrics": metrics,
        "results": results,
        "cost": cost,
        "audit_table": audit_table,
    }


def _build_audit_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for row in results:
        expert = row.get("expert_label", "UNKNOWN")
        model_label = row.get("model_verdict", "UNKNOWN")
        baseline_label = row.get("gpt4_label", "UNKNOWN")
        rows.append(
            {
                "pair_index": row.get("pair_index"),
                "patient_id": row.get("patient_id", ""),
                "trial_id": row.get("trial_id", ""),
                "criterion_type": row.get("criterion_type", ""),
                "criterion_text": _truncate(row.get("criterion_text", "")),
                "expert_label": expert,
                "TrialGPT baseline label": baseline_label,
                "model_verdict": model_label,
                "correct_model": model_label == expert,
                "correct_trialgpt": baseline_label == expert,
                "latency_ms": row.get("latency_ms", 0.0),
                "estimated_cost": row.get("estimated_cost", 0.0),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Page setup and data loading
# ---------------------------------------------------------------------------

st.title("Benchmark: MedGemma Clinical Trial Criterion Matching")
st.caption("Criterion-level MET / NOT_MET / UNKNOWN against expert labels (TrialGPT dataset).")

try:
    pinned_runs, agg = _load_pinned_data()
    run_4b = _load_run_artifacts(SEED42_MEDGEMMA_4B_RUN_ID)
    run_27b = _load_run_artifacts(SEED42_MEDGEMMA_27B_RUN_ID)
    run_v4_seed42 = _load_run_artifacts(SEED42_TWO_STAGE_V4_RUN_ID)
except Exception as exc:
    st.error(f"Benchmark data is blocked: {exc}")
    st.stop()

seed42_4b_acc = _accuracy(run_4b["results"], "model_verdict")
seed42_27b_acc = _accuracy(run_27b["results"], "model_verdict")
seed42_v4_acc = _accuracy(run_v4_seed42["results"], "model_verdict")
seed42_trialgpt_acc = _accuracy(run_v4_seed42["results"], "gpt4_label")
seed42_4b_by_type = _accuracy_by_criterion_type(run_4b["results"], "model_verdict")
seed42_27b_by_type = _accuracy_by_criterion_type(run_27b["results"], "model_verdict")
seed42_v4_by_type = _accuracy_by_criterion_type(run_v4_seed42["results"], "model_verdict")
seed42_transition = _compare_same_sample(run_27b["results"], run_v4_seed42["results"])

all_pinned_pairs = [row for run in pinned_runs for row in run.results]
v4_by_type = _accuracy_by_criterion_type(all_pinned_pairs, "model_verdict")
trialgpt_by_type = _accuracy_by_criterion_type(all_pinned_pairs, "gpt4_label")

takeaways_tab, audit_tab = st.tabs(["Takeaways", "Audit"])

# ---------------------------------------------------------------------------
# Tab 1: Takeaways
# ---------------------------------------------------------------------------

with takeaways_tab:
    st.subheader("1) Model Capacity Gap: 27B beats 4B on the same benchmark slice")
    st.markdown(
        "On the same 20 criterion pairs (seed=42), MedGemma 27B is materially stronger than MedGemma 4B. "
        "The gap appears in both inclusion and exclusion criteria."
    )
    sec1_col1, sec1_col2, sec1_col3 = st.columns(3)
    sec1_col1.metric("MedGemma 4B accuracy", _format_pct(seed42_4b_acc))
    sec1_col2.metric(
        "MedGemma 27B accuracy",
        _format_pct(seed42_27b_acc),
        _format_delta(seed42_27b_acc - seed42_4b_acc),
    )
    sec1_col3.metric("Backing sample", "n=20")

    section1_df = pd.DataFrame(
        [
            {
                "Model": "MedGemma 4B (standalone)",
                "Overall": _format_pct(seed42_4b_acc),
                "Inclusion": _format_pct(seed42_4b_by_type["inclusion"]),
                "Exclusion": _format_pct(seed42_4b_by_type["exclusion"]),
            },
            {
                "Model": "MedGemma 27B (standalone)",
                "Overall": _format_pct(seed42_27b_acc),
                "Inclusion": _format_pct(seed42_27b_by_type["inclusion"]),
                "Exclusion": _format_pct(seed42_27b_by_type["exclusion"]),
            },
        ]
    )
    st.dataframe(section1_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("2) Semantic Label Assignment: two-stage reduces label-mapping errors")
    st.markdown(
        "Single-stage models can reason clinically but still fail final label assignment "
        "(inclusion/exclusion polarity, severity qualifiers, JSON-label consistency). "
        "The two-stage method in `evaluate_criterion_two_stage()` separates Stage 1 medical analysis "
        "from Stage 2 label assignment."
    )
    sec2_col1, sec2_col2, sec2_col3 = st.columns(3)
    sec2_col1.metric("27B standalone accuracy", _format_pct(seed42_27b_acc))
    sec2_col2.metric(
        "27B two-stage v4 accuracy",
        _format_pct(seed42_v4_acc),
        _format_delta(seed42_v4_acc - seed42_27b_acc),
    )
    sec2_col3.metric(
        "Recovered errors",
        f"{seed42_transition['rescued_errors']}/{seed42_transition['base_errors']}",
    )

    section2_df = pd.DataFrame(
        [
            {"Comparison": "Shared pairs", "Value": seed42_transition["n_common"]},
            {"Comparison": "27B standalone errors", "Value": seed42_transition["base_errors"]},
            {"Comparison": "Two-stage v4 errors", "Value": seed42_transition["candidate_errors"]},
            {"Comparison": "Errors recovered by two-stage", "Value": seed42_transition["rescued_errors"]},
            {"Comparison": "Regressions introduced", "Value": seed42_transition["regressions"]},
            {"Comparison": "Label flips vs standalone", "Value": seed42_transition["label_flips"]},
        ]
    )
    st.dataframe(section2_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("3) Benchmark Positioning: competitive with TrialGPT baseline on multi-seed aggregate")
    st.markdown(
        "Across 4 seeds (n=80 pairs), two-stage v4 remains close to the TrialGPT baseline labels and "
        "shows a slight aggregate edge under this machine-evaluated protocol."
    )
    sec3_col1, sec3_col2, sec3_col3 = st.columns(3)
    sec3_col1.metric("Two-stage v4 accuracy (n=80)", _format_pct(agg.v4_accuracy))
    sec3_col2.metric("TrialGPT baseline (labels in dataset)", _format_pct(agg.gpt4_accuracy))
    sec3_col3.metric("Delta", _format_delta(agg.v4_accuracy - agg.gpt4_accuracy))

    seed42_df = pd.DataFrame(
        [
            {"Model": "MedGemma 4B (standalone)", "Accuracy": seed42_4b_acc},
            {"Model": "MedGemma 27B (standalone)", "Accuracy": seed42_27b_acc},
            {"Model": "MedGemma 27B two-stage (v4)", "Accuracy": seed42_v4_acc},
            {"Model": "TrialGPT baseline (labels in dataset)", "Accuracy": seed42_trialgpt_acc},
        ]
    )
    fig_seed42 = px.bar(
        seed42_df,
        x="Model",
        y="Accuracy",
        color="Model",
        text=seed42_df["Accuracy"].map(_format_pct),
        title="Backing Data: Seed=42 Accuracy Comparison",
    )
    fig_seed42.update_layout(yaxis_tickformat=".0%", showlegend=False, height=380)
    st.plotly_chart(fig_seed42, use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        aggregate_df = pd.DataFrame(
            [
                {"Model": "Two-stage v4 (n=80)", "Accuracy": agg.v4_accuracy},
                {
                    "Model": "TrialGPT baseline (labels in dataset, n=80)",
                    "Accuracy": agg.gpt4_accuracy,
                },
            ]
        )
        fig_aggregate = px.bar(
            aggregate_df,
            x="Model",
            y="Accuracy",
            color="Model",
            text=aggregate_df["Accuracy"].map(_format_pct),
            title="Backing Data: Multi-seed Aggregate",
        )
        fig_aggregate.update_layout(yaxis_tickformat=".0%", showlegend=False, height=360)
        st.plotly_chart(fig_aggregate, use_container_width=True)

    with chart_col2:
        type_df = pd.DataFrame(
            [
                {
                    "Criterion type": "Inclusion",
                    "Model": "Two-stage v4",
                    "Accuracy": v4_by_type["inclusion"],
                },
                {
                    "Criterion type": "Inclusion",
                    "Model": "TrialGPT baseline (labels in dataset)",
                    "Accuracy": trialgpt_by_type["inclusion"],
                },
                {
                    "Criterion type": "Exclusion",
                    "Model": "Two-stage v4",
                    "Accuracy": v4_by_type["exclusion"],
                },
                {
                    "Criterion type": "Exclusion",
                    "Model": "TrialGPT baseline (labels in dataset)",
                    "Accuracy": trialgpt_by_type["exclusion"],
                },
            ]
        )
        fig_type = px.bar(
            type_df,
            x="Criterion type",
            y="Accuracy",
            color="Model",
            barmode="group",
            text=type_df["Accuracy"].map(_format_pct),
            title="Backing Data: Inclusion vs Exclusion (n=80)",
        )
        fig_type.update_layout(yaxis_tickformat=".0%", height=360)
        st.plotly_chart(fig_type, use_container_width=True)

    st.caption(
        "TrialGPT baseline (published SoTA): 87.3% criterion accuracy in the TrialGPT paper "
        "(physician evaluation). This is context only; protocol differs from the machine label "
        "comparison above."
    )

# ---------------------------------------------------------------------------
# Tab 2: Audit
# ---------------------------------------------------------------------------

with audit_tab:
    st.subheader("Run Logs and Pair-Level Audit")

    run_choices: dict[str, str] = {}
    for run in sorted(pinned_runs, key=lambda item: item.seed or -1):
        run_choices[f"Two-stage v4 seed {run.seed} | {run.run_id}"] = run.run_id
    run_choices[f"Seed 42 MedGemma 4B standalone | {SEED42_MEDGEMMA_4B_RUN_ID}"] = (
        SEED42_MEDGEMMA_4B_RUN_ID
    )
    run_choices[f"Seed 42 MedGemma 27B standalone | {SEED42_MEDGEMMA_27B_RUN_ID}"] = (
        SEED42_MEDGEMMA_27B_RUN_ID
    )

    selected_label = st.selectbox("Select run", list(run_choices.keys()))
    selected_run_id = run_choices[selected_label]
    selected_run = _load_run_artifacts(selected_run_id)

    config = selected_run["config"]
    metrics = selected_run["metrics"]
    cost = selected_run["cost"]
    results = selected_run["results"]

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    summary_col1.metric("Seed", str(config.get("data", {}).get("seed", "--")))
    summary_col2.metric("Pairs", str(len(results)))
    summary_col3.metric("Total cost (USD)", f"${cost.get('total_cost_usd', 0.0):.4f}")
    summary_col4.metric("Avg latency (ms)", f"{cost.get('avg_latency_ms', 0.0):.0f}")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Model accuracy", _format_pct(metrics.get("accuracy", 0.0)))
    metric_col2.metric("Model macro-F1", f"{metrics.get('f1_macro', 0.0):.3f}")
    metric_col3.metric("Model kappa", f"{metrics.get('cohens_kappa', 0.0):.3f}")
    metric_col4.metric(
        "TrialGPT baseline accuracy (labels in dataset)",
        _format_pct(metrics.get("gpt4_baseline_accuracy", 0.0)),
    )
    st.caption(
        f"TrialGPT baseline macro-F1 (labels in dataset): "
        f"{metrics.get('gpt4_baseline_f1_macro', 0.0):.3f} | "
        f"Run ID: {selected_run_id}"
    )

    audit_df = _build_audit_table(results)
    filter_mode = st.radio(
        "Filter rows",
        ["All pairs", "Only model errors", "Only baseline errors", "Only disagreements"],
        horizontal=True,
    )

    filtered_df = audit_df.copy()
    if filter_mode == "Only model errors":
        filtered_df = filtered_df[~filtered_df["correct_model"]]
    elif filter_mode == "Only baseline errors":
        filtered_df = filtered_df[~filtered_df["correct_trialgpt"]]
    elif filter_mode == "Only disagreements":
        filtered_df = filtered_df[filtered_df["model_verdict"] != filtered_df["TrialGPT baseline label"]]

    st.dataframe(
        filtered_df[
            [
                "pair_index",
                "patient_id",
                "trial_id",
                "criterion_type",
                "criterion_text",
                "expert_label",
                "TrialGPT baseline label",
                "model_verdict",
                "correct_model",
                "correct_trialgpt",
                "latency_ms",
                "estimated_cost",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    if filtered_df.empty:
        st.info("No rows match the selected filter.")
    else:
        selected_pair_idx = st.selectbox(
            "Inspect pair index",
            sorted(filtered_df["pair_index"].tolist()),
        )
        selected_row = next(
            (row for row in results if row.get("pair_index") == selected_pair_idx),
            None,
        )
        if selected_row is not None:
            detail_col1, detail_col2, detail_col3 = st.columns(3)
            detail_col1.metric("Expert label", selected_row.get("expert_label", "--"))
            detail_col2.metric(
                "TrialGPT baseline label (labels in dataset)",
                selected_row.get("gpt4_label", "--"),
            )
            detail_col3.metric("Model verdict", selected_row.get("model_verdict", "--"))

            trial_id = selected_row.get("trial_id", "")
            st.write(f"Criterion type: {selected_row.get('criterion_type', '--')}")
            st.write(f"Criterion: {selected_row.get('criterion_text', '')}")
            if trial_id:
                st.markdown(f"[Open {trial_id} on ClinicalTrials.gov](https://clinicaltrials.gov/study/{trial_id})")

            if selected_row.get("stage1_reasoning"):
                st.markdown("**Stage 1 medical analysis**")
                st.code(selected_row.get("stage1_reasoning", ""), language="text")

            st.markdown("**Model reasoning**")
            st.code(selected_row.get("reasoning", ""), language="text")
            st.write(f"Evidence sentence indices: {selected_row.get('evidence_sentences', [])}")

    if selected_run.get("audit_table"):
        show_markdown = st.checkbox("Show sanitized run markdown", value=False)
        if show_markdown:
            st.markdown(_sanitize_audit_markdown(selected_run["audit_table"]))
