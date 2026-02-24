"""TrialMatch -- Clinical Trial Matching with MedGemma."""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load .env from project root (contains GOOGLE_API_KEY, HF_TOKEN)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

st.set_page_config(
    page_title="TrialMatch",
    page_icon="\U0001f3e5",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_home() -> None:
    st.title("TrialMatch: Clinical Trial Matching")

    # ── 1. Problem Statement ──────────────────────────────────────────
    st.markdown(
        "> We built **TrialMatch** with MedGemma to empower patient-centric clinical trial matching."
    )

    st.divider()

    # ── 2. Pipeline ───────────────────────────────────────────────────
    st.markdown("## Pipeline")
    st.markdown(
        "Three stages, each using the model best suited for the job:"
    )

    col_i, col_p, col_v = st.columns(3)
    with col_i:
        st.markdown(
            "#### 1. INGEST\n"
            "**MedGemma 4B** reads patient notes and imaging to extract structured clinical facts."
        )
    with col_p:
        st.markdown(
            "#### 2. PRESCREEN\n"
            "**Gemini 3 Pro** runs agentic search over the live ClinicalTrials.gov API, "
            "consulting **MedGemma 27B** for medical relevance filtering."
        )
    with col_v:
        st.markdown(
            "#### 3. VALIDATE\n"
            "**MedGemma 27B** reasons over each criterion, then **Gemini Pro** assigns the final label."
        )

    st.divider()

    # ── 3. Benchmark ──────────────────────────────────────────────────
    st.markdown("## Benchmark")
    st.markdown(
        "SoTA performance against the published **TrialGPT** paper on the same "
        "1,015 expert-labeled criterion pairs."
    )

    col1, col2 = st.columns(2)
    col1.metric("Aggregate (n=80, 4 seeds)", "87.5% vs 86.2%", "+1.3pp vs TrialGPT")
    col2.metric("Best seed (seed=42, n=20)", "95.0% vs 75.0%", "+20.0pp vs TrialGPT")

    with st.expander("All models compared (seed=42, n=20)"):
        st.markdown(
            "| Model | Accuracy | F1 | κ |\n"
            "|-------|----------|-------|-------|\n"
            "| **MedGemma 27B + Gemini Pro** | **95.0%** | **95.8%** | **0.922** |\n"
            "| MedGemma 4B + Gemini Pro | 80.0% | 83.1% | 0.688 |\n"
            "| TrialGPT baseline | 75.0% | 74.6% | — |\n"
            "| Gemini 3 Pro standalone | 75.0% | 55.8% | 0.583 |\n"
            "| MedGemma 27B standalone | 70.0% | 72.2% | 0.538 |\n"
            "| MedGemma 4B standalone | 35.0% | 31.5% | 0.030 |"
        )
    st.caption(
        "See the **Benchmark** page for full charts and pair-level audit logs."
    )

    st.divider()

    # ── 4. Demo ───────────────────────────────────────────────────────
    st.markdown("## Demo")
    st.markdown(
        "Walk through a real pipeline run on an NSCLC patient — from imaging intake "
        "to criterion-level eligibility decisions."
    )
    st.page_link(
        "pages/1_Pipeline_Demo.py",
        label="Open Pipeline Demo →",
        icon=":material/hub:",
    )


navigation = st.navigation(
    [
        st.Page(render_home, title="TrialMatch", icon="\U0001f3e5", url_path="", default=True),
        st.Page("pages/2_Benchmark.py", title="Benchmark", icon=":material/bar_chart:", url_path="Benchmark"),
        st.Page(
            "pages/1_Pipeline_Demo.py",
            title="Pipeline Demo",
            icon=":material/hub:",
            url_path="Pipeline_Demo",
        ),
    ],
    position="sidebar",
    expanded=True,
)

navigation.run()
