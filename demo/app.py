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
    st.markdown(
        "Agentic clinical trial matching pipeline that combines MedGemma reasoning with "
        "Gemini orchestration for criterion-level eligibility decisions."
    )

    st.markdown("### How MedGemma Is Used")
    st.markdown(
        """
1. **INGEST**: MedGemma 1.5 4B structures and normalizes patient clinical context.
2. **PRESCREEN**: Gemini 3 Pro orchestrates candidate trial retrieval and filtering.
3. **VALIDATE**: MedGemma 27B performs criterion-level medical reasoning, then a separate label assignment step produces the final MET / NOT_MET / UNKNOWN decision.
"""
    )

    st.markdown("### Top 3 Benchmark Takeaways")
    col1, col2, col3 = st.columns(3)
    col1.metric("27B vs 4B (seed=42, n=20)", "70.0% vs 60.0%", "+10.0pp")
    col2.metric("Two-stage vs 27B (seed=42, n=20)", "95.0% vs 70.0%", "+25.0pp")
    col3.metric("Multi-seed vs TrialGPT baseline (n=80)", "87.5% vs 86.2%", "+1.2pp")

    st.caption(
        "Published reference: TrialGPT paper reports 87.3% criterion-level accuracy under "
        "physician evaluation (protocol differs from this machine label comparison)."
    )
    st.markdown("Use the **Benchmark** page for full charts and pair-level audit logs.")


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
