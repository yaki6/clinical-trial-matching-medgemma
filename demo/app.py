"""TrialMatch -- Clinical Trial Matching with MedGemma."""

from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (contains GOOGLE_API_KEY, HF_TOKEN)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import streamlit as st

st.set_page_config(
    page_title="TrialMatch",
    page_icon="\U0001f3e5",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("TrialMatch: Clinical Trial Matching")
st.markdown(
    """
Multi-model AI pipeline for matching cancer patients to clinical trials.

**Pipeline**: INGEST \u2192 PRESCREEN \u2192 VALIDATE

**Models**: MedGemma 4B (normalization) \u00b7 MedGemma 27B (evaluation)
\u00b7 Gemini 3 Pro (orchestration)

---
Select a page from the sidebar to get started.
"""
)
