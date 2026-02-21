"""Pipeline step viewer components."""

import streamlit as st


def render_ingest_step(key_facts: dict) -> None:
    """Render the INGEST step showing extracted key facts."""
    with st.expander("Step 1: INGEST -- Key Facts", expanded=True):
        st.caption("Key clinical facts extracted from patient record")
        if not key_facts:
            st.warning("No key facts available")
            return
        for field, value in key_facts.items():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**{field}**")
            with col2:
                if isinstance(value, list):
                    st.markdown(", ".join(str(v) for v in value))
                else:
                    st.markdown(str(value))


def render_prescreen_placeholder() -> None:
    """Render PRESCREEN placeholder before execution."""
    with st.expander("Step 2: PRESCREEN -- Trial Search", expanded=False):
        st.info("Click 'Run Pipeline' to search ClinicalTrials.gov for matching trials.")


def render_validate_placeholder() -> None:
    """Render VALIDATE placeholder before execution."""
    with st.expander("Step 3: VALIDATE -- Eligibility Check", expanded=False):
        st.info("Eligibility evaluation will run after PRESCREEN finds candidate trials.")
