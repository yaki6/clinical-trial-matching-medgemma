"""Patient profile card component."""

from typing import Any

import streamlit as st


def render_patient_card(profile: dict) -> None:
    """Render a patient profile summary card."""
    topic_id = profile.get("topic_id", "Unknown")
    st.subheader(f"Patient: {topic_id}")

    with st.expander("Clinical Note", expanded=False):
        st.text(profile.get("profile_text", "No profile text available"))

    ambiguities = profile.get("ambiguities", [])
    if ambiguities:
        with st.expander("Ambiguities / Missing Info"):
            for a in ambiguities:
                st.warning(a)


def render_key_facts(key_facts: dict[str, Any]) -> None:
    """Render adapted key facts as a structured display."""
    if not key_facts:
        st.info("No key facts available")
        return

    for field, value in key_facts.items():
        if isinstance(value, list):
            st.markdown(f"**{field}**:")
            for item in value:
                st.markdown(f"  - {item}")
        else:
            st.markdown(f"**{field}**: {value}")
