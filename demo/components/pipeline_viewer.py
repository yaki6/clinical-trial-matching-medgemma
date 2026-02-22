"""Pipeline step viewer components."""

import streamlit as st

# Mapping from internal field names to patient-friendly display names.
_FRIENDLY_FIELD_NAMES: dict[str, str] = {
    "primary_diagnosis": "Diagnosis",
    "demographics": "Age & Sex",
    "tobacco_use": "Smoking History",
    "histopathology": "Pathology Results",
    "imaging_findings": "Imaging Results",
    "missing_info": "Information Gaps",
    "symptoms": "Current Symptoms",
}


def _friendly_field(field: str) -> str:
    """Return the patient-friendly name for a field, falling back to title case."""
    return _FRIENDLY_FIELD_NAMES.get(field, field.replace("_", " ").title())


def render_ingest_step(key_facts: dict, dev_mode: bool = False) -> None:
    """Render the INGEST step showing extracted key facts.

    Args:
        key_facts: Dictionary of extracted clinical facts.
        dev_mode: If True, show technical labels. If False, show patient-friendly labels.
    """
    if dev_mode:
        title = "Step 1: INGEST -- Key Facts"
        caption = "Key clinical facts extracted from patient record"
    else:
        title = "Your Clinical Profile"
        caption = "Here's what we extracted from your medical record:"

    with st.expander(title, expanded=True):
        st.caption(caption)
        if not key_facts:
            st.warning("No key facts available")
            return
        # In patient mode, skip redundant individual fields when a combined
        # field already exists (e.g. skip "age"/"sex" when "demographics" exists)
        skip_fields = set()
        if not dev_mode:
            if "demographics" in key_facts and "age" in key_facts:
                skip_fields.update({"age", "sex"})

        for field, value in key_facts.items():
            if field in skip_fields:
                continue
            col1, col2 = st.columns([1, 3])
            with col1:
                display_name = field if dev_mode else _friendly_field(field)
                st.markdown(f"**{display_name}**")
            with col2:
                if isinstance(value, list):
                    st.markdown(", ".join(str(v) for v in value))
                elif isinstance(value, dict):
                    # Flatten dict values for display (e.g. demographics: {age: 43, sex: female})
                    parts = [f"{k}: {v}" for k, v in value.items()]
                    st.markdown("; ".join(parts))
                else:
                    st.markdown(str(value))


def render_prescreen_placeholder(dev_mode: bool = False) -> None:
    """Render PRESCREEN placeholder before execution.

    Args:
        dev_mode: If True, show technical labels. If False, show patient-friendly labels.
    """
    if dev_mode:
        title = "Step 2: PRESCREEN -- Trial Search"
        message = "Click 'Run Pipeline' to search ClinicalTrials.gov for matching trials."
    else:
        title = "Finding Matching Trials"
        message = "Click 'Search for Trials' to find clinical trials that may match your condition."

    with st.expander(title, expanded=False):
        st.info(message)


def render_validate_placeholder(dev_mode: bool = False) -> None:
    """Render VALIDATE placeholder before execution.

    Args:
        dev_mode: If True, show technical labels. If False, show patient-friendly labels.
    """
    if dev_mode:
        title = "Step 3: VALIDATE -- Eligibility Check"
        message = "Eligibility evaluation will run after PRESCREEN finds candidate trials."
    else:
        title = "Checking Your Eligibility"
        message = "Eligibility evaluation will begin after matching trials are found."

    with st.expander(title, expanded=False):
        st.info(message)
