"""Patient profile card component."""

import re
from pathlib import Path
from typing import Any

import streamlit as st


def extract_friendly_label(profile: dict) -> str:
    """Extract a friendly patient label like '43F, Lung Adenocarcinoma' from key_facts.

    key_facts is a list of objects with "field" and "value" keys.
    Returns the friendly suffix or empty string if extraction fails.
    """
    key_facts = profile.get("key_facts", [])
    if not isinstance(key_facts, list):
        return ""

    age_sex = ""
    diagnosis = ""

    for fact in key_facts:
        if not isinstance(fact, dict):
            continue
        field = fact.get("field", "")
        value = fact.get("value", "")

        if field == "demographics" and value:
            # value can be a dict {"age": "43", "sex": "female"}
            # or a string "age: 43; sex: female"
            if isinstance(value, dict):
                age_part = str(value.get("age", ""))
                sex_raw = str(value.get("sex", ""))
                sex_part = sex_raw[0:1].upper() if sex_raw else ""
            elif isinstance(value, str):
                age_match = re.search(r"age:\s*(\d+)", value, re.IGNORECASE)
                sex_match = re.search(r"sex:\s*(\w+)", value, re.IGNORECASE)
                age_part = age_match.group(1) if age_match else ""
                sex_raw = sex_match.group(1) if sex_match else ""
                sex_part = sex_raw[0:1].upper() if sex_raw else ""
            else:
                age_part = ""
                sex_part = ""
            age_sex = f"{age_part}{sex_part}"

        if field == "primary_diagnosis" and value and isinstance(value, str):
            if len(value) > 40:
                diagnosis = value[:37].strip() + "..."
            else:
                diagnosis = value.strip()

    if age_sex and diagnosis:
        return f" ({age_sex}, {diagnosis})"
    elif age_sex:
        return f" ({age_sex})"
    elif diagnosis:
        return f" ({diagnosis})"
    return ""


def render_patient_card(profile: dict, dev_mode: bool = False) -> None:
    """Render a patient profile summary card.

    Args:
        profile: Patient profile dictionary.
        dev_mode: If True, show technical labels. If False, show patient-friendly labels.
    """
    topic_id = profile.get("topic_id", "Unknown")

    if dev_mode:
        st.subheader(f"Patient: {topic_id}")

        with st.expander("Clinical Note", expanded=False):
            st.markdown(profile.get("profile_text", "No profile text available"))

        ambiguities = profile.get("ambiguities", [])
        if ambiguities:
            with st.expander("Ambiguities / Missing Info"):
                for a in ambiguities:
                    st.warning(a)
    else:
        friendly_suffix = extract_friendly_label(profile)
        st.subheader(f"Patient: {topic_id}{friendly_suffix}")

        with st.expander("View Full Medical Record", expanded=False):
            st.markdown(profile.get("profile_text", "No profile text available"))

        ambiguities = profile.get("ambiguities", [])
        if ambiguities:
            with st.expander("Information Gaps in Your Record"):
                for a in ambiguities:
                    st.warning(a)


def render_medical_image(
    image_path: Path, image_meta: dict, dev_mode: bool = False
) -> None:
    """Render a medical image thumbnail with modality/location caption.

    Args:
        image_path: Resolved path to the image file.
        image_meta: Dict with keys: modality, plane, location, caption.
        dev_mode: If True, show file path and technical details.
    """
    modality = image_meta.get("modality", "Unknown")
    plane = image_meta.get("plane", "")
    location = image_meta.get("location", "")

    if dev_mode:
        label = f"Medical Image: {modality} ({plane}) -- {location}"
        with st.expander(label, expanded=True):
            st.image(str(image_path), use_container_width=True)
            st.caption(image_meta.get("caption", ""))
            st.caption(f"File: `{image_path.name}`")
    else:
        parts = [modality]
        if plane:
            parts.append(plane)
        label = " -- ".join(parts)
        with st.expander(f"Medical Image: {label}", expanded=True):
            st.image(str(image_path), use_container_width=True)
            if image_meta.get("caption"):
                st.caption(image_meta["caption"])


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
