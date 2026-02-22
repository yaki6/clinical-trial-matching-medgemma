"""Patient-friendly trial result card component.

Renders a single clinical trial's eligibility results in a card layout.
Designed for "patient mode" — uses plain-English verdicts and groups criteria
by outcome (met / not met / needs review) so non-technical users can quickly
understand their eligibility status.

Also supports a "dev mode" detailed expander that shows full criterion text,
type tags (inclusion/exclusion), per-criterion verdicts, and optional
two-stage reasoning traces.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Verdict display constants
# ---------------------------------------------------------------------------

_VERDICT_MESSAGES: dict[str, tuple[str, str]] = {
    # verdict -> (patient-facing message, streamlit color tag)
    "ELIGIBLE": ("You may be eligible for this trial", "green"),
    "EXCLUDED": ("You likely do not qualify for this trial", "red"),
    "UNCERTAIN": (
        "More information is needed to determine eligibility",
        "orange",
    ),
}

_CRITERION_ICONS: dict[str, str] = {
    "MET": "🟢",
    "NOT_MET": "🔴",
    "UNKNOWN": "🟡",
}

_TRUNCATE_LEN = 60


def _truncate(text: str, max_len: int = _TRUNCATE_LEN) -> str:
    """Truncate text to *max_len* characters, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "..."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_trial_card(
    nct_id: str,
    title: str,
    phase: str,
    status: str,
    verdict: str,
    criteria: list[dict],
) -> None:
    """Render a single trial's eligibility results as a bordered card.

    Parameters
    ----------
    nct_id:
        ClinicalTrials.gov identifier (e.g. ``"NCT12345678"``).
    title:
        Full official trial title.
    phase:
        Trial phase string (e.g. ``"Phase 2"``).
    status:
        Recruitment status (e.g. ``"Recruiting"``).
    verdict:
        Trial-level eligibility verdict — one of
        ``"ELIGIBLE"``, ``"EXCLUDED"``, ``"UNCERTAIN"``.
    criteria:
        List of per-criterion dicts, each containing:

        - ``text`` (str): full criterion text
        - ``type`` (str): ``"inclusion"`` or ``"exclusion"``
        - ``verdict`` (str): ``"MET"``, ``"NOT_MET"``, or ``"UNKNOWN"``
        - ``reasoning`` (str): model reasoning / explanation
        - ``stage1_reasoning`` (str | None): MedGemma reasoning in two-stage mode
    """
    with st.container(border=True):
        # -- Header: title, NCT ID caption, badges --
        st.markdown(f"### {title}")
        st.caption(f"{nct_id}")

        badge_cols = st.columns([1, 1, 3])
        with badge_cols[0]:
            st.markdown(f"`{phase}`")
        with badge_cols[1]:
            st.markdown(f"`{status}`")

        # -- Patient-friendly verdict message --
        message, color = _VERDICT_MESSAGES.get(
            verdict, ("Eligibility could not be determined", "orange")
        )
        st.markdown(f":{color}[**{message}**]")

        st.markdown("---")

        # -- Group criteria by outcome --
        met_criteria = [c for c in criteria if c.get("verdict") == "MET"]
        not_met_criteria = [c for c in criteria if c.get("verdict") == "NOT_MET"]
        unknown_criteria = [c for c in criteria if c.get("verdict") == "UNKNOWN"]

        if met_criteria:
            st.markdown("**Criteria you meet:**")
            for c in met_criteria:
                st.markdown(f"&ensp; :green[**\u2713**] {_truncate(c['text'])}")

        if not_met_criteria:
            st.markdown("**Criteria not met:**")
            for c in not_met_criteria:
                st.markdown(f"&ensp; :red[**\u2717**] {_truncate(c['text'])}")

        if unknown_criteria:
            st.markdown("**Needs further review:**")
            for c in unknown_criteria:
                st.markdown(f"&ensp; :orange[**?**] {_truncate(c['text'])}")

        # -- Detailed eligibility breakdown (dev-friendly) --
        with st.expander("View detailed eligibility breakdown"):
            for c in criteria:
                icon = _CRITERION_ICONS.get(c.get("verdict", ""), "")
                ctype_tag = c.get("type", "").upper()
                verdict_val = c.get("verdict", "UNKNOWN")

                st.markdown(
                    f"{icon} **{verdict_val}** &nbsp; `{ctype_tag}` &mdash; {c.get('text', '')}"
                )

                # Two-stage reasoning (MedGemma -> Gemini)
                if c.get("stage1_reasoning"):
                    st.markdown("*Stage 1 — MedGemma Medical Reasoning:*")
                    st.text(c["stage1_reasoning"])
                    if c.get("reasoning"):
                        st.markdown("*Stage 2 — Gemini Label Assignment:*")
                        st.text(c["reasoning"])
                elif c.get("reasoning"):
                    st.markdown("*Reasoning:*")
                    st.text(c["reasoning"])

                st.markdown("")  # visual spacing between criteria

        # -- ClinicalTrials.gov link --
        ctgov_url = f"https://clinicaltrials.gov/study/{nct_id}"
        st.markdown(
            f"[View on ClinicalTrials.gov]({ctgov_url})"
        )
