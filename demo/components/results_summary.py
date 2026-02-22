"""Top-level results summary card component.

Renders an aggregate overview of trial matching results in patient-friendly
language. Intended to appear above individual trial cards so the user gets
a quick snapshot before scrolling through details.

The component adapts its messaging based on outcomes:
- If eligible trials exist: encouraging message to review details.
- If only uncertain trials: suggestion to discuss with healthcare provider.
- If all excluded: supportive message to consult provider about alternatives.
"""

from __future__ import annotations

import streamlit as st


def render_results_summary(
    total_trials: int,
    eligible_count: int,
    uncertain_count: int,
    excluded_count: int,
) -> None:
    """Render a top-level summary card of trial matching results.

    Parameters
    ----------
    total_trials:
        Total number of trials evaluated.
    eligible_count:
        Number of trials with ``ELIGIBLE`` verdict.
    uncertain_count:
        Number of trials with ``UNCERTAIN`` verdict.
    excluded_count:
        Number of trials with ``EXCLUDED`` verdict.
    """
    with st.container(border=True):
        st.markdown("### Your Trial Matching Results")

        # -- Subtitle: total count --
        trial_word = "trial" if total_trials == 1 else "trials"
        st.markdown(
            f"We found **{total_trials}** clinical {trial_word} for your condition."
        )

        st.markdown("")  # visual spacing

        # -- Metric lines with icons --
        eligible_word = "trial" if eligible_count == 1 else "trials"
        uncertain_word = "trial" if uncertain_count == 1 else "trials"
        excluded_word = "trial" if excluded_count == 1 else "trials"

        st.markdown(
            f":white_check_mark: **{eligible_count}** {eligible_word} you may be eligible for"
        )
        st.markdown(
            f":grey_question: **{uncertain_count}** {uncertain_word} need more information"
        )
        st.markdown(
            f":x: **{excluded_count}** {excluded_word} you likely don't qualify for"
        )

        st.markdown("")  # visual spacing

        # -- Contextual encouragement message --
        if eligible_count > 0:
            st.info("Scroll down for details on each trial.")
        elif uncertain_count > 0:
            st.warning(
                "Some trials need additional information. "
                "Discuss with your healthcare provider."
            )
        else:
            # All excluded (or zero trials, which is also handled gracefully)
            st.error(
                "Based on available information, no matching trials were found. "
                "Your healthcare provider can help explore other options."
            )
