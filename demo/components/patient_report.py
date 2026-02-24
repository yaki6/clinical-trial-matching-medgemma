"""Patient-facing doctor discussion report helpers.

Pure helpers to build a markdown handoff report from trial verdicts and
criterion-level reasoning. Designed for unit testing and reuse in Streamlit UI.
"""

from __future__ import annotations

import re
from typing import Any

_NEUTRAL_REASONING_FALLBACK = "Evaluation unavailable for this criterion in this run."
_FALLBACK_GAP_MESSAGE = (
    "Criterion-specific data missing: Review criterion text with your doctor and "
    "collect missing records."
)
_MAX_GAP_ITEMS = 5

_ERROR_REASONING_MARKERS = (
    "error:",
    "client error",
    "bad request",
    "traceback",
    "request id",
    "cuda error",
    "https://",
    "http://",
)

_GAP_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"ECOG|performance status|karnofsky", re.IGNORECASE),
        "Performance status missing: Ask clinician to document ECOG/Karnofsky score.",
    ),
    (
        re.compile(
            r"EGFR|ALK|ROS1|BRAF|KRAS|MET|RET|NTRK|HER2|PD-?L1|biomarker|mutation",
            re.IGNORECASE,
        ),
        "Molecular marker data missing: Bring tumor genomic profiling and biomarker results.",
    ),
    (
        re.compile(r"stage|metastatic|locally advanced|stage\s*[IIIIVX0-9]", re.IGNORECASE),
        "Stage detail unclear: Confirm formal staging from oncology note or pathology.",
    ),
    (
        re.compile(r"RECIST|measurable lesion|radiologic|radiologically|CT|MRI|PET", re.IGNORECASE),
        "Imaging eligibility evidence missing: Bring recent imaging/report showing measurable disease.",
    ),
    (
        re.compile(
            r"organ function|creatinine|bilirubin|AST|ALT|ANC|platelet|hemoglobin|labs?",
            re.IGNORECASE,
        ),
        "Lab or organ function criteria not verifiable: Bring recent CBC/CMP and required labs.",
    ),
    (
        re.compile(
            r"prior treatment|line of therapy|treatment-naive|progressed|osimertinib|chemotherapy|immunotherapy|TKI",
            re.IGNORECASE,
        ),
        "Treatment history detail missing: Bring treatment timeline and current line of therapy.",
    ),
    (
        re.compile(r"brain metast|CNS", re.IGNORECASE),
        "CNS status unclear: Bring brain imaging and metastasis status documentation.",
    ),
    (
        re.compile(r"pregnan|contracept|childbearing", re.IGNORECASE),
        "Reproductive eligibility info missing: Discuss pregnancy or contraception requirements.",
    ),
    (
        re.compile(r"QTc|LVEF|ECG|cardiac|heart failure", re.IGNORECASE),
        "Cardiac safety criteria not verifiable: Bring ECG/echo or cardiology documentation.",
    ),
]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _looks_like_system_error(reasoning: str) -> bool:
    lowered = reasoning.lower()
    return any(marker in lowered for marker in _ERROR_REASONING_MARKERS)


def _sanitize_reasoning(reasoning: str) -> str:
    text = _normalize_text(reasoning)
    if not text or _looks_like_system_error(text):
        return _NEUTRAL_REASONING_FALLBACK
    squashed = " ".join(text.split())
    if len(squashed) <= 220:
        return squashed
    return f"{squashed[:217].rstrip()}..."


def _truncate(text: str, max_len: int = 110) -> str:
    raw = _normalize_text(text)
    if len(raw) <= max_len:
        return raw
    return f"{raw[: max_len - 3].rstrip()}..."


def _trial_url(nct_id: str) -> str:
    return f"https://clinicaltrials.gov/study/{nct_id}"


def _format_phase(value: Any) -> str:
    if isinstance(value, list):
        items = [str(v).strip() for v in value if str(v).strip()]
        return ", ".join(items) if items else "N/A"
    text = _normalize_text(value)
    return text or "N/A"


def derive_uncertain_gap_checklist(criteria: list[dict]) -> list[str]:
    """Return deterministic follow-up checklist items from UNKNOWN criteria."""
    checklist: list[str] = []
    seen: set[str] = set()

    for criterion in criteria or []:
        if _normalize_text(criterion.get("verdict", "")).upper() != "UNKNOWN":
            continue

        criterion_text = _normalize_text(criterion.get("text", ""))
        message = _FALLBACK_GAP_MESSAGE

        for pattern, mapped_message in _GAP_RULES:
            if pattern.search(criterion_text):
                message = mapped_message
                break

        if message in seen:
            continue

        checklist.append(message)
        seen.add(message)

        if len(checklist) >= _MAX_GAP_ITEMS:
            break

    return checklist


def _build_eligible_summary(criteria: list[dict]) -> list[str]:
    met_criteria = [
        c for c in criteria if _normalize_text(c.get("verdict", "")).upper() == "MET"
    ]
    if not met_criteria:
        return ["Marked eligible based on available criteria in this run."]

    summary = [f"Meets {len(met_criteria)} reviewed eligibility criteria in this run."]
    for criterion in met_criteria[:2]:
        summary.append(f"Matched criterion: {_truncate(criterion.get('text', ''))}")

    reasoning = _sanitize_reasoning(met_criteria[0].get("reasoning", ""))
    summary.append(f"Model reasoning: {reasoning}")
    return summary


def _build_uncertain_summary(criteria: list[dict]) -> list[str]:
    unknown_criteria = [
        c for c in criteria if _normalize_text(c.get("verdict", "")).upper() == "UNKNOWN"
    ]
    if not unknown_criteria:
        return ["Marked uncertain because criteria could not be fully verified in this run."]

    summary = [
        f"{len(unknown_criteria)} criteria need more information before eligibility can be confirmed."
    ]
    for criterion in unknown_criteria[:2]:
        summary.append(f"Needs follow-up: {_truncate(criterion.get('text', ''))}")

    reasoning = _sanitize_reasoning(unknown_criteria[0].get("reasoning", ""))
    summary.append(f"Current evaluation note: {reasoning}")
    return summary


def _normalize_criteria(criteria: Any) -> list[dict]:
    normalized: list[dict] = []
    for criterion in criteria or []:
        if not isinstance(criterion, dict):
            continue
        normalized.append(
            {
                "text": _normalize_text(criterion.get("text", "")),
                "type": _normalize_text(criterion.get("type", "")).lower() or "inclusion",
                "verdict": _normalize_text(criterion.get("verdict", "")).upper(),
                "reasoning": _sanitize_reasoning(_normalize_text(criterion.get("reasoning", ""))),
                "stage1_reasoning": _normalize_text(criterion.get("stage1_reasoning", "")),
            }
        )
    return normalized


def build_patient_report_data(trials: list[dict]) -> dict:
    """Build report-ready data for ELIGIBLE + UNCERTAIN trial sections."""
    eligible_trials: list[dict] = []
    uncertain_trials: list[dict] = []

    for trial in trials or []:
        verdict = _normalize_text(trial.get("verdict", "")).upper()
        if verdict not in {"ELIGIBLE", "UNCERTAIN"}:
            continue

        nct_id = _normalize_text(trial.get("nct_id", ""))
        criteria = _normalize_criteria(trial.get("criteria", []))
        entry = {
            "nct_id": nct_id,
            "title": _normalize_text(trial.get("title", "")) or "Untitled trial",
            "phase": _format_phase(trial.get("phase", "")),
            "status": _normalize_text(trial.get("status", "")) or "Unknown",
            "verdict": verdict,
            "url": _trial_url(nct_id) if nct_id else "",
            "criteria": criteria,
        }

        if verdict == "ELIGIBLE":
            entry["reasoning_summary"] = _build_eligible_summary(criteria)
            eligible_trials.append(entry)
        else:
            entry["reasoning_summary"] = _build_uncertain_summary(criteria)
            entry["gap_checklist"] = derive_uncertain_gap_checklist(criteria)
            uncertain_trials.append(entry)

    return {
        "eligible_count": len(eligible_trials),
        "uncertain_count": len(uncertain_trials),
        "included_trial_count": len(eligible_trials) + len(uncertain_trials),
        "eligible_trials": eligible_trials,
        "uncertain_trials": uncertain_trials,
    }


def render_patient_report_markdown(report_data: dict) -> str:
    """Render report data as patient-facing markdown."""
    eligible_trials = report_data.get("eligible_trials", [])
    uncertain_trials = report_data.get("uncertain_trials", [])

    lines: list[str] = [
        "## Doctor Discussion Report",
        "",
        "> Screening summary only. This report helps discussion with your doctor and is not a diagnosis.",
        "",
        f"### Likely Matches (ELIGIBLE): {len(eligible_trials)}",
    ]

    if not eligible_trials:
        lines.append("- No likely matches in this run.")
    else:
        for idx, trial in enumerate(eligible_trials, start=1):
            lines.extend(
                [
                    f"#### {idx}. [{trial['nct_id']}]({trial['url']}) - {trial['title']}",
                    f"- Status: {trial['status']} | Phase: {trial['phase']}",
                    "- Why this may fit:",
                ]
            )
            for item in trial.get("reasoning_summary", []):
                lines.append(f"  - {item}")
            lines.append("")

    lines.extend(
        [
            f"### Needs More Information (UNCERTAIN): {len(uncertain_trials)}",
        ]
    )
    if not uncertain_trials:
        lines.append("- No uncertain matches in this run.")
    else:
        for idx, trial in enumerate(uncertain_trials, start=1):
            lines.extend(
                [
                    f"#### {idx}. [{trial['nct_id']}]({trial['url']}) - {trial['title']}",
                    f"- Status: {trial['status']} | Phase: {trial['phase']}",
                    "- Why this needs more information:",
                ]
            )
            for item in trial.get("reasoning_summary", []):
                lines.append(f"  - {item}")

            checklist = trial.get("gap_checklist", [])
            if checklist:
                lines.append("- What to discuss or bring:")
                for item in checklist:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"- What to discuss or bring: {_FALLBACK_GAP_MESSAGE}")
            lines.append("")

    return "\n".join(lines).strip()
