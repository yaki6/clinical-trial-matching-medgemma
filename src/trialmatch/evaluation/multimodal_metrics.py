"""Evaluation metrics for multimodal (imaging + text) diagnosis tasks.

Scoring functions for comparing model-generated diagnoses and findings
against gold-standard references. Used for MedGemma multimodal evaluation.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


def score_diagnosis_exact(gold: str, predicted: str) -> bool:
    """Case-insensitive exact match after stripping whitespace.

    Args:
        gold: Gold-standard diagnosis string.
        predicted: Model-predicted diagnosis string.

    Returns:
        True if the stripped, lowercased strings are identical.
    """
    return gold.strip().lower() == predicted.strip().lower()


def score_diagnosis_substring(gold: str, predicted: str) -> bool:
    """Case-insensitive substring match in either direction.

    Checks whether gold is a substring of predicted, or predicted is a
    substring of gold. This captures cases where the model produces a
    more specific or more general version of the diagnosis.

    Args:
        gold: Gold-standard diagnosis string.
        predicted: Model-predicted diagnosis string.

    Returns:
        True if either string contains the other (case-insensitive).
        False if either string is empty after stripping.
    """
    gold_clean = gold.strip().lower()
    predicted_clean = predicted.strip().lower()

    if not gold_clean or not predicted_clean:
        return False

    return gold_clean in predicted_clean or predicted_clean in gold_clean


def score_findings_rouge(gold: str, predicted: str) -> dict[str, float]:
    """Compute ROUGE-L scores between gold and predicted findings text.

    Args:
        gold: Gold-standard findings narrative.
        predicted: Model-predicted findings narrative.

    Returns:
        Dict with keys "precision", "recall", "fmeasure", each a float
        in [0.0, 1.0]. Returns 0.0 for all if either input is empty.
    """
    gold_clean = gold.strip()
    predicted_clean = predicted.strip()

    if not gold_clean or not predicted_clean:
        return {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(gold_clean, predicted_clean)
    rouge_l = scores["rougeL"]

    return {
        "precision": float(rouge_l.precision),
        "recall": float(rouge_l.recall),
        "fmeasure": float(rouge_l.fmeasure),
    }


def parse_model_response(text: str) -> dict[str, str]:
    """Parse a structured model response with DIAGNOSIS: and FINDINGS: sections.

    Extracts content after each section header, stopping at the next
    recognized section header or end of text. Recognized section headers
    are DIAGNOSIS:, FINDINGS:, and DIFFERENTIAL:.

    Includes fallback strategies for models (like MedGemma 1.5 4B) that may
    not follow the expected format precisely.

    Args:
        text: Raw model response text.

    Returns:
        Dict with keys "diagnosis" and "findings", each a stripped string.
        Missing sections default to empty string.
    """
    result = {"diagnosis": "", "findings": ""}

    if not text or not text.strip():
        return result

    # Pattern: section header followed by content until next section or end.
    # Handles **DIAGNOSIS:** (bold markdown) and plain DIAGNOSIS:
    section_pattern = re.compile(
        r"(?:^|\n)\s*\**\s*(DIAGNOSIS|FINDINGS|DIFFERENTIAL|IMPRESSION)\s*\**\s*:\s*\**\s*",
        re.IGNORECASE,
    )

    # Find all section boundaries
    matches = list(section_pattern.finditer(text))

    for i, match in enumerate(matches):
        section_name = match.group(1).upper()
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()

        if section_name == "DIAGNOSIS":
            # Only take first DIAGNOSIS match (avoid overwriting with later noise)
            if not result["diagnosis"]:
                result["diagnosis"] = content
        elif section_name == "FINDINGS":
            if not result["findings"]:
                result["findings"] = content
        elif section_name == "IMPRESSION" and not result["findings"]:
            # MedGemma sometimes uses IMPRESSION instead of FINDINGS
            result["findings"] = content

    # Fallback: if no DIAGNOSIS section found, try to extract from full text
    if not result["diagnosis"]:
        # Look for patterns like "consistent with X" or "suggestive of X"
        diag_hints = re.findall(
            r"(?:consistent with|suggestive of|diagnosis[:\s]+(?:is)?|diagnosed as)\s+([^\n.]{5,80})",
            text,
            re.IGNORECASE,
        )
        if diag_hints:
            result["diagnosis"] = diag_hints[0].strip().rstrip(",;")

    # Fallback: if no FINDINGS, use the full text (minus JSON/code blocks)
    if not result["findings"] and len(text.strip()) > 50:
        # Remove JSON/code blocks that MedGemma sometimes hallucinates
        cleaned = re.sub(r"```[\s\S]*?```", "", text)
        cleaned = re.sub(r"\{[\s\S]*?\}", "", cleaned)
        cleaned = cleaned.strip()
        if len(cleaned) > 50:
            result["findings"] = cleaned[:2000]

    return result


def compute_aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary metrics across a list of per-case evaluation results.

    Args:
        results: List of per-case result dicts. Each dict is expected to
            have keys: exact_match (bool), substring_match (bool),
            rouge_recall (float), rouge_precision (float),
            rouge_fmeasure (float), llm_judge_score (str).

    Returns:
        Summary dict with:
        - accuracy_exact: fraction of exact matches
        - accuracy_substring: fraction of substring matches
        - mean_rouge_recall: mean ROUGE-L recall
        - mean_rouge_precision: mean ROUGE-L precision
        - mean_rouge_fmeasure: mean ROUGE-L F-measure
        - llm_judge_accuracy: fraction of "correct" llm_judge_score values
        - n: total number of cases
    """
    if not results:
        return {
            "accuracy_exact": 0.0,
            "accuracy_substring": 0.0,
            "mean_rouge_recall": 0.0,
            "mean_rouge_precision": 0.0,
            "mean_rouge_fmeasure": 0.0,
            "llm_judge_accuracy": 0.0,
            "n": 0,
        }

    n = len(results)

    accuracy_exact = sum(1 for r in results if r.get("exact_match")) / n
    accuracy_substring = sum(1 for r in results if r.get("substring_match")) / n

    mean_rouge_recall = sum(r.get("rouge_recall", 0.0) for r in results) / n
    mean_rouge_precision = sum(r.get("rouge_precision", 0.0) for r in results) / n
    mean_rouge_fmeasure = sum(r.get("rouge_fmeasure", 0.0) for r in results) / n

    llm_judge_correct = sum(
        1
        for r in results
        if str(r.get("llm_judge_score", "")).strip().lower() == "correct"
    )
    llm_judge_accuracy = llm_judge_correct / n

    return {
        "accuracy_exact": float(accuracy_exact),
        "accuracy_substring": float(accuracy_substring),
        "mean_rouge_recall": float(mean_rouge_recall),
        "mean_rouge_precision": float(mean_rouge_precision),
        "mean_rouge_fmeasure": float(mean_rouge_fmeasure),
        "llm_judge_accuracy": float(llm_judge_accuracy),
        "n": n,
    }
