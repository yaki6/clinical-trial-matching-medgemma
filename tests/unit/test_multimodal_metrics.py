"""Tests for multimodal evaluation metrics."""

import pytest

from trialmatch.evaluation.multimodal_metrics import (
    compute_aggregate_metrics,
    parse_model_response,
    score_diagnosis_exact,
    score_diagnosis_substring,
    score_findings_rouge,
)


# --- score_diagnosis_exact ---


class TestScoreDiagnosisExact:
    def test_exact_match(self):
        assert score_diagnosis_exact("Adenocarcinoma", "Adenocarcinoma") is True

    def test_case_insensitive(self):
        assert score_diagnosis_exact("adenocarcinoma", "ADENOCARCINOMA") is True

    def test_strips_whitespace(self):
        assert score_diagnosis_exact("  Adenocarcinoma  ", "Adenocarcinoma") is True

    def test_different_strings(self):
        assert score_diagnosis_exact("Adenocarcinoma", "Squamous Cell Carcinoma") is False

    def test_empty_strings_match(self):
        assert score_diagnosis_exact("", "") is True

    def test_substring_is_not_exact(self):
        assert score_diagnosis_exact("Adenocarcinoma", "Adenocarcinoma of the Lung") is False


# --- score_diagnosis_substring ---


class TestScoreDiagnosisSubstring:
    def test_gold_in_predicted(self):
        assert score_diagnosis_substring("Adenocarcinoma", "Adenocarcinoma of the Lung") is True

    def test_predicted_in_gold(self):
        assert score_diagnosis_substring("Adenocarcinoma of the Lung", "Adenocarcinoma") is True

    def test_exact_match_is_substring(self):
        assert score_diagnosis_substring("Adenocarcinoma", "Adenocarcinoma") is True

    def test_unrelated_strings(self):
        assert score_diagnosis_substring("Adenocarcinoma", "Melanoma") is False

    def test_case_insensitive(self):
        assert score_diagnosis_substring("adenocarcinoma", "ADENOCARCINOMA OF THE LUNG") is True

    def test_empty_gold_returns_false(self):
        assert score_diagnosis_substring("", "Adenocarcinoma") is False

    def test_empty_predicted_returns_false(self):
        assert score_diagnosis_substring("Adenocarcinoma", "") is False

    def test_both_empty_returns_false(self):
        assert score_diagnosis_substring("", "") is False

    def test_whitespace_only_returns_false(self):
        assert score_diagnosis_substring("   ", "Adenocarcinoma") is False


# --- score_findings_rouge ---


class TestScoreFindingsRouge:
    def test_identical_text_perfect_scores(self):
        text = "There is a 3cm spiculated mass in the right upper lobe."
        result = score_findings_rouge(text, text)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["fmeasure"] == pytest.approx(1.0)

    def test_partial_overlap(self):
        gold = "There is a 3cm spiculated mass in the right upper lobe."
        predicted = "A spiculated mass was found in the right upper lobe, measuring approximately 3cm."
        result = score_findings_rouge(gold, predicted)
        # Partial overlap should give scores between 0 and 1
        assert 0.0 < result["fmeasure"] < 1.0
        assert 0.0 < result["recall"] < 1.0
        assert 0.0 < result["precision"] < 1.0

    def test_no_overlap(self):
        gold = "Normal chest radiograph with no acute abnormality."
        predicted = "Severe cardiomegaly with bilateral pleural effusions."
        result = score_findings_rouge(gold, predicted)
        # Very low but may not be exactly 0 due to stemming/stopwords
        assert result["fmeasure"] < 0.5

    def test_empty_gold_returns_zeros(self):
        result = score_findings_rouge("", "Some predicted findings.")
        assert result == {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    def test_empty_predicted_returns_zeros(self):
        result = score_findings_rouge("Some gold findings.", "")
        assert result == {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    def test_both_empty_returns_zeros(self):
        result = score_findings_rouge("", "")
        assert result == {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    def test_whitespace_only_returns_zeros(self):
        result = score_findings_rouge("   ", "  \n  ")
        assert result == {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}

    def test_returns_float_values(self):
        result = score_findings_rouge("mass in lung", "mass in lung lobe")
        for key in ("precision", "recall", "fmeasure"):
            assert isinstance(result[key], float)


# --- parse_model_response ---


class TestParseModelResponse:
    def test_complete_response(self):
        text = (
            "DIAGNOSIS: Adenocarcinoma of the Lung\n"
            "\n"
            "FINDINGS: There is a 3cm spiculated mass in the right upper lobe "
            "with associated mediastinal lymphadenopathy.\n"
            "\n"
            "DIFFERENTIAL: Small cell carcinoma, Squamous cell carcinoma"
        )
        result = parse_model_response(text)
        assert result["diagnosis"] == "Adenocarcinoma of the Lung"
        assert "3cm spiculated mass" in result["findings"]
        assert "DIFFERENTIAL" not in result["findings"]

    def test_only_diagnosis(self):
        text = "DIAGNOSIS: Pneumothorax"
        result = parse_model_response(text)
        assert result["diagnosis"] == "Pneumothorax"
        assert result["findings"] == ""

    def test_only_findings(self):
        text = "FINDINGS: Bilateral pleural effusions with cardiomegaly."
        result = parse_model_response(text)
        assert result["diagnosis"] == ""
        assert "pleural effusions" in result["findings"]

    def test_empty_input(self):
        result = parse_model_response("")
        assert result == {"diagnosis": "", "findings": ""}

    def test_whitespace_only_input(self):
        result = parse_model_response("   \n\n  ")
        assert result == {"diagnosis": "", "findings": ""}

    def test_no_recognized_sections(self):
        text = "This is just free text without any recognized section headers."
        result = parse_model_response(text)
        assert result["diagnosis"] == ""
        assert result["findings"] == ""

    def test_case_insensitive_headers(self):
        text = "diagnosis: Pneumonia\nfindings: Consolidation in left lower lobe."
        result = parse_model_response(text)
        assert result["diagnosis"] == "Pneumonia"
        assert "Consolidation" in result["findings"]

    def test_extra_whitespace_around_content(self):
        text = "DIAGNOSIS:    Adenocarcinoma   \n\nFINDINGS:   Mass in lung   "
        result = parse_model_response(text)
        assert result["diagnosis"] == "Adenocarcinoma"
        assert result["findings"] == "Mass in lung"

    def test_findings_before_diagnosis(self):
        """Order of sections should not matter."""
        text = (
            "FINDINGS: Large pleural effusion.\n"
            "DIAGNOSIS: Malignant pleural effusion"
        )
        result = parse_model_response(text)
        assert result["diagnosis"] == "Malignant pleural effusion"
        assert "Large pleural effusion" in result["findings"]


# --- compute_aggregate_metrics ---


class TestComputeAggregateMetrics:
    def test_perfect_results(self):
        results = [
            {
                "exact_match": True,
                "substring_match": True,
                "rouge_recall": 1.0,
                "rouge_precision": 1.0,
                "rouge_fmeasure": 1.0,
                "llm_judge_score": "correct",
            },
            {
                "exact_match": True,
                "substring_match": True,
                "rouge_recall": 0.8,
                "rouge_precision": 0.9,
                "rouge_fmeasure": 0.85,
                "llm_judge_score": "correct",
            },
        ]
        m = compute_aggregate_metrics(results)
        assert m["accuracy_exact"] == 1.0
        assert m["accuracy_substring"] == 1.0
        assert m["mean_rouge_recall"] == pytest.approx(0.9)
        assert m["mean_rouge_precision"] == pytest.approx(0.95)
        assert m["mean_rouge_fmeasure"] == pytest.approx(0.925)
        assert m["llm_judge_accuracy"] == 1.0
        assert m["n"] == 2

    def test_mixed_results(self):
        results = [
            {
                "exact_match": True,
                "substring_match": True,
                "rouge_recall": 0.8,
                "rouge_precision": 0.7,
                "rouge_fmeasure": 0.75,
                "llm_judge_score": "correct",
            },
            {
                "exact_match": False,
                "substring_match": True,
                "rouge_recall": 0.4,
                "rouge_precision": 0.5,
                "rouge_fmeasure": 0.45,
                "llm_judge_score": "incorrect",
            },
            {
                "exact_match": False,
                "substring_match": False,
                "rouge_recall": 0.2,
                "rouge_precision": 0.3,
                "rouge_fmeasure": 0.24,
                "llm_judge_score": "correct",
            },
        ]
        m = compute_aggregate_metrics(results)
        assert m["accuracy_exact"] == pytest.approx(1 / 3)
        assert m["accuracy_substring"] == pytest.approx(2 / 3)
        assert m["mean_rouge_recall"] == pytest.approx((0.8 + 0.4 + 0.2) / 3)
        assert m["mean_rouge_precision"] == pytest.approx((0.7 + 0.5 + 0.3) / 3)
        assert m["mean_rouge_fmeasure"] == pytest.approx((0.75 + 0.45 + 0.24) / 3)
        assert m["llm_judge_accuracy"] == pytest.approx(2 / 3)
        assert m["n"] == 3

    def test_empty_list(self):
        m = compute_aggregate_metrics([])
        assert m["accuracy_exact"] == 0.0
        assert m["accuracy_substring"] == 0.0
        assert m["mean_rouge_recall"] == 0.0
        assert m["mean_rouge_precision"] == 0.0
        assert m["mean_rouge_fmeasure"] == 0.0
        assert m["llm_judge_accuracy"] == 0.0
        assert m["n"] == 0

    def test_llm_judge_case_insensitive(self):
        """llm_judge_score comparison should be case-insensitive."""
        results = [
            {
                "exact_match": True,
                "substring_match": True,
                "rouge_recall": 1.0,
                "rouge_precision": 1.0,
                "rouge_fmeasure": 1.0,
                "llm_judge_score": "Correct",
            },
            {
                "exact_match": True,
                "substring_match": True,
                "rouge_recall": 1.0,
                "rouge_precision": 1.0,
                "rouge_fmeasure": 1.0,
                "llm_judge_score": "CORRECT",
            },
        ]
        m = compute_aggregate_metrics(results)
        assert m["llm_judge_accuracy"] == 1.0

    def test_missing_keys_default_to_falsy(self):
        """Missing keys in result dicts should not crash."""
        results = [
            {},  # all keys missing
        ]
        m = compute_aggregate_metrics(results)
        assert m["accuracy_exact"] == 0.0
        assert m["accuracy_substring"] == 0.0
        assert m["mean_rouge_recall"] == 0.0
        assert m["llm_judge_accuracy"] == 0.0
        assert m["n"] == 1

    def test_returns_float_values(self):
        """All numeric values in the output should be floats."""
        results = [
            {
                "exact_match": True,
                "substring_match": True,
                "rouge_recall": 1.0,
                "rouge_precision": 1.0,
                "rouge_fmeasure": 1.0,
                "llm_judge_score": "correct",
            },
        ]
        m = compute_aggregate_metrics(results)
        for key in (
            "accuracy_exact",
            "accuracy_substring",
            "mean_rouge_recall",
            "mean_rouge_precision",
            "mean_rouge_fmeasure",
            "llm_judge_accuracy",
        ):
            assert isinstance(m[key], float), f"{key} is not float: {type(m[key])}"
