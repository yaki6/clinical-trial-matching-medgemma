"""Smoke tests for MedGemma 27B HF Inference Endpoint.

Requires environment variables:
    MEDGEMMA_27B_ENDPOINT_URL  — endpoint URL from deploy_medgemma27b.py
    HF_TOKEN                   — HuggingFace token with endpoint access

Run with:
    MEDGEMMA_27B_ENDPOINT_URL=https://xxxx.endpoints.huggingface.cloud \\
    HF_TOKEN=hf_xxx \\
    uv run pytest tests/smoke/test_medgemma27b_endpoint.py -v -m smoke
"""

from __future__ import annotations

import os

import pytest
from huggingface_hub import InferenceClient

from trialmatch.models.medgemma import MedGemmaAdapter
from trialmatch.validate.evaluator import evaluate_criterion

ENDPOINT_URL_VAR = "MEDGEMMA_27B_ENDPOINT_URL"

# Skip all tests in this module if the endpoint URL is not set
pytestmark = [pytest.mark.smoke, pytest.mark.e2e]


@pytest.fixture(scope="module")
def endpoint_url() -> str:
    url = os.environ.get(ENDPOINT_URL_VAR, "")
    if not url or url == "REPLACE_WITH_27B_ENDPOINT_URL":
        pytest.skip(f"{ENDPOINT_URL_VAR} not set — deploy endpoint first")
    return url


@pytest.fixture(scope="module")
def hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        pytest.skip("HF_TOKEN not set")
    return token


@pytest.fixture(scope="module")
def adapter_27b(endpoint_url: str, hf_token: str) -> MedGemmaAdapter:
    return MedGemmaAdapter(
        hf_token=hf_token,
        endpoint_url=endpoint_url,
        model_name="medgemma-27b",
    )


def test_endpoint_health_check(endpoint_url: str, hf_token: str) -> None:
    """Endpoint must respond to a /v1/chat/completions call."""
    client = InferenceClient(
        model=endpoint_url,
        token=hf_token,
        headers={"X-Scale-Up-Timeout": "300"},
    )
    response = client.chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5,
    )
    assert response.choices[0].message.content, "Expected non-empty response from health check"


def test_criterion_evaluation_smoke(adapter_27b: MedGemmaAdapter) -> None:
    """One real criterion pair must return a valid MET/NOT_MET/UNKNOWN verdict."""
    import asyncio

    from trialmatch.models.schema import CriterionVerdict

    patient_note = (
        "Patient is a 65-year-old male with type 2 diabetes mellitus, "
        "hypertension, and stage 3 chronic kidney disease. "
        "Current HbA1c is 8.2%. No prior chemotherapy."
    )
    criterion_text = "Diagnosis of type 2 diabetes mellitus"
    criterion_type = "inclusion"

    result = asyncio.run(
        evaluate_criterion(
            patient_note=patient_note,
            criterion_text=criterion_text,
            criterion_type=criterion_type,
            adapter=adapter_27b,
            timeout_seconds=120.0,
        )
    )

    valid_verdicts = (
        CriterionVerdict.MET,
        CriterionVerdict.NOT_MET,
        CriterionVerdict.UNKNOWN,
    )
    assert result.verdict in valid_verdicts, f"Expected a valid verdict, got: {result.verdict!r}"
    assert result.model_response is not None
    assert result.model_response.text, "Expected non-empty model response text"
    assert result.model_response.token_count_estimated is False, (
        "Expected exact token counts from chat_completion"
    )


def test_chat_completion_format(endpoint_url: str, hf_token: str) -> None:
    """The /v1/chat/completions endpoint must accept message-format requests
    and return exact token counts in usage."""
    client = InferenceClient(
        model=endpoint_url,
        token=hf_token,
        headers={"X-Scale-Up-Timeout": "300"},
    )
    response = client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": (
                    "You are a clinical trial matching assistant.\n\n"
                    "Does this patient meet the criterion?\n"
                    "Patient: 65yo male with type 2 diabetes.\n"
                    "Criterion: Diagnosis of type 2 diabetes mellitus (inclusion)\n"
                    "Respond with MET, NOT_MET, or UNKNOWN."
                ),
            }
        ],
        max_tokens=20,
    )
    assert response.choices[0].message.content, "Expected non-empty chat completion response"
    assert response.usage.prompt_tokens > 0, "Expected exact prompt token count in usage"
    assert response.usage.completion_tokens > 0, "Expected exact completion token count in usage"
