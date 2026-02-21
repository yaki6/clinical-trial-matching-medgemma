"""Tests for demo harness data and multimodal adapter integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from trialmatch.ingest.profile_adapter import (
    adapt_harness_patient,
    get_image_path,
    load_demo_harness,
    merge_image_findings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PATH = REPO_ROOT / "data" / "sot" / "ingest" / "nsclc_demo_harness.json"


@pytest.fixture
def harness_patients():
    """Load the demo harness patients."""
    return load_demo_harness(HARNESS_PATH)


@pytest.fixture
def sample_image_findings():
    """Sample MedGemma image extraction results."""
    return {
        "findings": ["enlarged heart", "clear lung fields"],
        "tumor_characteristics": "3cm mass in right lower lobe",
        "impression": "Suspicious mass with possible cardiomegaly",
        "modality_observed": "CT",
    }


class TestLoadDemoHarness:
    def test_loads_5_patients(self, harness_patients):
        assert len(harness_patients) == 5

    def test_all_have_required_fields(self, harness_patients):
        required = {"topic_id", "source_dataset", "ingest_mode", "ehr_text", "profile_text", "key_facts"}
        for p in harness_patients:
            missing = required - set(p.keys())
            assert not missing, f"{p['topic_id']} missing {missing}"

    def test_3_multimodal_2_text(self, harness_patients):
        modes = [p["ingest_mode"] for p in harness_patients]
        assert modes.count("multimodal") == 3
        assert modes.count("text") == 2

    def test_topic_ids_match(self, harness_patients):
        ids = {p["topic_id"] for p in harness_patients}
        assert ids == {"mpx1016", "mpx1575", "mpx1875", "6031552-1", "6000873-1"}


class TestHarnessImagePatients:
    def test_multimodal_patients_have_image(self, harness_patients):
        for p in harness_patients:
            if p["ingest_mode"] == "multimodal":
                assert p.get("image") is not None, f"{p['topic_id']} missing image"
                assert "file_path" in p["image"]

    def test_text_patients_have_null_image(self, harness_patients):
        for p in harness_patients:
            if p["ingest_mode"] == "text":
                assert p.get("image") is None, f"{p['topic_id']} should have null image"

    def test_image_files_exist(self, harness_patients):
        for p in harness_patients:
            img_path = get_image_path(p, base_dir=REPO_ROOT)
            if p["ingest_mode"] == "multimodal":
                assert img_path is not None
                assert img_path.exists(), f"Image not found: {img_path}"
            else:
                assert img_path is None


class TestGetImagePath:
    def test_returns_path_for_multimodal(self, harness_patients):
        mpx1016 = next(p for p in harness_patients if p["topic_id"] == "mpx1016")
        path = get_image_path(mpx1016, base_dir=REPO_ROOT)
        assert path is not None
        assert path.name == "MPX1016_synpic34317.png"

    def test_returns_none_for_text(self, harness_patients):
        text_p = next(p for p in harness_patients if p["topic_id"] == "6031552-1")
        assert get_image_path(text_p) is None


class TestMergeImageFindings:
    def test_adds_medgemma_imaging_key(self, sample_image_findings):
        key_facts = {"primary_diagnosis": "Lung adenocarcinoma", "demographics": "43F"}
        merged = merge_image_findings(key_facts, sample_image_findings)
        assert "medgemma_imaging" in merged
        assert merged["medgemma_imaging"] == sample_image_findings

    def test_does_not_overwrite_existing(self, sample_image_findings):
        key_facts = {"primary_diagnosis": "Lung adenocarcinoma"}
        merged = merge_image_findings(key_facts, sample_image_findings)
        assert merged["primary_diagnosis"] == "Lung adenocarcinoma"

    def test_original_dict_unchanged(self, sample_image_findings):
        key_facts = {"primary_diagnosis": "Lung adenocarcinoma"}
        merge_image_findings(key_facts, sample_image_findings)
        assert "medgemma_imaging" not in key_facts


class TestAdaptHarnessPatient:
    def test_text_only_patient(self, harness_patients):
        text_p = next(p for p in harness_patients if p["topic_id"] == "6031552-1")
        note, kf = adapt_harness_patient(text_p)
        assert isinstance(note, str)
        assert len(note) > 0
        assert isinstance(kf, dict)
        assert "medgemma_imaging" not in kf

    def test_multimodal_with_findings(self, harness_patients, sample_image_findings):
        mpx = next(p for p in harness_patients if p["topic_id"] == "mpx1016")
        note, kf = adapt_harness_patient(mpx, image_findings=sample_image_findings)
        assert "medgemma_imaging" in kf
        assert kf["medgemma_imaging"]["impression"] == "Suspicious mass with possible cardiomegaly"

    def test_multimodal_without_findings(self, harness_patients):
        mpx = next(p for p in harness_patients if p["topic_id"] == "mpx1016")
        note, kf = adapt_harness_patient(mpx)
        assert "medgemma_imaging" not in kf


class TestGenerateWithImageMock:
    """Test MedGemmaAdapter.generate_with_image() with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_strips_prompt_echo(self):
        from trialmatch.models.medgemma import MedGemmaAdapter

        adapter = MedGemmaAdapter(hf_token="fake", endpoint_url="http://fake")

        prompt = "Describe findings."
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = [
            {
                "input_text": prompt,
                "generated_text": f"{prompt}\nThe image shows a chest X-ray with cardiomegaly.",
            }
        ]

        with (
            patch("requests.post", return_value=mock_response) as mock_post,
            patch("builtins.open", MagicMock()),
            patch("base64.b64encode") as mock_b64,
        ):
            mock_b64.return_value.decode.return_value = "fakebase64"
            result = await adapter.generate_with_image(prompt, Path("/fake/image.png"))

        assert "The image shows a chest X-ray with cardiomegaly." in result.text
        assert prompt not in result.text

    @pytest.mark.asyncio
    async def test_retries_on_503(self):
        from trialmatch.models.medgemma import MedGemmaAdapter

        adapter = MedGemmaAdapter(
            hf_token="fake",
            endpoint_url="http://fake",
            max_retries=3,
            retry_backoff=0.01,
            max_wait=0.1,
        )

        error_resp = MagicMock()
        error_resp.status_code = 503
        error_resp.raise_for_status.side_effect = Exception("503 Service Unavailable")

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json.return_value = [
            {"input_text": "test", "generated_text": "test\nResult here."}
        ]

        with (
            patch("requests.post", side_effect=[error_resp, ok_resp]) as mock_post,
            patch("builtins.open", MagicMock()),
            patch("base64.b64encode") as mock_b64,
        ):
            mock_b64.return_value.decode.return_value = "fakebase64"
            result = await adapter.generate_with_image("test", Path("/fake/image.png"))

        assert "Result here." in result.text
        assert mock_post.call_count == 2
