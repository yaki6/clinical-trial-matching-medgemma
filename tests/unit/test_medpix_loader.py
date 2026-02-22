"""Unit tests for the MedPix Thorax benchmark dataset.

Validates the generated benchmark file at data/benchmark/medpix_thorax_10.json
produced by scripts/build_medpix_benchmark.py.
"""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark" / "medpix_thorax_10.json"

REQUIRED_FIELDS = {
    "uid",
    "history",
    "gold_diagnosis",
    "gold_findings",
    "image_path",
    "image_modality",
    "image_caption",
}


@pytest.fixture(scope="module")
def benchmark_data() -> list[dict]:
    """Load the benchmark JSON file once for all tests."""
    assert BENCHMARK_PATH.is_file(), (
        f"Benchmark file not found at {BENCHMARK_PATH}. "
        f"Run: python scripts/build_medpix_benchmark.py"
    )
    with open(BENCHMARK_PATH) as f:
        data = json.load(f)
    return data


class TestBenchmarkFileValidity:
    """Tests for the benchmark output file structure."""

    def test_is_valid_json_list(self, benchmark_data: list[dict]) -> None:
        """Output file is valid JSON containing a list."""
        assert isinstance(benchmark_data, list)

    def test_has_10_items(self, benchmark_data: list[dict]) -> None:
        """Output contains exactly 10 cases."""
        assert len(benchmark_data) == 10

    def test_each_item_is_dict(self, benchmark_data: list[dict]) -> None:
        """Each item in the list is a dictionary."""
        for i, item in enumerate(benchmark_data):
            assert isinstance(item, dict), f"Item {i} is not a dict: {type(item)}"


class TestRequiredFields:
    """Tests for presence and type of required fields."""

    def test_all_required_fields_present(self, benchmark_data: list[dict]) -> None:
        """Each item has all required fields."""
        for i, item in enumerate(benchmark_data):
            missing = REQUIRED_FIELDS - set(item.keys())
            assert not missing, (
                f"Item {i} (uid={item.get('uid', '?')}) missing fields: {missing}"
            )

    def test_gold_diagnosis_non_empty(self, benchmark_data: list[dict]) -> None:
        """All gold_diagnosis values are non-empty strings."""
        for i, item in enumerate(benchmark_data):
            val = item["gold_diagnosis"]
            assert isinstance(val, str), (
                f"Item {i}: gold_diagnosis is {type(val)}, expected str"
            )
            assert len(val.strip()) > 0, (
                f"Item {i} (uid={item['uid']}): gold_diagnosis is empty"
            )

    def test_gold_findings_non_empty(self, benchmark_data: list[dict]) -> None:
        """All gold_findings values are non-empty strings."""
        for i, item in enumerate(benchmark_data):
            val = item["gold_findings"]
            assert isinstance(val, str), (
                f"Item {i}: gold_findings is {type(val)}, expected str"
            )
            assert len(val.strip()) > 0, (
                f"Item {i} (uid={item['uid']}): gold_findings is empty"
            )


class TestImagePaths:
    """Tests for image path validity."""

    def test_all_image_paths_exist(self, benchmark_data: list[dict]) -> None:
        """All image_path files exist on disk."""
        for i, item in enumerate(benchmark_data):
            full_path = PROJECT_ROOT / item["image_path"]
            assert full_path.is_file(), (
                f"Item {i} (uid={item['uid']}): image not found at {full_path}"
            )

    def test_image_paths_are_png(self, benchmark_data: list[dict]) -> None:
        """All image paths end with .png."""
        for i, item in enumerate(benchmark_data):
            assert item["image_path"].endswith(".png"), (
                f"Item {i}: image_path does not end with .png: {item['image_path']}"
            )


class TestUniqueness:
    """Tests for data uniqueness."""

    def test_uid_values_unique(self, benchmark_data: list[dict]) -> None:
        """All uid values are unique (no duplicate cases)."""
        uids = [item["uid"] for item in benchmark_data]
        assert len(uids) == len(set(uids)), (
            f"Duplicate UIDs found: {[u for u in uids if uids.count(u) > 1]}"
        )
