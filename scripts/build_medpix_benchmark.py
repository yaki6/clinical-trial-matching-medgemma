#!/usr/bin/env python3
"""Build a 10-case MedPix Thorax benchmark dataset.

Loads the full MedPix JSONL dataset, filters to cases with diagnosis + findings
+ at least one Thorax image, deterministically samples 10 cases (seed=42),
and writes the benchmark to data/benchmark/medpix_thorax_10.json.

Usage:
    python scripts/build_medpix_benchmark.py
"""

import json
import random
import sys
from pathlib import Path

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "ingest_design" / "patient-ehr-image-dataset" / "full_dataset.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmark"
OUTPUT_PATH = OUTPUT_DIR / "medpix_thorax_10.json"

SEED = 42
SAMPLE_SIZE = 10
TARGET_LOCATION_CATEGORY = "Thorax"


def load_jsonl(path: Path) -> list[dict]:
    """Load all rows from a JSONL file."""
    rows = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed line {line_num}: {e}", file=sys.stderr)
    return rows


def filter_thorax_cases(rows: list[dict]) -> list[dict]:
    """Filter to cases with diagnosis, findings, and at least one Thorax image.

    Returns list of dicts with the benchmark schema, using the first Thorax
    image per case.
    """
    filtered = []
    for row in rows:
        if not (row.get("has_diagnosis") and row.get("has_findings")):
            continue

        thorax_images = [
            img for img in row.get("images", [])
            if img.get("location_category") == TARGET_LOCATION_CATEGORY
        ]

        if not thorax_images:
            continue

        # Take the first Thorax image
        first_img = thorax_images[0]

        # Resolve image path: prepend ingest_design/ to the file_path
        image_path = str(Path("ingest_design") / first_img["file_path"])

        filtered.append({
            "uid": row["uid"],
            "history": row.get("history", ""),
            "gold_diagnosis": row.get("diagnosis", ""),
            "gold_findings": row.get("findings", ""),
            "image_path": image_path,
            "image_modality": first_img.get("modality", ""),
            "image_caption": first_img.get("caption", ""),
            "title": row.get("title", ""),
        })

    return filtered


def sample_cases(cases: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministically sample n cases using the given seed."""
    rng = random.Random(seed)
    if len(cases) < n:
        print(
            f"WARNING: Only {len(cases)} cases available, requested {n}. "
            f"Returning all.",
            file=sys.stderr,
        )
        return cases
    return rng.sample(cases, n)


def verify_image_paths(cases: list[dict], project_root: Path) -> tuple[int, list[str]]:
    """Verify all image paths exist on disk.

    Returns (verified_count, list_of_missing_paths).
    """
    verified = 0
    missing = []
    for case in cases:
        full_path = project_root / case["image_path"]
        if full_path.is_file():
            verified += 1
        else:
            missing.append(case["image_path"])
    return verified, missing


def main() -> int:
    """Build the MedPix Thorax benchmark dataset."""
    # Load dataset
    if not DATASET_PATH.is_file():
        print(f"ERROR: Dataset not found at {DATASET_PATH}", file=sys.stderr)
        return 1

    print(f"Loading dataset from {DATASET_PATH}")
    rows = load_jsonl(DATASET_PATH)
    print(f"  Total cases loaded: {len(rows)}")

    # Filter to Thorax cases with diagnosis + findings
    thorax_cases = filter_thorax_cases(rows)
    print(f"  Filtered Thorax cases (has_diagnosis + has_findings + Thorax image): {len(thorax_cases)}")

    if len(thorax_cases) < SAMPLE_SIZE:
        print(
            f"ERROR: Need at least {SAMPLE_SIZE} cases but only found {len(thorax_cases)}",
            file=sys.stderr,
        )
        return 1

    # Deterministic sample
    sampled = sample_cases(thorax_cases, SAMPLE_SIZE, SEED)
    print(f"  Sampled {len(sampled)} cases (seed={SEED})")

    # Verify image paths
    verified, missing = verify_image_paths(sampled, PROJECT_ROOT)
    print(f"  Image paths verified: {verified}/{len(sampled)}")
    if missing:
        for p in missing:
            print(f"    MISSING: {p}", file=sys.stderr)
        print("ERROR: Some image paths do not exist on disk.", file=sys.stderr)
        return 1

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(sampled, f, indent=2)

    print(f"\nBenchmark written to {OUTPUT_PATH}")
    print(f"  Cases: {len(sampled)}")
    print("  Image paths:")
    for case in sampled:
        print(f"    {case['image_path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
