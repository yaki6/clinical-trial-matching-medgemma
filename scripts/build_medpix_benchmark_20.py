#!/usr/bin/env python3
"""Build a 20-case MedPix multi-region benchmark dataset.

Stratified sampling: 4 cases per body region (5 regions = 20 total).
Filters to complete cases (diagnosis + findings + image on disk).
Uses first image per case matching that region.

Usage:
    python scripts/build_medpix_benchmark_20.py
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "ingest_design" / "patient-ehr-image-dataset" / "full_dataset.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmark"
OUTPUT_PATH = OUTPUT_DIR / "medpix_multiregion_20.json"

SEED = 42
CASES_PER_REGION = 4
REGIONS = [
    "Head",
    "Thorax",
    "Abdomen",
    "Spine and Muscles",
    "Reproductive and Urinary System",
]


def load_jsonl(path: Path) -> list[dict]:
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


def build_region_pools(rows: list[dict]) -> dict[str, list[dict]]:
    """Group eligible cases by their primary body region.

    A case is eligible if it has diagnosis, findings, and at least one image
    in the target region with file on disk.
    Each case appears in at most one region pool (its first matching image's region).
    """
    pools: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        if not (row.get("has_diagnosis") and row.get("has_findings")):
            continue

        # Group images by region
        region_images: dict[str, list[dict]] = defaultdict(list)
        for img in row.get("images", []):
            cat = img.get("location_category", "")
            if cat in REGIONS:
                region_images[cat].append(img)

        # Add case to each region it has images for
        for region in REGIONS:
            if region not in region_images:
                continue
            first_img = region_images[region][0]
            image_path = str(Path("ingest_design") / first_img["file_path"])
            full_path = PROJECT_ROOT / image_path

            if not full_path.is_file():
                continue

            pools[region].append({
                "uid": row["uid"],
                "history": row.get("history", ""),
                "gold_diagnosis": row.get("diagnosis", ""),
                "gold_findings": row.get("findings", ""),
                "image_path": image_path,
                "image_modality": first_img.get("modality", ""),
                "image_caption": first_img.get("caption", ""),
                "image_plane": first_img.get("plane", ""),
                "location_category": region,
                "title": row.get("title", ""),
                "age": first_img.get("age", ""),
                "sex": first_img.get("sex", ""),
            })

    return dict(pools)


def stratified_sample(
    pools: dict[str, list[dict]], per_region: int, seed: int
) -> list[dict]:
    """Sample `per_region` cases from each region, avoiding duplicate UIDs."""
    rng = random.Random(seed)
    sampled = []
    used_uids: set[str] = set()

    for region in REGIONS:
        candidates = [c for c in pools.get(region, []) if c["uid"] not in used_uids]
        n = min(per_region, len(candidates))
        if n < per_region:
            print(
                f"WARNING: {region} has only {len(candidates)} unique cases, "
                f"requested {per_region}",
                file=sys.stderr,
            )
        chosen = rng.sample(candidates, n)
        for c in chosen:
            used_uids.add(c["uid"])
        sampled.extend(chosen)

    return sampled


def main() -> int:
    if not DATASET_PATH.is_file():
        print(f"ERROR: Dataset not found at {DATASET_PATH}", file=sys.stderr)
        return 1

    print(f"Loading dataset from {DATASET_PATH}")
    rows = load_jsonl(DATASET_PATH)
    print(f"  Total cases: {len(rows)}")

    pools = build_region_pools(rows)
    print(f"\n  Region pools (eligible cases):")
    for region in REGIONS:
        count = len(pools.get(region, []))
        print(f"    {region}: {count}")

    sampled = stratified_sample(pools, CASES_PER_REGION, SEED)
    print(f"\n  Sampled {len(sampled)} cases (seed={SEED}, {CASES_PER_REGION}/region)")

    # Summary table
    print(f"\n  {'UID':<12} {'Region':<35} {'Modality':<25} {'Diagnosis'}")
    print(f"  {'-'*12} {'-'*35} {'-'*25} {'-'*40}")
    for c in sampled:
        diag = c["gold_diagnosis"][:40] if c["gold_diagnosis"] else "N/A"
        print(f"  {c['uid']:<12} {c['location_category']:<35} {c['image_modality']:<25} {diag}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(sampled, f, indent=2)

    print(f"\nBenchmark written to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
