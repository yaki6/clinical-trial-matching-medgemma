#!/usr/bin/env python3
"""Extract TREC 2022 ground truth data from the combined markdown file.

Parses data/trec2022_ground_truth.md and writes standalone files:
  - data/trec2022_ground_truth/patient.jsonl
  - data/trec2022_ground_truth/qrels.tsv
  - data/trec2022_ground_truth/expected_outcomes.json

Validates: 647 qrels entries, 1 patient record, 118/20/509 label distribution.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "data" / "trec2022_ground_truth.md"
OUT_DIR = REPO_ROOT / "data" / "trec2022_ground_truth"

# File delimiters and line ranges (1-indexed, from grep analysis)
FILES = {
    "expected_outcomes.json": (182, 1769),
    "patient.jsonl": (1773, 1773),
    "qrels.tsv": (1780, 2428),
}


def extract():
    if not SOURCE.exists():
        print(f"ERROR: Source file not found: {SOURCE}")
        sys.exit(1)

    lines = SOURCE.read_text().splitlines()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, (start, end) in FILES.items():
        # Convert to 0-indexed
        content_lines = lines[start - 1 : end]
        # Strip trailing empty lines
        while content_lines and not content_lines[-1].strip():
            content_lines.pop()
        content = "\n".join(content_lines) + "\n"
        out_path = OUT_DIR / filename
        out_path.write_text(content)
        print(f"  Wrote {filename}: {len(content_lines)} lines")

    validate()


def validate():
    # Validate patient
    patient_path = OUT_DIR / "patient.jsonl"
    patient_lines = [l for l in patient_path.read_text().splitlines() if l.strip()]
    assert len(patient_lines) == 1, f"Expected 1 patient, got {len(patient_lines)}"
    patient = json.loads(patient_lines[0])
    assert patient["_id"] == "trec-20226"
    print(f"  Patient: {patient['_id']} ({patient['structured_profile']['diagnosis']})")

    # Validate qrels
    qrels_path = OUT_DIR / "qrels.tsv"
    qrels_lines = [l for l in qrels_path.read_text().splitlines() if l.strip()]
    header = qrels_lines[0]
    assert "query-id" in header, f"Bad qrels header: {header}"
    data_lines = qrels_lines[1:]
    assert len(data_lines) == 647, f"Expected 647 qrels, got {len(data_lines)}"

    # Check distribution
    scores = {}
    for line in data_lines:
        parts = line.split("\t")
        score = int(parts[2])
        scores[score] = scores.get(score, 0) + 1
    assert scores.get(2, 0) == 118, f"Expected 118 eligible, got {scores.get(2, 0)}"
    assert scores.get(1, 0) == 20, f"Expected 20 partial, got {scores.get(1, 0)}"
    assert scores.get(0, 0) == 509, f"Expected 509 excluded, got {scores.get(0, 0)}"
    print(f"  Qrels: {len(data_lines)} entries — eligible={scores[2]}, partial={scores[1]}, excluded={scores[0]}")

    # Validate expected_outcomes
    outcomes_path = OUT_DIR / "expected_outcomes.json"
    outcomes = json.loads(outcomes_path.read_text())
    assert "_metadata" in outcomes
    dist = outcomes["_metadata"]["label_distribution"]
    assert dist["total"] == 647
    print(f"  Expected outcomes: loaded ({dist['total']} total)")

    print("\nAll validations passed.")


if __name__ == "__main__":
    print("Extracting TREC 2022 ground truth...")
    extract()
