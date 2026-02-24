#!/usr/bin/env python3
"""Fetch trial criteria from CT.gov API for TREC VALIDATE sample.

Stratified-samples 10 trials from qrels (seed=42), fetches eligibility
criteria, parses into inclusion/exclusion lists, and caches to JSON.

Usage:
    uv run python scripts/fetch_trec_trial_criteria.py
"""

import asyncio
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "trec2022_ground_truth"
CACHE_PATH = DATA_DIR / "trial_criteria_cache.json"

N_PER_LEVEL = 4  # ~3-4 per level → 10-12 total
SEED = 42


def load_qrels() -> dict[str, int]:
    qrels = {}
    for line in (DATA_DIR / "qrels.tsv").read_text().splitlines():
        if line.startswith("query-id") or not line.strip():
            continue
        parts = line.split("\t")
        qrels[parts[1]] = int(parts[2])
    return qrels


def sample_trials(qrels: dict[str, int], n_per_level: int, seed: int) -> dict[str, int]:
    """Stratified sample: n_per_level from each qrel level."""
    rng = random.Random(seed)
    sampled = {}
    for level in [0, 1, 2]:
        ncts = [nct for nct, s in qrels.items() if s == level]
        rng.shuffle(ncts)
        for nct in ncts[:n_per_level]:
            sampled[nct] = level
    return sampled


def parse_criteria(raw_text: str) -> tuple[list[str], list[str]]:
    """Parse eligibility criteria text into inclusion and exclusion lists.

    Handles two formats:
    1. Standard: "Inclusion Criteria:" / "Exclusion Criteria:" sections
    2. Legacy NCI: "DISEASE CHARACTERISTICS:" / "PATIENT CHARACTERISTICS:" /
       "PRIOR CONCURRENT THERAPY:" (all treated as inclusion-like criteria)
       with inline exclusion markers like "No ...", "Must not ..."
    """
    inclusion = []
    exclusion = []

    # Try standard format first
    has_standard = ("inclusion criteria" in raw_text.lower() or
                    "exclusion criteria" in raw_text.lower())

    if has_standard:
        current_list = None
        for line in raw_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            lower = line.lower()
            if "inclusion criteria" in lower:
                current_list = inclusion
                continue
            elif "exclusion criteria" in lower:
                current_list = exclusion
                continue
            cleaned = re.sub(r"^[-*•]\s*", "", line).strip()
            if cleaned and current_list is not None:
                current_list.append(cleaned)
    else:
        # Legacy NCI format: treat all content as criteria
        # Section headers become context, bullet points become criteria
        # "No ..." / "Must not ..." patterns → exclusion, rest → inclusion
        current_section = ""
        for line in raw_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Section headers (all caps with colon)
            if re.match(r"^[A-Z\s]+:$", line):
                current_section = line.rstrip(":")
                continue
            cleaned = re.sub(r"^[-*•]\s*", "", line).strip()
            if not cleaned:
                continue
            # Heuristic: negative patterns → exclusion
            lower = cleaned.lower()
            if (lower.startswith("no ") or lower.startswith("not ") or
                    "must not" in lower or "excluded" in lower or
                    "ineligible" in lower):
                exclusion.append(cleaned)
            else:
                inclusion.append(cleaned)

    return inclusion, exclusion


async def fetch_all(sampled: dict[str, int]) -> dict:
    from trialmatch.prescreen.ctgov_client import CTGovClient

    # Load existing cache to avoid re-fetching
    cache = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())
        print(f"Loaded {len(cache)} cached entries")

    client = CTGovClient()
    errors = []

    for nct_id, qrel_score in sorted(sampled.items()):
        if nct_id in cache:
            print(f"  [CACHED] {nct_id} (qrel={qrel_score})")
            continue

        try:
            raw = await client.get_details(nct_id)
            proto = raw.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            elig_mod = proto.get("eligibilityModule", {})

            criteria_text = elig_mod.get("eligibilityCriteria", "")
            inclusion, exclusion = parse_criteria(criteria_text)

            last_update = status_mod.get("lastUpdatePostDateStruct", {}).get("date", "")
            # Flag criteria drift: updated after 2022
            drift = False
            if last_update:
                try:
                    year = int(last_update.split("-")[0])
                    drift = year > 2022
                except (ValueError, IndexError):
                    pass

            cache[nct_id] = {
                "brief_title": id_mod.get("briefTitle", ""),
                "eligibility_criteria": criteria_text,
                "inclusion_criteria": inclusion,
                "exclusion_criteria": exclusion,
                "status": status_mod.get("overallStatus", ""),
                "last_update": last_update,
                "criteria_drift_warning": drift,
                "qrel_score": qrel_score,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            drift_marker = " [DRIFT!]" if drift else ""
            print(f"  [OK] {nct_id} (qrel={qrel_score}) — {len(inclusion)} incl, {len(exclusion)} excl{drift_marker}")

        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "Not Found" in error_msg:
                print(f"  [404] {nct_id} (qrel={qrel_score}) — trial removed from CT.gov")
            else:
                print(f"  [ERR] {nct_id} (qrel={qrel_score}) — {error_msg}")
            errors.append({"nct_id": nct_id, "error": error_msg})

    await client.aclose()

    # Save cache
    CACHE_PATH.write_text(json.dumps(cache, indent=2))
    print(f"\nCached {len(cache)} trials to {CACHE_PATH}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e['nct_id']}: {e['error']}")

    return cache


async def main():
    qrels = load_qrels()
    sampled = sample_trials(qrels, N_PER_LEVEL, SEED)

    print(f"Sampled {len(sampled)} trials (seed={SEED}):")
    for level in [0, 1, 2]:
        ncts = [n for n, s in sampled.items() if s == level]
        print(f"  qrel={level}: {ncts}")

    print(f"\nFetching from CT.gov API...")
    cache = await fetch_all(sampled)

    # Summary
    print(f"\n{'='*40}")
    print(f"SUMMARY")
    print(f"{'='*40}")
    for nct_id in sorted(sampled.keys()):
        if nct_id in cache:
            c = cache[nct_id]
            drift = " [CRITERIA DRIFT]" if c.get("criteria_drift_warning") else ""
            print(f"  {nct_id} qrel={sampled[nct_id]} status={c['status']} "
                  f"incl={len(c['inclusion_criteria'])} excl={len(c['exclusion_criteria'])}{drift}")
        else:
            print(f"  {nct_id} qrel={sampled[nct_id]} — NOT FETCHED")


if __name__ == "__main__":
    asyncio.run(main())
