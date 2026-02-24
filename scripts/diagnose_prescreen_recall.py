#!/usr/bin/env python3
"""Diagnose PRESCREEN recall: determine theoretical ceiling via CT.gov API.

Tests:
1. Direct lookup: which gold eligible NCT IDs still exist in CT.gov?
2. Paginated search: how many gold trials appear in condition=mesothelioma results?
3. Search strategy: what queries find the most gold trials?

Usage:
    uv run python scripts/diagnose_prescreen_recall.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "trec2022_ground_truth"

ALL_STATUSES = [
    "RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING",
    "COMPLETED", "TERMINATED", "WITHDRAWN", "UNKNOWN",
]


def load_qrels() -> dict[str, int]:
    qrels = {}
    for line in (DATA_DIR / "qrels.tsv").read_text().splitlines():
        if line.startswith("query-id") or not line.strip():
            continue
        parts = line.split("\t")
        qrels[parts[1]] = int(parts[2])
    return qrels


async def test_direct_lookup(client, eligible_ncts: list[str]):
    """Test 1: Check if gold eligible NCT IDs exist in CT.gov via direct lookup."""
    print("\n" + "=" * 60)
    print("TEST 1: Direct NCT ID lookup (sample of 20)")
    print("=" * 60)

    # Sample 20 trials spanning the NCT ID range
    sample = eligible_ncts[::max(1, len(eligible_ncts) // 20)][:20]
    found = []
    not_found = []
    statuses = {}

    for nct_id in sample:
        try:
            result = await client.get_details(nct_id)
            proto = result.get("protocolSection", {})
            status = proto.get("statusModule", {}).get("overallStatus", "UNKNOWN")
            found.append(nct_id)
            statuses[nct_id] = status
            print(f"  [OK] {nct_id} — {status}")
        except Exception as e:
            not_found.append(nct_id)
            print(f"  [MISS] {nct_id} — {e}")
        # Small delay to avoid rate limiting
        await asyncio.sleep(1.6)

    print(f"\nDirect lookup: {len(found)}/{len(sample)} found")
    status_counts = {}
    for s in statuses.values():
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"Status distribution: {status_counts}")
    return found, not_found, statuses


async def test_paginated_search(client, eligible_set: set, condition: str = "mesothelioma"):
    """Test 2: Paginate through ALL condition search results, check overlap with gold."""
    print("\n" + "=" * 60)
    print(f"TEST 2: Paginated search (condition={condition}, all statuses)")
    print("=" * 60)

    all_nct_ids = []
    page_token = None
    page = 0

    while True:
        result = await client.search(
            condition=condition,
            status=ALL_STATUSES,
            page_size=100,
            page_token=page_token,
        )
        studies = result.get("studies", [])
        if not studies:
            break

        for s in studies:
            nct_id = s.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
            if nct_id:
                all_nct_ids.append(nct_id)

        page += 1
        found_so_far = len(set(all_nct_ids) & eligible_set)
        print(f"  Page {page}: {len(studies)} studies (total: {len(all_nct_ids)}, found eligible: {found_so_far})")

        page_token = result.get("nextPageToken")
        if not page_token:
            break
        await asyncio.sleep(1.6)

    all_set = set(all_nct_ids)
    overlap = all_set & eligible_set
    print(f"\nTotal studies retrieved: {len(all_nct_ids)}")
    print(f"Gold eligible found: {len(overlap)}/{len(eligible_set)} ({len(overlap)/len(eligible_set):.1%})")
    if overlap:
        print(f"Found: {sorted(overlap)[:20]}{'...' if len(overlap) > 20 else ''}")

    missed = eligible_set - all_set
    if missed:
        print(f"Missed: {len(missed)} trials")
        print(f"Sample missed: {sorted(missed)[:10]}")

    return all_nct_ids, overlap, missed


async def test_broader_searches(client, eligible_set: set):
    """Test 3: Try different search conditions to find more gold trials."""
    print("\n" + "=" * 60)
    print("TEST 3: Alternative search strategies")
    print("=" * 60)

    search_configs = [
        {"condition": "mesothelioma", "label": "mesothelioma (broad)"},
        {"condition": "malignant pleural mesothelioma", "label": "malignant pleural mesothelioma"},
        {"condition": "pleural neoplasm", "label": "pleural neoplasm"},
        {"condition": "pleural effusion", "label": "pleural effusion"},
        {"condition": "asbestos", "label": "asbestos"},
        {"condition": "lung cancer", "label": "lung cancer (very broad)"},
        {"condition": "thoracic neoplasm", "label": "thoracic neoplasm"},
    ]

    cumulative_found = set()

    for cfg in search_configs:
        all_ncts = []
        page_token = None
        while True:
            result = await client.search(
                condition=cfg["condition"],
                status=ALL_STATUSES,
                page_size=100,
                page_token=page_token,
            )
            studies = result.get("studies", [])
            if not studies:
                break
            for s in studies:
                nct_id = s.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
                if nct_id:
                    all_ncts.append(nct_id)
            page_token = result.get("nextPageToken")
            if not page_token:
                break
            await asyncio.sleep(1.6)

        found = set(all_ncts) & eligible_set
        new = found - cumulative_found
        cumulative_found |= found
        print(f"  {cfg['label']}: {len(all_ncts)} trials, {len(found)} eligible found ({len(new)} new), cumulative: {len(cumulative_found)}")
        await asyncio.sleep(1.6)

    print(f"\nCumulative recall: {len(cumulative_found)}/{len(eligible_set)} ({len(cumulative_found)/len(eligible_set):.1%})")
    missed = eligible_set - cumulative_found
    print(f"Still missed: {len(missed)}")
    if missed:
        print(f"Sample missed: {sorted(missed)[:20]}")
    return cumulative_found, missed


async def main():
    from trialmatch.prescreen.ctgov_client import CTGovClient

    qrels = load_qrels()
    eligible_set = {nct for nct, s in qrels.items() if s == 2}
    eligible_sorted = sorted(eligible_set)
    print(f"Gold eligible trials: {len(eligible_set)}")
    print(f"NCT ID range: {eligible_sorted[0]} — {eligible_sorted[-1]}")

    client = CTGovClient(timeout_seconds=30.0)

    try:
        # Test 1: Direct lookup of sample
        found, not_found, statuses = await test_direct_lookup(client, eligible_sorted)

        # Test 2: Paginated search for ALL mesothelioma trials
        all_ncts, overlap, missed = await test_paginated_search(client, eligible_set)

        # Test 3: Alternative searches for missed trials
        if missed:
            cumulative, still_missed = await test_broader_searches(client, eligible_set)

            # If there are still many missed, do direct lookup on 10
            if still_missed and len(still_missed) > 5:
                print("\n" + "=" * 60)
                print("TEST 4: Direct lookup of missed trials (sample of 10)")
                print("=" * 60)
                missed_sample = sorted(still_missed)[:10]
                for nct_id in missed_sample:
                    try:
                        result = await client.get_details(nct_id)
                        proto = result.get("protocolSection", {})
                        status = proto.get("statusModule", {}).get("overallStatus", "UNKNOWN")
                        conds = proto.get("conditionsModule", {}).get("conditions", [])
                        title = proto.get("identificationModule", {}).get("briefTitle", "")
                        print(f"  {nct_id}: {status} | {conds[:3]} | {title[:80]}")
                    except Exception as e:
                        print(f"  {nct_id}: ERROR — {e}")
                    await asyncio.sleep(1.6)

        # Save diagnostic results
        out_path = REPO_ROOT / "runs" / "prescreen_diagnosis.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "eligible_count": len(eligible_set),
            "test1_direct_lookup": {
                "sample_size": 20,
                "found": len(found),
                "statuses": statuses,
            },
            "test2_paginated_search": {
                "total_retrieved": len(all_ncts),
                "eligible_found": len(overlap),
                "eligible_found_ncts": sorted(overlap),
                "recall": len(overlap) / len(eligible_set),
            },
        }
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nDiagnostic results saved to {out_path}")

    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
