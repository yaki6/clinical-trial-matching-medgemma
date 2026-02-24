#!/usr/bin/env python3
"""Look up conditions for trials missed by mesothelioma/pleural neoplasm/pleural effusion searches.

From diagnostic run:
  mesothelioma: 23/118
  pleural neoplasm: +46 → 69/118
  pleural effusion: +27 → 96/118
  Still missing: 22 trials

This script identifies what conditions the 22 missed trials are registered under.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dotenv import load_dotenv
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "trec2022_ground_truth"

# Found by diagnostic: these 96 NCT IDs were found via mesothelioma + pleural neoplasm + pleural effusion
# We need to check the rest. Rather than hardcode, we'll do it fresh.
FOUND_VIA_CONDITION_SEARCH = set()  # Will be populated


def load_eligible_ncts() -> set[str]:
    qrels = {}
    for line in (DATA_DIR / "qrels.tsv").read_text().splitlines():
        if line.startswith("query-id") or not line.strip():
            continue
        parts = line.split("\t")
        qrels[parts[1]] = int(parts[2])
    return {nct for nct, s in qrels.items() if s == 2}


async def main():
    from trialmatch.prescreen.ctgov_client import CTGovClient

    ALL_STATUSES = [
        "RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING",
        "COMPLETED", "TERMINATED", "WITHDRAWN", "UNKNOWN",
    ]

    eligible = load_eligible_ncts()
    print(f"Total eligible: {len(eligible)}")

    client = CTGovClient(timeout_seconds=30.0)

    # Quick paginated search for our 3 known conditions to rebuild the found set
    found_set = set()
    for condition in ["mesothelioma", "pleural neoplasm", "pleural effusion"]:
        page_token = None
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
                nct = s.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
                if nct in eligible:
                    found_set.add(nct)
            page_token = result.get("nextPageToken")
            if not page_token:
                break
            await asyncio.sleep(1.6)
        print(f"  {condition}: cumulative found = {len(found_set)}")

    missed = eligible - found_set
    print(f"\nMissed trials: {len(missed)}")
    print(f"NCT IDs: {sorted(missed)}")

    # Look up each missed trial to see its conditions
    print(f"\n{'='*70}")
    print("Conditions of missed trials:")
    print(f"{'='*70}")

    condition_groups = {}
    for nct_id in sorted(missed):
        try:
            result = await client.get_details(nct_id)
            proto = result.get("protocolSection", {})
            conds = proto.get("conditionsModule", {}).get("conditions", [])
            status = proto.get("statusModule", {}).get("overallStatus", "?")
            title = proto.get("identificationModule", {}).get("briefTitle", "")
            keywords = proto.get("conditionsModule", {}).get("keywords", [])
            print(f"  {nct_id} ({status})")
            print(f"    Conditions: {conds}")
            if keywords:
                print(f"    Keywords: {keywords[:5]}")
            print(f"    Title: {title[:100]}")
            for c in conds:
                cl = c.lower()
                condition_groups.setdefault(cl, []).append(nct_id)
        except Exception as e:
            print(f"  {nct_id}: ERROR — {e}")
        await asyncio.sleep(1.6)

    # Summary
    print(f"\n{'='*70}")
    print("Condition frequency in missed trials:")
    print(f"{'='*70}")
    for cond, ncts in sorted(condition_groups.items(), key=lambda x: -len(x[1])):
        print(f"  {cond}: {len(ncts)} trials")

    # Save results
    out = {
        "eligible_total": len(eligible),
        "found_via_3_searches": len(found_set),
        "missed_count": len(missed),
        "missed_ncts": sorted(missed),
        "missed_conditions": {k: v for k, v in condition_groups.items()},
    }
    out_path = REPO_ROOT / "runs" / "prescreen_missed_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {out_path}")

    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
