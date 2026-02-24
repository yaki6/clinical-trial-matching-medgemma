#!/usr/bin/env python3
"""Analyze CT.gov condition terms for TREC 2022 eligible trials.

Fetches condition metadata for all 118 eligible trials and determines
which search terms the PRESCREEN agent needs to achieve high recall.

Usage:
    uv run python scripts/analyze_trec_search_coverage.py
"""

import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "trec2022_ground_truth"


def load_qrels() -> dict[str, int]:
    qrels = {}
    for line in (DATA_DIR / "qrels.tsv").read_text().splitlines():
        if line.startswith("query-id") or not line.strip():
            continue
        parts = line.split("\t")
        qrels[parts[1]] = int(parts[2])
    return qrels


async def fetch_all_eligible_trials(eligible_ncts: list[str]) -> dict:
    """Fetch trial metadata from CT.gov for all eligible trials."""
    from trialmatch.prescreen.ctgov_client import CTGovClient

    client = CTGovClient()
    results = {}
    errors = []

    for i, nct_id in enumerate(eligible_ncts):
        try:
            raw = await client.get_details(nct_id)
            if raw:
                # Parse raw CT.gov API v2 response structure
                proto = raw.get("protocolSection", {})
                id_mod = proto.get("identificationModule", {})
                status_mod = proto.get("statusModule", {})
                cond_mod = proto.get("conditionsModule", {})
                design_mod = proto.get("designModule", {})
                arms_mod = proto.get("armsInterventionsModule", {})

                results[nct_id] = {
                    "conditions": cond_mod.get("conditions", []),
                    "keywords": cond_mod.get("keywords", []),
                    "title": id_mod.get("briefTitle", id_mod.get("officialTitle", "")),
                    "status": status_mod.get("overallStatus", ""),
                    "phase": design_mod.get("phases", []),
                    "study_type": design_mod.get("studyType", ""),
                    "interventions": [
                        iv.get("name", "") for iv in arms_mod.get("interventions", [])
                    ] if arms_mod.get("interventions") else [],
                }
            else:
                errors.append(nct_id)
            if (i + 1) % 20 == 0:
                print(f"  Fetched {i+1}/{len(eligible_ncts)}...")
        except Exception as e:
            errors.append(nct_id)
            print(f"  ERROR fetching {nct_id}: {e}")

        # Rate limiting
        await asyncio.sleep(1.6)

    await client.aclose()
    return results, errors


def analyze_conditions(trials: dict) -> dict:
    """Analyze condition term distribution."""
    # Collect all condition terms
    all_conditions = Counter()
    condition_to_trials = defaultdict(set)

    for nct_id, data in trials.items():
        for cond in data["conditions"]:
            cond_lower = cond.lower().strip()
            all_conditions[cond_lower] += 1
            condition_to_trials[cond_lower].add(nct_id)

    return all_conditions, condition_to_trials


def compute_search_recall(condition_to_trials: dict, eligible_set: set, search_terms: list[str]) -> dict:
    """Compute what recall a given set of search terms would achieve."""
    found = set()
    term_contributions = {}

    for term in search_terms:
        term_lower = term.lower()
        term_matches = set()
        for cond, ncts in condition_to_trials.items():
            if term_lower in cond:
                term_matches |= ncts
        new_found = term_matches - found
        term_contributions[term] = {
            "total_matches": len(term_matches),
            "new_unique": len(new_found),
            "ncts": sorted(term_matches & eligible_set),
        }
        found |= term_matches

    return {
        "total_found": len(found),
        "recall": len(found) / len(eligible_set) if eligible_set else 0,
        "term_contributions": term_contributions,
        "missed": sorted(eligible_set - found),
    }


async def main():
    qrels = load_qrels()
    eligible_ncts = sorted(nct for nct, score in qrels.items() if score == 2)
    print(f"Total eligible trials: {len(eligible_ncts)}")
    print(f"First 5: {eligible_ncts[:5]}")

    # Check cache
    cache_path = DATA_DIR / "eligible_trial_metadata.json"
    if cache_path.exists():
        print(f"\nLoading cached metadata from {cache_path}")
        cached = json.loads(cache_path.read_text())
        trials = cached["trials"]
        errors = cached.get("errors", [])
        # Check if cache has empty conditions (bad parse from previous run)
        bad_parse = [nct for nct, data in trials.items() if not data.get("conditions")]
        missing = [nct for nct in eligible_ncts if nct not in trials]
        to_refetch = list(set(bad_parse + missing))
        if to_refetch:
            print(f"  Refetching {len(to_refetch)} trials ({len(bad_parse)} bad parse, {len(missing)} missing)...")
            new_trials, new_errors = await fetch_all_eligible_trials(to_refetch)
            trials.update(new_trials)
            errors = [e for e in errors if e not in new_trials]
            errors.extend(new_errors)
            cache_path.write_text(json.dumps({"trials": trials, "errors": errors}, indent=2))
    else:
        print(f"\nFetching {len(eligible_ncts)} trials from CT.gov (this takes ~3 minutes)...")
        trials, errors = await fetch_all_eligible_trials(eligible_ncts)
        cache_path.write_text(json.dumps({"trials": trials, "errors": errors}, indent=2))

    print(f"\nFetched: {len(trials)}, Errors: {len(errors)}")
    if errors:
        print(f"  Error NCTs: {errors[:10]}{'...' if len(errors) > 10 else ''}")

    # Analyze condition terms
    all_conditions, condition_to_trials = analyze_conditions(trials)
    eligible_set = set(eligible_ncts) & set(trials.keys())

    print(f"\n{'='*60}")
    print("CONDITION TERM DISTRIBUTION (Top 30)")
    print(f"{'='*60}")
    for cond, count in all_conditions.most_common(30):
        ncts_in_eligible = len(condition_to_trials[cond] & eligible_set)
        print(f"  {count:3d} trials | {cond}")

    # Categorize trials by condition type
    print(f"\n{'='*60}")
    print("SEARCH STRATEGY ANALYSIS")
    print(f"{'='*60}")

    # Strategy 1: Current (just "malignant pleural mesothelioma")
    strategies = {
        "Current: 'malignant pleural mesothelioma'": ["malignant pleural mesothelioma"],
        "Add: 'mesothelioma' (broad)": ["malignant pleural mesothelioma", "mesothelioma"],
        "Add: 'pleural effusion'": ["malignant pleural mesothelioma", "mesothelioma", "pleural effusion"],
        "Add: 'metastatic cancer'": ["malignant pleural mesothelioma", "mesothelioma", "pleural effusion", "metastatic cancer"],
        "Add: 'lung cancer'": ["malignant pleural mesothelioma", "mesothelioma", "pleural effusion", "metastatic cancer", "lung cancer"],
        "Add: 'solid tumor' + 'neoplasm'": ["malignant pleural mesothelioma", "mesothelioma", "pleural effusion", "metastatic cancer", "lung cancer", "solid tumor", "neoplasm"],
        "Full broad: + 'cancer' + 'carcinoma'": ["malignant pleural mesothelioma", "mesothelioma", "pleural effusion", "metastatic cancer", "lung cancer", "solid tumor", "neoplasm", "cancer", "carcinoma"],
    }

    for label, terms in strategies.items():
        result = compute_search_recall(condition_to_trials, eligible_set, terms)
        print(f"\n  {label}")
        print(f"    Recall: {result['recall']:.1%} ({result['total_found']}/{len(eligible_set)})")
        print(f"    Missed: {len(result['missed'])}")
        for term, contrib in result["term_contributions"].items():
            if contrib["new_unique"] > 0:
                print(f"      +'{term}' adds {contrib['new_unique']} trials (total matches: {contrib['total_matches']})")

    # Show trials not matchable by any mesothelioma/pleural/cancer term
    full_result = compute_search_recall(
        condition_to_trials, eligible_set,
        ["mesothelioma", "pleural", "lung", "cancer", "tumor", "neoplasm", "carcinoma", "malignant", "metastatic", "solid"]
    )
    if full_result["missed"]:
        print(f"\n{'='*60}")
        print(f"UNREACHABLE TRIALS (not matched by any broad term): {len(full_result['missed'])}")
        print(f"{'='*60}")
        for nct in full_result["missed"][:20]:
            if nct in trials:
                t = trials[nct]
                print(f"  {nct}: conditions={t['conditions']}, title={t['title'][:80]}")

    # Analyze trials by category
    print(f"\n{'='*60}")
    print("TRIAL CATEGORIZATION")
    print(f"{'='*60}")

    categories = {
        "mesothelioma_specific": [],
        "pleural_effusion": [],
        "lung_cancer_broad": [],
        "metastatic_general": [],
        "other_cancer": [],
        "observational_diagnostic": [],
        "unknown": [],
    }

    for nct_id in sorted(eligible_set):
        data = trials[nct_id]
        conds_lower = [c.lower() for c in data["conditions"]]
        conds_str = " ".join(conds_lower)

        if any("mesothelioma" in c for c in conds_lower):
            categories["mesothelioma_specific"].append(nct_id)
        elif any("pleural" in c for c in conds_lower):
            categories["pleural_effusion"].append(nct_id)
        elif any("lung" in c for c in conds_lower):
            categories["lung_cancer_broad"].append(nct_id)
        elif any("metastatic" in c for c in conds_lower):
            categories["metastatic_general"].append(nct_id)
        elif any(term in conds_str for term in ["cancer", "carcinoma", "neoplasm", "tumor", "malignant"]):
            categories["other_cancer"].append(nct_id)
        elif data["study_type"] == "OBSERVATIONAL":
            categories["observational_diagnostic"].append(nct_id)
        else:
            categories["unknown"].append(nct_id)

    for cat, ncts in categories.items():
        print(f"\n  {cat}: {len(ncts)} trials")
        if ncts and len(ncts) <= 10:
            for nct in ncts:
                t = trials[nct]
                print(f"    {nct}: {t['conditions']}")
        elif ncts:
            for nct in ncts[:5]:
                t = trials[nct]
                print(f"    {nct}: {t['conditions']}")
            print(f"    ... and {len(ncts)-5} more")

    # Save full analysis
    analysis = {
        "total_eligible": len(eligible_ncts),
        "fetched": len(trials),
        "errors": errors,
        "condition_distribution": dict(all_conditions.most_common(50)),
        "categories": {k: v for k, v in categories.items()},
        "strategy_recalls": {},
    }
    for label, terms in strategies.items():
        result = compute_search_recall(condition_to_trials, eligible_set, terms)
        analysis["strategy_recalls"][label] = {
            "recall": result["recall"],
            "found": result["total_found"],
            "missed_count": len(result["missed"]),
            "missed_first_10": result["missed"][:10],
        }

    out_path = DATA_DIR / "search_coverage_analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    print(f"\n\nAnalysis saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
