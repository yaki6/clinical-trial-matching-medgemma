#!/usr/bin/env python3
"""Analyze findings benchmark results by case and region."""
import json

with open("runs/medpix-multimodal-findings_only-20260223-222130/scored_results.json") as f:
    data = json.load(f)

print(f"{'UID':<12} {'Region':<30} {'MG Judge':>10} {'GM Judge':>10} {'MG ROUGE-R':>11} {'GM ROUGE-R':>11}")
print("-" * 94)
for d in data:
    mg = d["medgemma"]
    gm = d["gemini"]
    region = d.get("location_category", "")
    print(f"{d['uid']:<12} {region:<30} {mg['findings_judge_score']:>10} {gm['findings_judge_score']:>10} {mg['rouge_recall']:>11.3f} {gm['rouge_recall']:>11.3f}")

print("\n--- By Region (LLM Judge) ---")
regions = sorted(set(d.get("location_category", "") for d in data))
for region in regions:
    rd = [d for d in data if d.get("location_category", "") == region]
    mg_good = sum(1 for d in rd if d["medgemma"]["findings_judge_score"] == "good")
    gm_good = sum(1 for d in rd if d["gemini"]["findings_judge_score"] == "good")
    mg_valid = sum(1 for d in rd if d["medgemma"]["findings_judge_score"] != "error")
    gm_valid = sum(1 for d in rd if d["gemini"]["findings_judge_score"] != "error")
    mg_r = sum(d["medgemma"]["rouge_recall"] for d in rd) / len(rd)
    gm_r = sum(d["gemini"]["rouge_recall"] for d in rd) / len(rd)
    print(f"  {region:<30}  MG: {mg_good}/{mg_valid} good  GM: {gm_good}/{gm_valid} good  |  ROUGE-R: MG={mg_r:.3f} GM={gm_r:.3f}")

# MedGemma strengths
print("\n--- MedGemma 1.5 4B Findings Where Judge Said 'good' ---")
for d in data:
    if d["medgemma"]["findings_judge_score"] == "good":
        print(f"  {d['uid']} ({d.get('location_category','')}) - {d['title'][:50]}")
        print(f"    Explanation: {d['medgemma']['findings_judge_explanation']}")
