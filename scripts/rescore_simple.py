#!/usr/bin/env python3
"""Simple rescore: ROUGE locally + LLM judge with timeout."""
import asyncio, json, os, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trialmatch.evaluation.multimodal_metrics import parse_model_response, score_findings_rouge
from trialmatch.models.gemini import GeminiAdapter

FINDINGS_JUDGE = """You are a radiologist evaluating imaging findings.

GOLD STANDARD: {gold}
MODEL FINDINGS: {pred}

Score: "good" (main abnormality correct), "partial" (some relevant findings but missed primary), "poor" (missed primary or called normal when abnormal).
Respond ONLY JSON: {{"score": "good", "key_finding_identified": true, "explanation": "reason"}}"""

async def judge_one(judge, gold, pred, timeout=30):
    prompt = FINDINGS_JUDGE.format(gold=gold, pred=pred[:2000])
    try:
        resp = await asyncio.wait_for(judge.generate(prompt=prompt, max_tokens=200), timeout=timeout)
        return json.loads(resp.text)
    except asyncio.TimeoutError:
        return {"score": "error", "key_finding_identified": False, "explanation": "timeout"}
    except Exception as e:
        return {"score": "error", "key_finding_identified": False, "explanation": str(e)[:100]}

async def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/medpix-multimodal-findings_only-20260223-222130")
    with open(run_dir / "raw_responses.json") as f:
        raw = json.load(f)
    print(f"Loaded {len(raw)} cases", flush=True)

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    judge = GeminiAdapter(api_key=api_key, model="gemini-3-pro-preview")

    scored = []
    for i, d in enumerate(raw):
        uid = d["uid"]
        mg_p = parse_model_response(d["medgemma_raw"])
        gm_p = parse_model_response(d["gemini_raw"])
        if not mg_p["findings"].strip(): mg_p["findings"] = d["medgemma_raw"]
        if not gm_p["findings"].strip(): gm_p["findings"] = d["gemini_raw"]

        mg_rouge = score_findings_rouge(d["gold_findings"], mg_p["findings"])
        gm_rouge = score_findings_rouge(d["gold_findings"], gm_p["findings"])

        print(f"  [{i+1}/{len(raw)}] {uid}: judging MedGemma...", end="", flush=True)
        mg_j = await judge_one(judge, d["gold_findings"], mg_p["findings"])
        print(f" {mg_j['score']}  |  judging Gemini...", end="", flush=True)
        gm_j = await judge_one(judge, d["gold_findings"], gm_p["findings"])
        print(f" {gm_j['score']}", flush=True)

        scored.append({
            "uid": uid, "title": d.get("title",""), "location_category": d.get("location_category",""),
            "gold_diagnosis": d["gold_diagnosis"], "gold_findings": d["gold_findings"],
            "medgemma": {
                "predicted_findings": mg_p["findings"][:1000],
                "findings_judge_score": mg_j.get("score","error"),
                "findings_judge_key_finding": mg_j.get("key_finding_identified", False),
                "findings_judge_explanation": mg_j.get("explanation",""),
                "rouge_recall": mg_rouge["recall"], "rouge_precision": mg_rouge["precision"], "rouge_fmeasure": mg_rouge["fmeasure"],
                "latency_ms": d.get("medgemma_latency_ms",0), "cost": d.get("medgemma_cost",0),
            },
            "gemini": {
                "predicted_findings": gm_p["findings"][:1000],
                "findings_judge_score": gm_j.get("score","error"),
                "findings_judge_key_finding": gm_j.get("key_finding_identified", False),
                "findings_judge_explanation": gm_j.get("explanation",""),
                "rouge_recall": gm_rouge["recall"], "rouge_precision": gm_rouge["precision"], "rouge_fmeasure": gm_rouge["fmeasure"],
                "latency_ms": d.get("gemini_latency_ms",0), "cost": d.get("gemini_cost",0),
            },
        })

    with open(run_dir / "scored_results.json", "w") as f:
        json.dump(scored, f, indent=2, default=str)

    # Summary
    n = len(scored)
    def agg(key):
        r = [s[key] for s in scored]
        valid = [x for x in r if x["findings_judge_score"] != "error"]
        vn = len(valid) if valid else 1
        return {
            "n_cases": n, "n_judged": len(valid),
            "findings_judge_good": sum(1 for x in valid if x["findings_judge_score"]=="good") / vn,
            "findings_judge_partial": sum(1 for x in valid if x["findings_judge_score"]=="partial") / vn,
            "findings_judge_poor": sum(1 for x in valid if x["findings_judge_score"]=="poor") / vn,
            "findings_key_finding_rate": sum(1 for x in valid if x.get("findings_judge_key_finding")) / vn,
            "rouge_recall": sum(x["rouge_recall"] for x in r) / n,
            "rouge_precision": sum(x["rouge_precision"] for x in r) / n,
            "rouge_f1": sum(x["rouge_fmeasure"] for x in r) / n,
            "avg_latency_ms": sum(x["latency_ms"] for x in r) / n,
            "total_cost": sum(x["cost"] for x in r),
        }
    summary = {"mode": "findings_only", "medgemma_4b": agg("medgemma"), "gemini_pro": agg("gemini")}
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    mg, gm = summary["medgemma_4b"], summary["gemini_pro"]
    print(f"\n{'='*75}", flush=True)
    print(f"  20-Case Multi-Region Findings Benchmark Results", flush=True)
    print(f"{'='*75}", flush=True)
    print(f"{'Metric':<40} {'MedGemma 1.5 4B':>15} {'Gemini Flash':>15}", flush=True)
    print(f"{'-'*75}", flush=True)
    for label, mk, gk in [
        ("LLM Judge (good)", "findings_judge_good", "findings_judge_good"),
        ("LLM Judge (partial)", "findings_judge_partial", "findings_judge_partial"),
        ("LLM Judge (poor)", "findings_judge_poor", "findings_judge_poor"),
        ("Key Finding Rate", "findings_key_finding_rate", "findings_key_finding_rate"),
        ("ROUGE-L Recall", "rouge_recall", "rouge_recall"),
        ("ROUGE-L Precision", "rouge_precision", "rouge_precision"),
        ("ROUGE-L F1", "rouge_f1", "rouge_f1"),
        ("Avg Latency (ms)", "avg_latency_ms", "avg_latency_ms"),
    ]:
        mv = mg[mk]; gv = gm[gk]
        if "latency" in mk:
            print(f"{label:<40} {mv:>15.0f} {gv:>15.0f}", flush=True)
        else:
            print(f"{label:<40} {mv:>15.1%} {gv:>15.1%}", flush=True)
    print(f"{'='*75}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
