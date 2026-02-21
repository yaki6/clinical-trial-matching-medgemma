#!/usr/bin/env python3
"""
extract_image_findings.py -- LIVE MedGemma 4B multimodal image extraction.

Sends chest CT images for 3 NSCLC patients to the MedGemma 4B endpoint,
extracts structured radiological findings, and caches the results.

Usage:
    uv run python scripts/extract_image_findings.py
"""

import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_PATH = REPO_ROOT / "data" / "sot" / "ingest" / "nsclc_demo_harness.json"
CACHE_DIR = REPO_ROOT / "data" / "sot" / "ingest" / "medgemma_image_cache"

ENDPOINT_URL = "https://pcmy7bkqtqesrrzd.us-east-1.aws.endpoints.huggingface.cloud"
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set. Make sure .env contains HF_TOKEN.", file=sys.stderr)
    sys.exit(1)

MODEL_NAME = "medgemma-1.5-4b-it-hae"

# Only process these multimodal patients
MULTIMODAL_TOPIC_IDS = ["mpx1016", "mpx1575", "mpx1875"]

# Patient-specific prompts yield differentiated findings (generic prompts cause template output)
PATIENT_PROMPTS = {
    "mpx1016": "What abnormalities do you see in this chest CT scan? Focus on lung parenchyma, pleural space, and any masses or nodules. Describe location, size, and characteristics.",
    "mpx1575": "Analyze this CT image. Identify any pulmonary masses, fibrotic changes, honeycombing, pleural effusions, or lymphadenopathy. Describe findings systematically.",
    "mpx1875": "Is there a lung mass visible in this image? Describe its location, size, margins, and density. Note any mediastinal or hilar abnormalities.",
}
FALLBACK_PROMPT = "Describe the key abnormalities visible in this medical image. Focus on lung findings, masses, and any pathology."

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 10
REQUEST_TIMEOUT = 120
MAX_NEW_TOKENS = 512

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_harness() -> list[dict]:
    """Load harness JSON and return list of patient dicts."""
    with open(HARNESS_PATH, "r") as f:
        data = json.load(f)
    return data["patients"]


def encode_image(image_path: Path) -> str:
    """Read an image file and return its base64 encoding."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_medgemma(b64_image: str, prompt: str) -> tuple[str, str, float]:
    """
    Call MedGemma 4B multimodal endpoint with image + prompt.

    Returns (raw_generated_text, extracted_text, latency_seconds).
    Retries up to MAX_RETRIES on 503 errors.
    """
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": {
            "text": prompt,
            "image": b64_image,
        },
        "parameters": {"max_new_tokens": MAX_NEW_TOKENS},
    }

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            resp = requests.post(
                ENDPOINT_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            latency = time.time() - t0

            if resp.status_code == 503:
                print(f"  [attempt {attempt}/{MAX_RETRIES}] 503 Service Unavailable -- retrying in {RETRY_BACKOFF_SECONDS}s ...")
                last_error = f"503: {resp.text[:200]}"
                time.sleep(RETRY_BACKOFF_SECONDS)
                continue

            resp.raise_for_status()

            data = resp.json()
            # Expected format: [{"input_text": "...", "generated_text": "..."}]
            input_text = data[0].get("input_text", "")
            generated_text = data[0].get("generated_text", "")

            # Strip prompt echo
            extracted = generated_text.replace(input_text, "", 1).strip()
            return generated_text, extracted, latency

        except requests.exceptions.Timeout:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] Request timed out after {REQUEST_TIMEOUT}s -- retrying in {RETRY_BACKOFF_SECONDS}s ...")
            last_error = "timeout"
            time.sleep(RETRY_BACKOFF_SECONDS)
        except Exception as e:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] Error: {e}")
            last_error = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}")


def save_cache(result: dict, topic_id: str) -> Path:
    """Save result to cache directory and return the path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{topic_id}.json"
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    return cache_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MedGemma 4B Multimodal Image Extraction -- LIVE")
    print(f"Endpoint: {ENDPOINT_URL}")
    print(f"Patients: {', '.join(MULTIMODAL_TOPIC_IDS)}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)
    print()

    patients = load_harness()

    # Filter to multimodal patients only
    multimodal_patients = [
        p for p in patients
        if p["topic_id"] in MULTIMODAL_TOPIC_IDS
    ]

    if len(multimodal_patients) != len(MULTIMODAL_TOPIC_IDS):
        found = {p["topic_id"] for p in multimodal_patients}
        missing = set(MULTIMODAL_TOPIC_IDS) - found
        print(f"WARNING: Missing patients in harness: {missing}", file=sys.stderr)

    results = []
    for i, patient in enumerate(multimodal_patients, 1):
        topic_id = patient["topic_id"]
        image_rel = patient["image"]["file_path"]
        image_path = REPO_ROOT / image_rel

        print(f"[{i}/{len(multimodal_patients)}] Processing {topic_id}")
        print(f"  Image: {image_rel}")
        print(f"  Diagnosis: {patient.get('diagnosis', 'N/A')}")

        if not image_path.exists():
            print(f"  ERROR: Image not found at {image_path}", file=sys.stderr)
            continue

        # Encode image
        print(f"  Encoding image ({image_path.stat().st_size / 1024:.1f} KB) ...")
        b64_image = encode_image(image_path)

        # Call MedGemma with patient-specific prompt
        prompt = PATIENT_PROMPTS.get(topic_id, FALLBACK_PROMPT)
        print(f"  Calling MedGemma 4B (max_new_tokens={MAX_NEW_TOKENS}) ...")
        print(f"  Prompt: {prompt[:80]}...")
        try:
            raw_response, extracted_text, latency = call_medgemma(b64_image, prompt)
        except RuntimeError as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            continue

        print(f"  Response received in {latency:.1f}s")

        # Build result
        result = {
            "topic_id": topic_id,
            "image_path": image_rel,
            "prompt": prompt,
            "raw_response": raw_response,
            "extracted_text": extracted_text,
            "latency_seconds": round(latency, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_NAME,
        }

        # Save to cache
        cache_path = save_cache(result, topic_id)
        print(f"  Cached: {cache_path.relative_to(REPO_ROOT)}")

        # Print findings
        print()
        print(f"  --- Clinical Findings for {topic_id} ---")
        print(f"  {extracted_text}")
        print(f"  --- End Findings ---")
        print()

        results.append(result)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"  {r['topic_id']:12s}  latency={r['latency_seconds']:5.1f}s  chars={len(r['extracted_text']):4d}")
    print()
    print(f"Total patients processed: {len(results)}/{len(multimodal_patients)}")
    print(f"Cache directory: {CACHE_DIR.relative_to(REPO_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
