#!/usr/bin/env python3
"""Run LIVE MedGemma imaging extraction for curated demo patients.

Routes imaging to Vertex 4B when configured, otherwise HF MedGemma fallback.
Writes artifacts to data/sot/ingest/medgemma_image_cache.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from trialmatch.live_runtime import create_imaging_adapter  # noqa: E402

HARNESS_PATH = REPO_ROOT / "data" / "sot" / "ingest" / "nsclc_demo_harness.json"
CACHE_DIR = REPO_ROOT / "data" / "sot" / "ingest" / "medgemma_image_cache"
MULTIMODAL_TOPIC_IDS = ["mpx1016", "mpx1575", "mpx1875"]

PATIENT_PROMPTS = {
    "mpx1016": (
        "What abnormalities do you see in this chest CT scan? "
        "Focus on lung parenchyma, pleural space, and any masses or nodules. "
        "Describe location, size, and characteristics."
    ),
    "mpx1575": (
        "Analyze this CT image. Identify any pulmonary masses, fibrotic changes, "
        "honeycombing, pleural effusions, or lymphadenopathy. Describe findings systematically."
    ),
    "mpx1875": (
        "Is there a lung mass visible in this image? Describe its location, size, "
        "margins, and density. Note any mediastinal or hilar abnormalities."
    ),
}
FALLBACK_PROMPT = (
    "Describe the key abnormalities visible in this medical image. "
    "Focus on lung findings, masses, and any pathology."
)


def load_harness() -> list[dict]:
    with open(HARNESS_PATH) as f:
        data = json.load(f)
    return data["patients"]


def save_cache(topic_id: str, payload: dict) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"{topic_id}.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


async def main_async() -> None:
    load_dotenv(REPO_ROOT / ".env")
    hf_token = os.environ.get("HF_TOKEN", "")
    adapter = create_imaging_adapter(hf_token=hf_token)

    patients = {
        p["topic_id"]: p
        for p in load_harness()
        if p["topic_id"] in MULTIMODAL_TOPIC_IDS
    }

    print("=" * 70)
    print("MedGemma LIVE multimodal extraction")
    print(f"Adapter: {adapter.name}")
    print(f"Patients: {', '.join(MULTIMODAL_TOPIC_IDS)}")
    print("=" * 70)

    ok = 0
    for idx, topic_id in enumerate(MULTIMODAL_TOPIC_IDS, start=1):
        patient = patients.get(topic_id)
        if not patient:
            raise RuntimeError(f"Missing patient in harness: {topic_id}")
        image_rel = patient["image"]["file_path"]
        image_path = REPO_ROOT / image_rel
        if not image_path.exists():
            raise RuntimeError(f"Image not found for {topic_id}: {image_path}")

        prompt = PATIENT_PROMPTS.get(topic_id, FALLBACK_PROMPT)
        print(f"[{idx}/{len(MULTIMODAL_TOPIC_IDS)}] {topic_id}: {image_rel}")
        response = await adapter.generate_with_image(prompt, image_path, max_tokens=512)
        payload = {
            "topic_id": topic_id,
            "image_path": image_rel,
            "prompt": prompt,
            "raw_response": response.text,
            "extracted_text": response.text,
            "latency_seconds": round(response.latency_ms / 1000, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": adapter.name,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "estimated_cost": response.estimated_cost,
        }
        cache_path = save_cache(topic_id, payload)
        print(f"  saved: {cache_path.relative_to(REPO_ROOT)}")
        ok += 1

    print(f"done: {ok}/{len(MULTIMODAL_TOPIC_IDS)} patients")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
