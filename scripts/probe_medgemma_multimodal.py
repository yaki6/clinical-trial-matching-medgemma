"""Probe MedGemma endpoint for multimodal (image) support.

Tests three API paths:
1. chat_completion with image (OpenAI-compatible Messages API)
2. text_generation with text-only (current working path)
3. image_to_text (HF pipeline)

Usage: uv run python scripts/probe_medgemma_multimodal.py
"""

import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

ENDPOINT_URL = "https://pcmy7bkqtqesrrzd.us-east-1.aws.endpoints.huggingface.cloud"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
IMAGE_PATH = Path("ingest_design/MedPix-2-0/images/MPX1016_synpic34317.png")

PROMPT = (
    "You are a radiologist. Describe the key clinical findings in this chest CT image. "
    "Output JSON: {\"findings\": [\"...\"], \"impression\": \"...\"}"
)


def encode_image_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_chat_completion_with_image(client: InferenceClient, b64: str) -> None:
    """Test 1: chat_completion with image_url content."""
    print("\n=== TEST 1: chat_completion + image ===")
    try:
        start = time.time()
        response = client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            max_tokens=512,
        )
        elapsed = time.time() - start
        text = response.choices[0].message.content
        print(f"  STATUS: SUCCESS ({elapsed:.1f}s)")
        print(f"  RESPONSE ({len(text)} chars): {text[:500]}")
        if hasattr(response, "usage") and response.usage:
            print(f"  USAGE: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"  STATUS: FAILED ({elapsed:.1f}s)")
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")
        return False


def test_chat_completion_text_only(client: InferenceClient) -> None:
    """Test 2: chat_completion text-only (baseline)."""
    print("\n=== TEST 2: chat_completion text-only ===")
    try:
        start = time.time()
        response = client.chat_completion(
            messages=[{"role": "user", "content": "Say OK if ready."}],
            max_tokens=10,
        )
        elapsed = time.time() - start
        text = response.choices[0].message.content
        print(f"  STATUS: SUCCESS ({elapsed:.1f}s)")
        print(f"  RESPONSE: {text[:200]}")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"  STATUS: FAILED ({elapsed:.1f}s)")
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")
        return False


def test_text_generation(client: InferenceClient) -> None:
    """Test 3: text_generation (current working path)."""
    print("\n=== TEST 3: text_generation text-only ===")
    try:
        start = time.time()
        prompt = "<start_of_turn>user\nSay OK if ready.<end_of_turn>\n<start_of_turn>model\n"
        result = client.text_generation(prompt, max_new_tokens=10)
        elapsed = time.time() - start
        print(f"  STATUS: SUCCESS ({elapsed:.1f}s)")
        print(f"  RESPONSE: {result[:200]}")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"  STATUS: FAILED ({elapsed:.1f}s)")
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")
        return False


def test_image_to_text(client: InferenceClient) -> None:
    """Test 4: image_to_text pipeline."""
    print("\n=== TEST 4: image_to_text pipeline ===")
    try:
        start = time.time()
        result = client.image_to_text(str(IMAGE_PATH))
        elapsed = time.time() - start
        print(f"  STATUS: SUCCESS ({elapsed:.1f}s)")
        print(f"  RESPONSE: {result}")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"  STATUS: FAILED ({elapsed:.1f}s)")
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")
        return False


def test_raw_post_with_image(client: InferenceClient, b64: str) -> None:
    """Test 5: raw POST to endpoint root with image payload."""
    print("\n=== TEST 5: raw POST with image ===")
    try:
        start = time.time()
        payload = {
            "inputs": f"<start_of_turn>user\n{PROMPT}<end_of_turn>\n<start_of_turn>model\n",
            "parameters": {"max_new_tokens": 256},
            "image": b64,
        }
        result = client.post(json=payload)
        elapsed = time.time() - start
        print(f"  STATUS: SUCCESS ({elapsed:.1f}s)")
        decoded = json.loads(result) if isinstance(result, (bytes, str)) else result
        print(f"  RESPONSE: {str(decoded)[:500]}")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"  STATUS: FAILED ({elapsed:.1f}s)")
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")
        return False


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)
    if not IMAGE_PATH.exists():
        print(f"ERROR: Image not found: {IMAGE_PATH}")
        sys.exit(1)

    img_size = IMAGE_PATH.stat().st_size
    print(f"Image: {IMAGE_PATH} ({img_size / 1024:.0f} KB)")
    print(f"Endpoint: {ENDPOINT_URL}")

    client = InferenceClient(model=ENDPOINT_URL, token=HF_TOKEN)
    b64 = encode_image_b64(IMAGE_PATH)
    print(f"Base64 size: {len(b64) / 1024:.0f} KB")

    # Run tests sequentially — avoid overloading endpoint
    results = {}
    results["text_generation"] = test_text_generation(client)
    results["chat_completion_text"] = test_chat_completion_text_only(client)
    results["chat_completion_image"] = test_chat_completion_with_image(client, b64)
    results["image_to_text"] = test_image_to_text(client)
    results["raw_post_image"] = test_raw_post_with_image(client, b64)

    print("\n=== SUMMARY ===")
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test}: {status}")


if __name__ == "__main__":
    main()
