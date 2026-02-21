"""Probe MedGemma multimodal v2 — test image-text-to-text pipeline formats.

The endpoint returned "You must provide text for this pipeline" for image_to_text,
meaning it DOES support image input but needs the right format.

Tests:
1. Direct HTTP POST with image + text (requests library)
2. visual_question_answering (image + question)
3. image_to_text with prompt in parameters
"""

import base64
import json
import os
import sys
import time
from pathlib import Path

import requests as req
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

ENDPOINT_URL = "https://pcmy7bkqtqesrrzd.us-east-1.aws.endpoints.huggingface.cloud"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
IMAGE_PATH = Path("ingest_design/MedPix-2-0/images/MPX1016_synpic34317.png")

PROMPT = "Describe the key clinical findings in this medical image."


def encode_image_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_http_post_multipart(b64: str) -> None:
    """Test 1: Direct HTTP POST with multipart image + text."""
    print("\n=== TEST 1: HTTP POST multipart (image file + text) ===")
    try:
        start = time.time()
        with open(IMAGE_PATH, "rb") as img_file:
            resp = req.post(
                ENDPOINT_URL,
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                files={"image": img_file},
                data={"text": PROMPT},
                timeout=60,
            )
        elapsed = time.time() - start
        print(f"  HTTP {resp.status_code} ({elapsed:.1f}s)")
        print(f"  RESPONSE: {resp.text[:500]}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")


def test_http_post_json_with_image(b64: str) -> None:
    """Test 2: HTTP POST JSON with base64 image in inputs."""
    print("\n=== TEST 2: HTTP POST JSON (inputs=text, image=b64) ===")
    try:
        start = time.time()
        payload = {
            "inputs": {
                "text": PROMPT,
                "image": b64,
            },
            "parameters": {"max_new_tokens": 256},
        }
        resp = req.post(
            ENDPOINT_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        elapsed = time.time() - start
        print(f"  HTTP {resp.status_code} ({elapsed:.1f}s)")
        print(f"  RESPONSE: {resp.text[:500]}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")


def test_http_post_json_gemma_template(b64: str) -> None:
    """Test 3: HTTP POST JSON with Gemma template + image."""
    print("\n=== TEST 3: HTTP POST JSON (Gemma template + image) ===")
    gemma_prompt = f"<start_of_turn>user\n{PROMPT}<end_of_turn>\n<start_of_turn>model\n"
    try:
        start = time.time()
        payload = {
            "inputs": {
                "text": gemma_prompt,
                "image": b64,
            },
            "parameters": {"max_new_tokens": 256},
        }
        resp = req.post(
            ENDPOINT_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        elapsed = time.time() - start
        print(f"  HTTP {resp.status_code} ({elapsed:.1f}s)")
        print(f"  RESPONSE: {resp.text[:500]}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")


def test_http_post_data_uri(b64: str) -> None:
    """Test 4: HTTP POST JSON with data URI image in inputs string."""
    print("\n=== TEST 4: HTTP POST JSON (inputs=string with image ref) ===")
    try:
        start = time.time()
        # Some endpoints accept image as a separate field alongside text inputs
        payload = {
            "inputs": PROMPT,
            "image": f"data:image/png;base64,{b64}",
            "parameters": {"max_new_tokens": 256},
        }
        resp = req.post(
            ENDPOINT_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        elapsed = time.time() - start
        print(f"  HTTP {resp.status_code} ({elapsed:.1f}s)")
        print(f"  RESPONSE: {resp.text[:500]}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")


def test_vqa(client: InferenceClient) -> None:
    """Test 5: visual_question_answering."""
    print("\n=== TEST 5: visual_question_answering ===")
    try:
        start = time.time()
        result = client.visual_question_answering(
            image=str(IMAGE_PATH),
            question=PROMPT,
        )
        elapsed = time.time() - start
        print(f"  STATUS: SUCCESS ({elapsed:.1f}s)")
        print(f"  RESPONSE: {result}")
    except Exception as e:
        elapsed = time.time() - start
        print(f"  STATUS: FAILED ({elapsed:.1f}s)")
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")


def test_endpoint_info(client: InferenceClient) -> None:
    """Test 6: Get endpoint info to see supported routes."""
    print("\n=== TEST 6: Endpoint info ===")
    try:
        info = client.get_endpoint_info()
        print(f"  INFO: {json.dumps(info, indent=2, default=str)[:800]}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")


def test_http_get_info() -> None:
    """Test 7: GET endpoint root for API info."""
    print("\n=== TEST 7: GET endpoint root ===")
    try:
        resp = req.get(
            ENDPOINT_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            timeout=15,
        )
        print(f"  HTTP {resp.status_code}")
        print(f"  RESPONSE: {resp.text[:500]}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {str(e)[:300]}")


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)
    if not IMAGE_PATH.exists():
        print(f"ERROR: Image not found: {IMAGE_PATH}")
        sys.exit(1)

    img_size = IMAGE_PATH.stat().st_size
    print(f"Image: {IMAGE_PATH} ({img_size / 1024:.0f} KB)")

    client = InferenceClient(model=ENDPOINT_URL, token=HF_TOKEN)
    b64 = encode_image_b64(IMAGE_PATH)
    print(f"Base64 size: {len(b64) / 1024:.0f} KB")

    # Get endpoint info first
    test_endpoint_info(client)
    test_http_get_info()

    # Test multimodal formats
    test_http_post_json_with_image(b64)
    test_http_post_json_gemma_template(b64)
    test_http_post_multipart(b64)
    test_http_post_data_uri(b64)
    test_vqa(client)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
