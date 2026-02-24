#!/usr/bin/env python3
"""Diagnostic: Is MedGemma v1.5 actually processing images?

Test 1: Health check (text only)
Test 2: Same prompt WITH vs WITHOUT image
Test 3: Simple image description
Test 4: Check base64 encoding sizes
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trialmatch.models.vertex_medgemma import VertexMedGemmaAdapter


async def main():
    project_id = os.environ.get("GCP_PROJECT_ID", "gen-lang-client-0517724223")
    endpoint_id = os.environ.get("VERTEX_ENDPOINT_ID_4B", "923518299076034560")

    dedicated_dns = os.environ.get("VERTEX_DEDICATED_DNS_4B", "")
    if not dedicated_dns:
        import subprocess
        try:
            result = subprocess.run(
                ["gcloud", "ai", "endpoints", "describe", endpoint_id,
                 "--region=us-central1", "--format=value(dedicatedEndpointDns)"],
                capture_output=True, text=True, timeout=30,
            )
            dedicated_dns = result.stdout.strip()
            if dedicated_dns:
                print(f"Discovered dedicated DNS: {dedicated_dns}")
        except Exception:
            pass

    adapter = VertexMedGemmaAdapter(
        project_id=project_id,
        region="us-central1",
        endpoint_id=endpoint_id,
        model_name="medgemma-1.5-4b-diag",
        dedicated_endpoint_dns=dedicated_dns or None,
    )

    print(f"Predict URL: {adapter._predict_url}")

    # --- Test 0: Health check with retries ---
    print("\n" + "=" * 60)
    print("TEST 0: Health check (text-only, with retries)")
    print("=" * 60)

    for attempt in range(3):
        try:
            resp = await adapter.generate("Say hello", max_tokens=10)
            print(f"  Attempt {attempt+1}: OK — '{resp.text.strip()[:80]}'")
            break
        except Exception as e:
            print(f"  Attempt {attempt+1}: FAILED — {e}")
            if attempt < 2:
                print(f"  Waiting 10s before retry...")
                await asyncio.sleep(10)
    else:
        print("FATAL: Endpoint not responding after 3 attempts. Aborting.")
        sys.exit(1)

    test_image = Path("ingest_design/MedPix-2-0/images/MPX1452_synpic58265.png")
    rgba_image = Path("ingest_design/MedPix-2-0/images/MPX1063_synpic22165.png")

    from PIL import Image
    img = Image.open(test_image)
    print(f"\nTest image: {test_image}")
    print(f"  Mode: {img.mode}, Size: {img.size}, Bytes: {test_image.stat().st_size}")
    img.close()

    # --- Test 1: WITH vs WITHOUT image ---
    print("\n" + "=" * 60)
    print("TEST 1: WITH image vs WITHOUT image")
    print("=" * 60)

    prompt = "What type of medical image is this? Describe what you see in 2-3 sentences."

    for attempt in range(3):
        try:
            print(f"\n--- 1a: WITH image (attempt {attempt+1}) ---")
            resp_with = await adapter.generate_with_image(
                prompt=prompt, image_path=test_image, max_tokens=200,
            )
            print(f"  Latency: {resp_with.latency_ms:.0f}ms, Tokens: in={resp_with.input_tokens} out={resp_with.output_tokens}")
            print(f"  Response: {resp_with.text.strip()[:400]}")
            break
        except Exception as e:
            print(f"  FAILED: {e}")
            if attempt < 2:
                await asyncio.sleep(10)
    else:
        print("  Could not get image response after 3 attempts")
        resp_with = None

    print(f"\n--- 1b: WITHOUT image (text only) ---")
    try:
        resp_without = await adapter.generate(prompt=prompt, max_tokens=200)
        print(f"  Latency: {resp_without.latency_ms:.0f}ms")
        print(f"  Response: {resp_without.text.strip()[:400]}")
    except Exception as e:
        print(f"  FAILED: {e}")
        resp_without = None

    if resp_with and resp_without:
        same = resp_with.text.strip() == resp_without.text.strip()
        print(f"\n>>> Responses identical? {same}")
        if same:
            print(">>> CRITICAL: Image is NOT being processed!")
        else:
            print(">>> Image IS affecting output")

    # --- Test 2: Minimal prompt ---
    print("\n" + "=" * 60)
    print("TEST 2: Ultra-simple prompt")
    print("=" * 60)

    for attempt in range(2):
        try:
            resp2 = await adapter.generate_with_image(
                prompt="Describe this image.",
                image_path=test_image,
                max_tokens=150,
            )
            print(f"  Response: {resp2.text.strip()[:400]}")
            break
        except Exception as e:
            print(f"  FAILED: {e}")
            if attempt < 1:
                await asyncio.sleep(5)

    # --- Test 3: Text-first ordering (official recommendation) ---
    print("\n" + "=" * 60)
    print("TEST 3: Compare image ordering (image-first vs text-first)")
    print("=" * 60)

    # Current: image first, text second (our code)
    # Official: text first, image second
    import base64, mimetypes

    b64_data, mime = VertexMedGemmaAdapter._encode_image(test_image)
    image_block = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_data}"}}
    text_block = {"type": "text", "text": "What do you see in this medical image? Be specific."}

    # 3a: Image first (current code)
    print("\n--- 3a: Image-first (current) ---")
    try:
        payload_img_first = {
            "instances": [{
                "@requestFormat": "chatCompletions",
                "messages": [{"role": "user", "content": [image_block, text_block]}],
                "max_tokens": 200,
                "temperature": 0.0,
            }]
        }
        start = time.perf_counter()
        data = await adapter._post_with_retry(payload_img_first)
        elapsed = (time.perf_counter() - start) * 1000
        text, _, _, _ = adapter._extract_text_and_usage(data)
        print(f"  Latency: {elapsed:.0f}ms")
        print(f"  Response: {text.strip()[:400]}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 3b: Text first (official recommendation)
    print("\n--- 3b: Text-first (official) ---")
    try:
        payload_txt_first = {
            "instances": [{
                "@requestFormat": "chatCompletions",
                "messages": [{"role": "user", "content": [text_block, image_block]}],
                "max_tokens": 200,
                "temperature": 0.0,
            }]
        }
        start = time.perf_counter()
        data = await adapter._post_with_retry(payload_txt_first)
        elapsed = (time.perf_counter() - start) * 1000
        text, _, _, _ = adapter._extract_text_and_usage(data)
        print(f"  Latency: {elapsed:.0f}ms")
        print(f"  Response: {text.strip()[:400]}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # --- Test 4: With system message (official recommendation) ---
    print("\n" + "=" * 60)
    print("TEST 4: With system message (official format)")
    print("=" * 60)

    try:
        payload_sys = {
            "instances": [{
                "@requestFormat": "chatCompletions",
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an expert radiologist."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the key findings in this medical image."},
                            image_block,
                        ],
                    },
                ],
                "max_tokens": 300,
                "temperature": 0.0,
            }]
        }
        start = time.perf_counter()
        data = await adapter._post_with_retry(payload_sys)
        elapsed = (time.perf_counter() - start) * 1000
        text, _, _, _ = adapter._extract_text_and_usage(data)
        print(f"  Latency: {elapsed:.0f}ms")
        print(f"  Response: {text.strip()[:500]}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # --- Test 5: Base64 sizes ---
    print("\n" + "=" * 60)
    print("TEST 5: Base64 encoding verification")
    print("=" * 60)

    b64_rgb, mime_rgb = VertexMedGemmaAdapter._encode_image(test_image)
    print(f"  RGB image: mime={mime_rgb}, b64_len={len(b64_rgb)}, ~{len(b64_rgb)*3/4/1024:.1f} KB")

    if rgba_image.exists():
        img2 = Image.open(rgba_image)
        print(f"  RGBA image: mode={img2.mode}, size={img2.size}")
        img2.close()
        b64_rgba, mime_rgba = VertexMedGemmaAdapter._encode_image(rgba_image)
        print(f"  RGBA→RGB: mime={mime_rgba}, b64_len={len(b64_rgba)}, ~{len(b64_rgba)*3/4/1024:.1f} KB")


if __name__ == "__main__":
    asyncio.run(main())
