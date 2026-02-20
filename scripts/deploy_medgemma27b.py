#!/usr/bin/env python3
"""Deploy MedGemma 27B HuggingFace Inference Endpoint.

Usage:
    HF_TOKEN=hf_xxx uv run python scripts/deploy_medgemma27b.py
    HF_TOKEN=hf_xxx uv run python scripts/deploy_medgemma27b.py --check   # check existing
    HF_TOKEN=hf_xxx uv run python scripts/deploy_medgemma27b.py --delete  # delete endpoint

Hardware options (comment/uncomment in ENDPOINT_CONFIG below):
    nvidia-a100  x1 = 1× A100 80GB (~$2.50/hr)  ← default, cheapest A100
    nvidia-l40s  x2 = 2× L40S 48GB = 96GB total (~$3.60/hr) ← cheaper per GB
    nvidia-h100  x1 = 1× H100 80GB (~$10/hr)    ← fastest

MedGemma 27B requires ~54GB VRAM (27B × 2 bytes bfloat16).
Minimum: A100 80GB (single) or 2× L40S 48GB.

Uses TGI (Text Generation Inference) — unlike 4B which uses default HF image.
Scale-to-zero is critical: A100 at $2.50/hr adds up fast when idle.
"""

from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import HfApi, InferenceClient

ENDPOINT_NAME = "medgemma-27b-text-it"
MODEL_REPO = "google/medgemma-27b-text-it"

ENDPOINT_CONFIG = {
    "repository": MODEL_REPO,
    # "custom" framework + TGI docker loads directly in bf16 (~54GB).
    # "pytorch" would load float32 first (~108GB) → OOM on A100 80GB.
    "framework": "custom",
    "task": "text-generation",
    "accelerator": "gpu",
    "instance_type": "nvidia-a100",
    "instance_size": "x1",
    "region": "us-east-1",
    "vendor": "aws",
    "min_replica": 0,
    "max_replica": 1,
    "scale_to_zero_timeout": 15,
    "type": "protected",
}

# TGI docker image — loads model directly in bf16 with paged attention
CUSTOM_IMAGE = {
    "health_route": "/health",
    "url": "ghcr.io/huggingface/text-generation-inference:3.0.0",
    "env": {
        "DTYPE": "bfloat16",
        "MAX_INPUT_LENGTH": "4096",
        "MAX_TOTAL_TOKENS": "8192",
        "MAX_BATCH_PREFILL_TOKENS": "4096",
    },
}


def get_api(token: str) -> HfApi:
    return HfApi(token=token)


def deploy(token: str) -> str:
    """Deploy the endpoint and wait until running. Returns endpoint URL."""
    api = get_api(token)
    print(f"Creating endpoint '{ENDPOINT_NAME}' for {MODEL_REPO}...")
    print(f"  Instance: {ENDPOINT_CONFIG['instance_type']} {ENDPOINT_CONFIG['instance_size']}")
    print(f"  Scale-to-zero: {ENDPOINT_CONFIG['scale_to_zero_timeout']} min")

    endpoint = api.create_inference_endpoint(
        name=ENDPOINT_NAME,
        repository=ENDPOINT_CONFIG["repository"],
        framework=ENDPOINT_CONFIG["framework"],
        task=ENDPOINT_CONFIG["task"],
        accelerator=ENDPOINT_CONFIG["accelerator"],
        instance_type=ENDPOINT_CONFIG["instance_type"],
        instance_size=ENDPOINT_CONFIG["instance_size"],
        region=ENDPOINT_CONFIG["region"],
        vendor=ENDPOINT_CONFIG["vendor"],
        min_replica=ENDPOINT_CONFIG["min_replica"],
        max_replica=ENDPOINT_CONFIG["max_replica"],
        scale_to_zero_timeout=ENDPOINT_CONFIG["scale_to_zero_timeout"],
        type=ENDPOINT_CONFIG["type"],
        custom_image=CUSTOM_IMAGE,
    )
    print("Endpoint created. Waiting for 'running' status (up to 15 min)...")
    endpoint.wait(timeout=900)

    url = endpoint.url
    print(f"\nEndpoint is RUNNING: {url}")
    return url


def check(token: str) -> None:
    """Check status of existing endpoint."""
    api = get_api(token)
    try:
        endpoint = api.get_inference_endpoint(ENDPOINT_NAME)
        print(f"Endpoint: {ENDPOINT_NAME}")
        print(f"  Status: {endpoint.status}")
        print(f"  URL: {endpoint.url}")
    except Exception as e:
        print(f"Endpoint not found or error: {e}", file=sys.stderr)
        sys.exit(1)


def delete(token: str) -> None:
    """Delete the endpoint."""
    api = get_api(token)
    try:
        api.delete_inference_endpoint(ENDPOINT_NAME)
        print(f"Endpoint '{ENDPOINT_NAME}' deleted.")
    except Exception as e:
        print(f"Error deleting endpoint: {e}", file=sys.stderr)
        sys.exit(1)


def health_check(url: str, token: str) -> None:
    """Run a quick text generation to verify endpoint is responding."""
    print("\nRunning health check...")
    client = InferenceClient(model=url, token=token)
    prompt = "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
    result = client.text_generation(prompt, max_new_tokens=10)
    print(f"Health check OK — response: {result!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage MedGemma 27B HF Inference Endpoint")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Check status of existing endpoint")
    group.add_argument("--delete", action="store_true", help="Delete the endpoint")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr)
        sys.exit(1)

    if args.check:
        check(token)
    elif args.delete:
        delete(token)
    else:
        url = deploy(token)
        health_check(url, token)
        print("\n" + "=" * 60)
        print("Next steps:")
        print(f"  1. Copy this URL: {url}")
        print("  2. Add to your .env:  MEDGEMMA_27B_ENDPOINT_URL=" + url)
        print("  3. Update configs/phase0_medgemma27b_only.yaml endpoint_url field")
        print("  4. Update configs/phase0_three_way.yaml endpoint_url field for medgemma-27b")
        print("  5. Run smoke tests:")
        print(f"     MEDGEMMA_27B_ENDPOINT_URL={url} \\")
        print("     HF_TOKEN=$HF_TOKEN \\")
        print("     uv run pytest tests/smoke/test_medgemma27b_endpoint.py -v -m smoke")
        print("=" * 60)


if __name__ == "__main__":
    main()
