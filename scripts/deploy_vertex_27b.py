"""Deploy MedGemma 27B to Vertex AI Model Garden.

Uses vLLM container with bitsandbytes int8 quantization on 2x NVIDIA L4.
27B × 1 byte (int8) = 27GB, fits in 2x L4 (48GB VRAM total).

Based on: https://github.com/google-health/medgemma/blob/main/notebooks/quick_start_with_model_garden.ipynb
"""

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from google.cloud import aiplatform

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "gen-lang-client-0517724223")
REGION = "us-central1"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID = "google/medgemma-27b-it"
DISPLAY_NAME = "medgemma-27b-it-int8"
# Reuse existing endpoint
ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID_27B", "6588061467889631232")

# vLLM container from Model Garden
SERVE_DOCKER_URI = (
    "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/"
    "pytorch-vllm-serve:20250430_0916_RC00_maas"
)

# Hardware: 2x L4 on g2-standard-24 (48GB VRAM, 27B int8 ~27GB + KV cache)
MACHINE_TYPE = "g2-standard-24"
ACCELERATOR_TYPE = "NVIDIA_L4"
ACCELERATOR_COUNT = 2


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set. Required to download model from HuggingFace.")
        sys.exit(1)

    print(f"Project: {PROJECT_ID}")
    print(f"Region: {REGION}")
    print(f"Model: {MODEL_ID}")
    print(f"Machine: {MACHINE_TYPE} with {ACCELERATOR_COUNT}x {ACCELERATOR_TYPE}")
    print(f"Quantization: bitsandbytes int8 (~27GB model weights)")
    print(f"Container: {SERVE_DOCKER_URI}")
    print(f"Reusing endpoint: {ENDPOINT_ID}")
    print()

    aiplatform.init(project=PROJECT_ID, location=REGION)

    # --- Step 1: Upload model with int8 quantization ---
    print("=" * 60)
    print("Step 1: Uploading model to Vertex AI (int8 quantized)...")
    print("=" * 60)

    vllm_args = [
        "python", "-m", "vllm.entrypoints.api_server",
        "--host=0.0.0.0",
        "--port=8080",
        f"--model={MODEL_ID}",
        f"--tensor-parallel-size={ACCELERATOR_COUNT}",
        "--swap-space=16",
        "--gpu-memory-utilization=0.95",
        "--max-model-len=8192",
        "--max-num-seqs=4",
        "--enable-chunked-prefill",
        "--disable-log-stats",
        "--quantization=bitsandbytes",
        "--load-format=bitsandbytes",
    ]

    env_vars = {
        "MODEL_ID": MODEL_ID,
        "DEPLOY_SOURCE": "script",
        "VLLM_USE_V1": "0",
        "HF_TOKEN": HF_TOKEN,
    }

    model = aiplatform.Model.upload(
        display_name=DISPLAY_NAME,
        serving_container_image_uri=SERVE_DOCKER_URI,
        serving_container_args=vllm_args,
        serving_container_ports=[8080],
        serving_container_predict_route="/generate",
        serving_container_health_route="/ping",
        serving_container_environment_variables=env_vars,
    )

    print(f"Model uploaded: {model.resource_name}")
    print()

    # --- Step 2: Deploy to existing endpoint ---
    print("=" * 60)
    print("Step 2: Deploying model to existing endpoint...")
    print(f"  Endpoint ID: {ENDPOINT_ID}")
    print("  This takes 15-30 minutes...")
    print("=" * 60)

    endpoint = aiplatform.Endpoint(ENDPOINT_ID)
    deploy_start = time.time()

    model.deploy(
        endpoint=endpoint,
        machine_type=MACHINE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
        accelerator_count=ACCELERATOR_COUNT,
        deploy_request_timeout=3600,
        service_account=None,
    )

    deploy_elapsed = time.time() - deploy_start
    print(f"Deployment complete in {deploy_elapsed/60:.1f} minutes")
    print()

    # --- Step 3: Extract endpoint details ---
    dedicated_dns = ""
    try:
        dedicated_dns = endpoint.gca_resource.dedicated_endpoint_dns
    except Exception:
        pass

    print("=" * 60)
    print("DEPLOYMENT SUCCESSFUL")
    print("=" * 60)
    print(f"Endpoint ID: {ENDPOINT_ID}")
    print(f"Endpoint resource: {endpoint.resource_name}")
    if dedicated_dns:
        print(f"Dedicated DNS: {dedicated_dns}")
    print()
    print("Add these to your .env:")
    print(f"  GCP_PROJECT_ID={PROJECT_ID}")
    print(f"  GCP_REGION={REGION}")
    print(f"  VERTEX_ENDPOINT_ID_27B={ENDPOINT_ID}")
    if dedicated_dns:
        print(f"  VERTEX_DEDICATED_DNS_27B={dedicated_dns}")
    print()
    print("To run benchmark:")
    print("  uv run trialmatch phase0 --config configs/phase0_vertex_27b.yaml")


if __name__ == "__main__":
    main()
