"""Deploy MedGemma 27B to Vertex AI Model Garden.

Uses vLLM container with 4x NVIDIA L4 on g2-standard-48.
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
DISPLAY_NAME = "medgemma-27b-it"
ENDPOINT_DISPLAY_NAME = "medgemma-27b-endpoint"

# vLLM container from Model Garden
SERVE_DOCKER_URI = (
    "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/"
    "pytorch-vllm-serve:20250430_0916_RC00_maas"
)

# Hardware: 4x L4 on g2-standard-48 (96GB VRAM total, 27B bf16 ~54GB)
MACHINE_TYPE = "g2-standard-48"
ACCELERATOR_TYPE = "NVIDIA_L4"
ACCELERATOR_COUNT = 4


def main():
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set. Required to download model from HuggingFace.")
        sys.exit(1)

    print(f"Project: {PROJECT_ID}")
    print(f"Region: {REGION}")
    print(f"Model: {MODEL_ID}")
    print(f"Machine: {MACHINE_TYPE} with {ACCELERATOR_COUNT}x {ACCELERATOR_TYPE}")
    print(f"Container: {SERVE_DOCKER_URI}")
    print()

    aiplatform.init(project=PROJECT_ID, location=REGION)

    # --- Step 1: Upload model ---
    print("=" * 60)
    print("Step 1: Uploading model to Vertex AI...")
    print("=" * 60)

    vllm_args = [
        "python", "-m", "vllm.entrypoints.api_server",
        "--host=0.0.0.0",
        "--port=8080",
        f"--model={MODEL_ID}",
        f"--tensor-parallel-size={ACCELERATOR_COUNT}",
        "--swap-space=16",
        "--gpu-memory-utilization=0.95",
        "--max-model-len=8192",  # Conservative for stability
        "--max-num-seqs=4",
        "--enable-chunked-prefill",
        "--disable-log-stats",
    ]

    env_vars = {
        "MODEL_ID": MODEL_ID,
        "DEPLOY_SOURCE": "script",
        "VLLM_USE_V1": "0",  # Use v0 engine for stability
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

    # --- Step 2: Create endpoint ---
    print("=" * 60)
    print("Step 2: Creating endpoint...")
    print("=" * 60)

    endpoint = aiplatform.Endpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        dedicated_endpoint_enabled=True,
    )

    print(f"Endpoint created: {endpoint.resource_name}")
    print()

    # --- Step 3: Deploy model to endpoint ---
    print("=" * 60)
    print("Step 3: Deploying model (this takes 15-30 minutes)...")
    print("=" * 60)

    deploy_start = time.time()

    model.deploy(
        endpoint=endpoint,
        machine_type=MACHINE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
        accelerator_count=ACCELERATOR_COUNT,
        deploy_request_timeout=3600,  # 1 hour timeout
        service_account=None,  # Use default
    )

    deploy_elapsed = time.time() - deploy_start
    print(f"Deployment complete in {deploy_elapsed/60:.1f} minutes")
    print()

    # --- Step 4: Extract endpoint details ---
    endpoint_id = endpoint.resource_name.split("/")[-1]

    # Try to get dedicated endpoint DNS
    dedicated_dns = ""
    try:
        dedicated_dns = endpoint.gca_resource.dedicated_endpoint_dns
    except Exception:
        pass

    print("=" * 60)
    print("DEPLOYMENT SUCCESSFUL")
    print("=" * 60)
    print(f"Endpoint ID: {endpoint_id}")
    print(f"Endpoint resource: {endpoint.resource_name}")
    if dedicated_dns:
        print(f"Dedicated DNS: {dedicated_dns}")
    print()
    print("Add these to your .env:")
    print(f"  GCP_PROJECT_ID={PROJECT_ID}")
    print(f"  GCP_REGION={REGION}")
    print(f"  VERTEX_ENDPOINT_ID_27B={endpoint_id}")
    if dedicated_dns:
        print(f"  VERTEX_DEDICATED_DNS_27B={dedicated_dns}")
    print()
    print("To run benchmark:")
    print("  uv run trialmatch phase0 --config configs/phase0_vertex_27b.yaml")


if __name__ == "__main__":
    main()
