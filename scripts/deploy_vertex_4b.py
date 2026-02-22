"""Deploy / undeploy / status / smoke-test / run MedGemma 4B (multimodal) on Vertex AI.

Uses vLLM container with multimodal support (--limit-mm-per-prompt) on 1x NVIDIA L4.
4B x 2 bytes (fp16) = 8GB model weights, fits easily in 1x L4 (24GB VRAM).

Usage:
    uv run python scripts/deploy_vertex_4b.py deploy      # upload model + deploy + smoke test
    uv run python scripts/deploy_vertex_4b.py undeploy    # undeploy all models (stops billing)
    uv run python scripts/deploy_vertex_4b.py status      # list deployed models
    uv run python scripts/deploy_vertex_4b.py smoke-test  # poll until inference succeeds
    uv run python scripts/deploy_vertex_4b.py run -- <cmd> # deploy → run cmd → undeploy
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from google.cloud import aiplatform

# --- Configuration ---
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "gen-lang-client-0517724223")
REGION = "us-central1"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID = "google/medgemma-4b-it"
DISPLAY_NAME = "medgemma-4b-it-imaging"
ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID_4B", "923518299076034560")
DEDICATED_DNS = os.environ.get("VERTEX_DEDICATED_DNS_4B", "")

# vLLM container from Model Garden (same as 27B)
SERVE_DOCKER_URI = (
    "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/"
    "pytorch-vllm-serve:20250430_0916_RC00_maas"
)

# Hardware: 1x L4 on g2-standard-16 (24GB VRAM, 4B bf16 ~8GB + KV cache)
# T4 (16GB) is NOT viable — OOMs with vLLM multimodal overhead (tested Feb 22).
# If L4 quota exhausted by 27B, use time-share: undeploy 27B → deploy 4B → undeploy 4B → redeploy 27B.
MACHINE_TYPE = "g2-standard-16"
ACCELERATOR_TYPE = "NVIDIA_L4"
ACCELERATOR_COUNT = 1

VLLM_ARGS = [
    "python", "-m", "vllm.entrypoints.api_server",
    "--host=0.0.0.0",
    "--port=8080",
    f"--model={MODEL_ID}",
    f"--tensor-parallel-size={ACCELERATOR_COUNT}",
    "--swap-space=8",
    "--gpu-memory-utilization=0.90",
    "--max-model-len=4096",
    "--max-num-seqs=4",
    "--limit-mm-per-prompt=image=4",
    "--enable-chunked-prefill",
    "--disable-log-stats",
]

ENV_VARS = {
    "MODEL_ID": MODEL_ID,
    "DEPLOY_SOURCE": "script",
    "VLLM_USE_V1": "0",
    "HF_TOKEN": HF_TOKEN,
}


def _init():
    aiplatform.init(project=PROJECT_ID, location=REGION)


def _get_endpoint():
    return aiplatform.Endpoint(ENDPOINT_ID)


def _find_existing_model():
    models = aiplatform.Model.list(filter=f'display_name="{DISPLAY_NAME}"')
    if models:
        models.sort(key=lambda m: m.create_time, reverse=True)
        return models[0]
    return None


def _get_deployed_models(endpoint):
    return list(endpoint.gca_resource.deployed_models)


def cmd_status(_args):
    _init()
    endpoint = _get_endpoint()

    print(f"Endpoint ID:   {ENDPOINT_ID}")
    print(f"Endpoint name: {endpoint.display_name}")

    deployed = _get_deployed_models(endpoint)
    if not deployed:
        print("\nNo models deployed. Endpoint is idle ($0/hr).")
        return

    print(f"\nDeployed models ({len(deployed)}):")
    for dm in deployed:
        print(f"  - ID: {dm.id}")
        print(f"    Model: {dm.model}")
        print(f"    Display name: {dm.display_name}")
        print(f"    Machine type: {dm.dedicated_resources.machine_spec.machine_type}")
        accel = dm.dedicated_resources.machine_spec.accelerator_type
        count = dm.dedicated_resources.machine_spec.accelerator_count
        print(f"    Accelerator: {count}x {accel.name if hasattr(accel, 'name') else accel}")

    dns = ""
    try:
        dns = endpoint.gca_resource.dedicated_endpoint_dns
    except Exception:
        pass
    if dns:
        print(f"\nDedicated DNS: {dns}")


def cmd_deploy(_args):
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set. Required to download model from HuggingFace.")
        sys.exit(1)

    _init()

    print(f"Project:  {PROJECT_ID}")
    print(f"Region:   {REGION}")
    print(f"Model:    {MODEL_ID}")
    print(f"Hardware: {MACHINE_TYPE} with {ACCELERATOR_COUNT}x {ACCELERATOR_TYPE}")
    print(f"Endpoint: {ENDPOINT_ID}")
    print(f"Multimodal: --limit-mm-per-prompt=image=4")
    print()

    endpoint = _get_endpoint()
    deployed = _get_deployed_models(endpoint)
    if deployed:
        print(f"Endpoint already has {len(deployed)} deployed model(s).")
        print("Run 'undeploy' first if you want to redeploy.")
        sys.exit(1)

    # Step 1: Find or upload model
    print("=" * 60)
    print("Step 1: Checking Model Registry...")
    print("=" * 60)

    model = _find_existing_model()
    if model:
        print(f"Found existing model: {model.resource_name}")
        print("Skipping re-upload.")
    else:
        print("Uploading new model...")
        model = aiplatform.Model.upload(
            display_name=DISPLAY_NAME,
            serving_container_image_uri=SERVE_DOCKER_URI,
            serving_container_args=VLLM_ARGS,
            serving_container_ports=[8080],
            serving_container_predict_route="/generate",
            serving_container_health_route="/ping",
            serving_container_environment_variables=ENV_VARS,
        )
        print(f"Model uploaded: {model.resource_name}")
    print()

    # Step 2: Deploy
    print("=" * 60)
    print("Step 2: Deploying to endpoint...")
    print(f"  Endpoint ID: {ENDPOINT_ID}")
    print("  This takes 10-20 minutes...")
    print("=" * 60)

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

    dns = ""
    try:
        dns = endpoint.gca_resource.dedicated_endpoint_dns
    except Exception:
        pass

    print("=" * 60)
    print("DEPLOYMENT SUCCESSFUL")
    print("=" * 60)
    print(f"Endpoint ID: {ENDPOINT_ID}")
    if dns:
        print(f"Dedicated DNS: {dns}")
    print()

    # Step 3: Smoke test
    print("Running smoke test...")
    _run_smoke_test(dns or DEDICATED_DNS, max_attempts=20, interval=30)


def cmd_undeploy(_args):
    _init()
    endpoint = _get_endpoint()

    deployed = _get_deployed_models(endpoint)
    if not deployed:
        print("No models deployed. Nothing to undeploy.")
        return

    print(f"Undeploying {len(deployed)} model(s) from endpoint {ENDPOINT_ID}...")
    endpoint.undeploy_all()
    print("Done. All models undeployed. Billing stopped ($0/hr).")


def cmd_smoke_test(_args):
    _init()
    endpoint = _get_endpoint()

    deployed = _get_deployed_models(endpoint)
    if not deployed:
        print("ERROR: No models deployed. Run 'deploy' first.")
        sys.exit(1)

    dns = DEDICATED_DNS
    try:
        dns = endpoint.gca_resource.dedicated_endpoint_dns or dns
    except Exception:
        pass

    _run_smoke_test(dns, max_attempts=10, interval=30)


def cmd_run(args):
    """Deploy → run command → undeploy. Always undeploys, even on failure."""
    if not args.cmd:
        print("ERROR: No command provided. Usage: run -- <command>")
        sys.exit(1)

    cmd_str = " ".join(args.cmd)
    print(f"=== RUN MODE: deploy → {cmd_str} → undeploy ===")
    print()

    # Step 1: Deploy (if not already deployed)
    _init()
    endpoint = _get_endpoint()
    deployed = _get_deployed_models(endpoint)
    already_deployed = bool(deployed)

    if already_deployed:
        print("Endpoint already has deployed model(s). Skipping deploy.")
    else:
        cmd_deploy(args)

    # Step 2: Run the user command
    print()
    print("=" * 60)
    print(f"Running: {cmd_str}")
    print("=" * 60)

    cmd_exit_code = 1
    try:
        result = subprocess.run(args.cmd, cwd=os.getcwd())
        cmd_exit_code = result.returncode
        if cmd_exit_code == 0:
            print(f"\nCommand succeeded (exit {cmd_exit_code})")
        else:
            print(f"\nCommand failed (exit {cmd_exit_code})")
    except KeyboardInterrupt:
        print("\nCommand interrupted by user")
    except Exception as e:
        print(f"\nCommand error: {e}")

    # Step 3: Always undeploy (unless it was already deployed before we started)
    if not already_deployed:
        print()
        print("=" * 60)
        print("Auto-undeploying to stop billing...")
        print("=" * 60)
        try:
            cmd_undeploy(args)
        except Exception as e:
            print(f"WARNING: Undeploy failed: {e}")
            print("MANUAL ACTION REQUIRED: run 'undeploy' to stop billing!")
    else:
        print("\nSkipping auto-undeploy (endpoint was already deployed before run).")

    sys.exit(cmd_exit_code)


def _run_smoke_test(dedicated_dns, max_attempts=10, interval=30):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from trialmatch.models.vertex_medgemma import VertexMedGemmaAdapter

    adapter = VertexMedGemmaAdapter(
        project_id=PROJECT_ID,
        region=REGION,
        endpoint_id=ENDPOINT_ID,
        model_name="medgemma-4b-vertex",
        dedicated_endpoint_dns=dedicated_dns or None,
        gpu_hourly_rate=1.15,  # 1x L4
    )

    print(f"Polling endpoint (max {max_attempts} attempts, {interval}s interval)...")
    for attempt in range(1, max_attempts + 1):
        start = time.time()
        ok = asyncio.run(adapter.health_check())
        elapsed = (time.time() - start) * 1000

        if ok:
            print(f"  Attempt {attempt}: OK ({elapsed:.0f}ms)")
            print(f"\nSmoke test passed! First inference latency: {elapsed:.0f}ms")
            return
        else:
            print(f"  Attempt {attempt}: not ready ({elapsed:.0f}ms)")
            if attempt < max_attempts:
                time.sleep(interval)

    print(f"\nSmoke test FAILED after {max_attempts} attempts.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Manage MedGemma 4B (multimodal) on Vertex AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("deploy", help="Upload model (if needed) + deploy + smoke test")
    sub.add_parser("undeploy", help="Undeploy all models (stops billing, keeps endpoint)")
    sub.add_parser("status", help="Show endpoint status and deployed models")
    sub.add_parser("smoke-test", help="Poll until first successful inference")
    run_parser = sub.add_parser(
        "run", help="Deploy → run command → auto-undeploy (always undeploys)"
    )
    run_parser.add_argument(
        "cmd", nargs=argparse.REMAINDER,
        help="Command to run (after --). E.g.: run -- uv run trialmatch phase0 ..."
    )

    args = parser.parse_args()

    # Strip leading '--' from remainder args
    if hasattr(args, "cmd") and args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]

    commands = {
        "deploy": cmd_deploy,
        "undeploy": cmd_undeploy,
        "status": cmd_status,
        "smoke-test": cmd_smoke_test,
        "run": cmd_run,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
