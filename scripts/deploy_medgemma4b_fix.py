#!/usr/bin/env python3
"""Fix MedGemma 4B endpoint: push custom handler to resolve CUDA crash + temperature warning.

Diagnosis (from research report):
  1. Default HF toolkit routes `temperature` to AutoProcessor → warning, no effect
  2. Implicit float16 dtype on L4 GPU → cuBLAS misaligned-address CUDA crash
  3. Non-contiguous tensors → GEMM kernel failures

Fix: custom handler.py that loads in bfloat16, forces contiguous tensors,
     and routes temperature to model.generate() only.

Usage:
    # Push handler + update endpoint (default)
    HF_TOKEN=hf_xxx uv run python scripts/deploy_medgemma4b_fix.py

    # Just push handler (don't update endpoint yet)
    HF_TOKEN=hf_xxx uv run python scripts/deploy_medgemma4b_fix.py --push-only

    # Just update endpoint to use existing handler repo
    HF_TOKEN=hf_xxx uv run python scripts/deploy_medgemma4b_fix.py --update-only

    # Check current endpoint status
    HF_TOKEN=hf_xxx uv run python scripts/deploy_medgemma4b_fix.py --check

    # Rollback: revert endpoint to original Google model (no custom handler)
    HF_TOKEN=hf_xxx uv run python scripts/deploy_medgemma4b_fix.py --rollback
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, InferenceClient

ENDPOINT_NAME = "medgemma-1-5-4b-it-hae"
ORIGINAL_MODEL_REPO = "google/medgemma-1.5-4b-it"
HANDLER_REPO = "yakilee/medgemma-4b-handler"  # custom handler repo

HANDLER_FILE = Path(__file__).parent / "handler_medgemma4b.py"

REQUIREMENTS_TXT = """\
torch>=2.1.0
transformers>=4.45.0
Pillow>=10.0.0
accelerate>=0.27.0
"""


def get_api(token: str) -> HfApi:
    return HfApi(token=token)


def push_handler(token: str) -> str:
    """Create handler repo on HF Hub and push handler.py + requirements.txt."""
    api = get_api(token)

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=HANDLER_REPO, repo_type="model", exist_ok=True)
        print(f"Repo '{HANDLER_REPO}' ready.")
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload handler.py
    handler_path = str(HANDLER_FILE)
    if not Path(handler_path).exists():
        print(f"ERROR: {handler_path} not found", file=sys.stderr)
        sys.exit(1)

    api.upload_file(
        path_or_fileobj=handler_path,
        path_in_repo="handler.py",
        repo_id=HANDLER_REPO,
        repo_type="model",
        commit_message="fix: custom handler for bfloat16 dtype + contiguous tensors + temperature routing",
    )
    print(f"Pushed handler.py → {HANDLER_REPO}")

    # Upload requirements.txt
    api.upload_file(
        path_or_fileobj=REQUIREMENTS_TXT.encode(),
        path_in_repo="requirements.txt",
        repo_id=HANDLER_REPO,
        repo_type="model",
        commit_message="chore: pin handler dependencies",
    )
    print(f"Pushed requirements.txt → {HANDLER_REPO}")

    repo_url = f"https://huggingface.co/{HANDLER_REPO}"
    print(f"Handler repo: {repo_url}")
    return repo_url


def update_endpoint(token: str) -> None:
    """Update endpoint to use custom handler repo with env vars."""
    api = get_api(token)
    print(f"Updating endpoint '{ENDPOINT_NAME}' → repo={HANDLER_REPO}")

    api.update_inference_endpoint(
        name=ENDPOINT_NAME,
        repository=HANDLER_REPO,
        env={
            "HF_TOKEN": token,  # handler needs token to access gated model
            "CUDA_LAUNCH_BLOCKING": "1",  # synchronous CUDA for better error traces
        },
    )
    print("Endpoint updated. It will restart automatically (2-5 min).")
    print("Monitor status: uv run python scripts/deploy_medgemma4b_fix.py --check")


def rollback(token: str) -> None:
    """Revert endpoint to original Google model repo (no custom handler)."""
    api = get_api(token)
    print(f"Rolling back endpoint '{ENDPOINT_NAME}' → repo={ORIGINAL_MODEL_REPO}")

    api.update_inference_endpoint(
        name=ENDPOINT_NAME,
        repository=ORIGINAL_MODEL_REPO,
        env={},  # clear custom env vars
    )
    print("Endpoint reverted to original model. Restarting...")


def check(token: str) -> None:
    """Check endpoint status and current configuration."""
    api = get_api(token)
    try:
        endpoint = api.get_inference_endpoint(ENDPOINT_NAME)
        raw = endpoint.raw
        model = raw.get("model", {})
        print(f"Endpoint: {ENDPOINT_NAME}")
        print(f"  Status:     {endpoint.status}")
        print(f"  URL:        {endpoint.url}")
        print(f"  Repository: {model.get('repository', 'N/A')}")
        print(f"  Framework:  {model.get('framework', 'N/A')}")
        print(f"  Task:       {model.get('task', 'N/A')}")
        print(f"  Env vars:   {model.get('env', {})}")

        if endpoint.status == "running" and endpoint.url:
            print("\nRunning quick health check...")
            client = InferenceClient(model=endpoint.url, token=token)
            try:
                result = client.text_generation(
                    "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n",
                    max_new_tokens=5,
                )
                print(f"  Health: OK — response: {result!r}")
            except Exception as e:
                print(f"  Health: FAILED — {e!s:.120}")
    except Exception as e:
        print(f"Endpoint not found or error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix MedGemma 4B endpoint: custom handler for CUDA stability"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--push-only", action="store_true", help="Push handler, don't update endpoint")
    group.add_argument("--update-only", action="store_true", help="Update endpoint (handler already pushed)")
    group.add_argument("--check", action="store_true", help="Check endpoint status")
    group.add_argument("--rollback", action="store_true", help="Revert to original model")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr)
        sys.exit(1)

    if args.check:
        check(token)
    elif args.rollback:
        rollback(token)
    elif args.push_only:
        push_handler(token)
    elif args.update_only:
        update_endpoint(token)
    else:
        # Full deploy: push handler + update endpoint
        push_handler(token)
        print()
        update_endpoint(token)
        print()
        print("=" * 60)
        print("Next steps:")
        print("  1. Wait 2-5 min for endpoint to restart")
        print("  2. Check status:")
        print("     uv run python scripts/deploy_medgemma4b_fix.py --check")
        print("  3. If it fails, rollback:")
        print("     uv run python scripts/deploy_medgemma4b_fix.py --rollback")
        print("=" * 60)


if __name__ == "__main__":
    main()
