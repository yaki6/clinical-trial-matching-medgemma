#!/usr/bin/env bash
# deploy_vertex.sh — Deploy MedGemma to Vertex AI Model Garden
#
# Usage:
#   bash scripts/deploy_vertex.sh --model-size=4b --region=australia-southeast1
#   bash scripts/deploy_vertex.sh --model-size=27b --region=us-central1
#   bash scripts/deploy_vertex.sh --teardown --endpoint-id=<id> --region=<region>
set -euo pipefail

# ---------------------------------------------------------------------------
# Color helpers (disabled when not a terminal)
# ---------------------------------------------------------------------------
if [[ -t 1 ]]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  CYAN='\033[0;36m'
  BOLD='\033[1m'
  RESET='\033[0m'
else
  RED='' GREEN='' YELLOW='' CYAN='' BOLD='' RESET=''
fi

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
die()     { error "$@"; exit 1; }

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_SIZE="4b"
REGION="us-central1"
PROJECT=""
TEARDOWN=false
ENDPOINT_ID=""
POLL_INTERVAL=30   # seconds between deployment status checks

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
for arg in "$@"; do
  case "$arg" in
    --model-size=*) MODEL_SIZE="${arg#*=}" ;;
    --region=*)     REGION="${arg#*=}" ;;
    --project=*)    PROJECT="${arg#*=}" ;;
    --teardown)     TEARDOWN=true ;;
    --endpoint-id=*) ENDPOINT_ID="${arg#*=}" ;;
    --help|-h)
      echo "Usage:"
      echo "  $0 --model-size=4b  --region=australia-southeast1"
      echo "  $0 --model-size=27b --region=us-central1"
      echo "  $0 --teardown --endpoint-id=<id> --region=<region>"
      echo ""
      echo "Options:"
      echo "  --model-size   4b or 27b (default: 4b)"
      echo "  --region       GCP region (default: us-central1)"
      echo "  --project      GCP project ID (default: from gcloud config)"
      echo "  --teardown     Undeploy and delete the endpoint"
      echo "  --endpoint-id  Endpoint ID (required for --teardown)"
      exit 0
      ;;
    *) die "Unknown argument: $arg" ;;
  esac
done

# ---------------------------------------------------------------------------
# Model ID mapping
# ---------------------------------------------------------------------------
case "$MODEL_SIZE" in
  4b)  MODEL_ID="google/medgemma@medgemma-4b-it" ;;
  27b) MODEL_ID="google/medgemma@medgemma-27b-it" ;;
  *)   die "Invalid --model-size=$MODEL_SIZE. Must be 4b or 27b." ;;
esac

# ---------------------------------------------------------------------------
# Resolve project from gcloud config if not provided
# ---------------------------------------------------------------------------
if [[ -z "$PROJECT" ]]; then
  PROJECT="$(gcloud config get-value project 2>/dev/null)" || true
  if [[ -z "$PROJECT" ]]; then
    die "No --project specified and gcloud config has no default project."
  fi
fi

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
info "Running preflight checks..."

# 1. Verify gcloud auth
if ! gcloud auth print-access-token &>/dev/null; then
  die "gcloud auth failed. Run 'gcloud auth login' first."
fi
success "gcloud authenticated"

# 2. Check Vertex AI API enabled
if ! gcloud services list --enabled --filter="name:aiplatform.googleapis.com" \
       --project="$PROJECT" --format="value(name)" 2>/dev/null | grep -q aiplatform; then
  die "Vertex AI API (aiplatform.googleapis.com) is not enabled in project $PROJECT.\n" \
      "       Run: gcloud services enable aiplatform.googleapis.com --project=$PROJECT"
fi
success "Vertex AI API enabled"

echo ""
info "${BOLD}Configuration${RESET}"
info "  Project:    $PROJECT"
info "  Region:     $REGION"
info "  Model size: $MODEL_SIZE"
info "  Model ID:   $MODEL_ID"
echo ""

# ---------------------------------------------------------------------------
# Teardown path
# ---------------------------------------------------------------------------
if [[ "$TEARDOWN" == true ]]; then
  if [[ -z "$ENDPOINT_ID" ]]; then
    die "--teardown requires --endpoint-id=<id>"
  fi

  info "Tearing down endpoint ${BOLD}$ENDPOINT_ID${RESET} ..."

  # Step 1: Undeploy model from endpoint
  info "Undeploying model from endpoint..."
  if gcloud ai endpoints undeploy-model "$ENDPOINT_ID" \
       --project="$PROJECT" \
       --region="$REGION" \
       --quiet 2>&1; then
    success "Model undeployed from endpoint $ENDPOINT_ID"
  else
    warn "Undeploy may have partially failed (endpoint may have no deployed models)."
  fi

  # Step 2: Delete endpoint
  info "Deleting endpoint..."
  if gcloud ai endpoints delete "$ENDPOINT_ID" \
       --project="$PROJECT" \
       --region="$REGION" \
       --quiet 2>&1; then
    success "Endpoint $ENDPOINT_ID deleted"
  else
    die "Failed to delete endpoint $ENDPOINT_ID"
  fi

  echo ""
  success "${BOLD}Teardown complete.${RESET}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------
info "Deploying ${BOLD}MedGemma $MODEL_SIZE${RESET} to Vertex AI Model Garden..."
echo ""

DISPLAY_NAME="medgemma-${MODEL_SIZE}"

DEPLOY_OUTPUT=$(gcloud ai model-garden models deploy \
  --model="$MODEL_ID" \
  --project="$PROJECT" \
  --region="$REGION" \
  --endpoint-display-name="$DISPLAY_NAME" \
  --accept-eula \
  --asynchronous 2>&1) || die "Deploy command failed:\n$DEPLOY_OUTPUT"

info "Deploy command accepted."
echo "$DEPLOY_OUTPUT"
echo ""

# Extract operation name from output.
# Typical output contains a line like:
#   operationName: projects/.../locations/.../operations/1234567890
OP_NAME=$(echo "$DEPLOY_OUTPUT" | grep -oE 'operations/[0-9]+' | head -1) || true

if [[ -z "$OP_NAME" ]]; then
  # Try alternative: full resource name
  OP_NAME=$(echo "$DEPLOY_OUTPUT" | grep -oE 'projects/[^[:space:]]+/operations/[0-9]+' | head -1) || true
fi

if [[ -z "$OP_NAME" ]]; then
  warn "Could not parse operation name from deploy output."
  warn "Check status manually:"
  echo "  gcloud ai operations list --project=$PROJECT --region=$REGION"
  exit 0
fi

info "Operation: ${BOLD}$OP_NAME${RESET}"
info "Polling deployment status every ${POLL_INTERVAL}s ..."
echo ""

# ---------------------------------------------------------------------------
# Poll until done
# ---------------------------------------------------------------------------
SECONDS_ELAPSED=0
while true; do
  DONE=$(gcloud ai operations describe "$OP_NAME" \
    --project="$PROJECT" \
    --region="$REGION" \
    --format="value(done)" 2>/dev/null) || true

  if [[ "$DONE" == "True" || "$DONE" == "true" ]]; then
    break
  fi

  SECONDS_ELAPSED=$((SECONDS_ELAPSED + POLL_INTERVAL))
  MINUTES=$((SECONDS_ELAPSED / 60))
  info "  Still deploying... (${MINUTES}m ${SECONDS_ELAPSED}s elapsed)"
  sleep "$POLL_INTERVAL"
done

echo ""
success "Deployment operation completed!"

# ---------------------------------------------------------------------------
# Extract endpoint ID
# ---------------------------------------------------------------------------
OP_DETAIL=$(gcloud ai operations describe "$OP_NAME" \
  --project="$PROJECT" \
  --region="$REGION" \
  --format=json 2>/dev/null) || true

# The endpoint ID is typically in the response/metadata of the operation.
DEPLOYED_ENDPOINT=$(echo "$OP_DETAIL" \
  | grep -oE 'endpoints/[0-9]+' | head -1 | sed 's|endpoints/||') || true

if [[ -z "$DEPLOYED_ENDPOINT" ]]; then
  warn "Could not auto-extract endpoint ID from operation response."
  warn "List endpoints manually:"
  echo "  gcloud ai endpoints list --project=$PROJECT --region=$REGION --filter=displayName=$DISPLAY_NAME"
  echo ""
  echo "Then set the env var yourself (see below)."
else
  success "Endpoint ID: ${BOLD}$DEPLOYED_ENDPOINT${RESET}"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}========================================${RESET}"
echo -e "${GREEN}${BOLD}  Deployment Complete${RESET}"
echo -e "${GREEN}${BOLD}========================================${RESET}"
echo ""
info "Project:     $PROJECT"
info "Region:      $REGION"
info "Model:       MedGemma $MODEL_SIZE ($MODEL_ID)"
info "Display:     $DISPLAY_NAME"
if [[ -n "$DEPLOYED_ENDPOINT" ]]; then
  info "Endpoint ID: $DEPLOYED_ENDPOINT"
fi
echo ""

# Env var guidance
ENV_VAR_NAME="VERTEX_ENDPOINT_ID_${MODEL_SIZE^^}"
echo -e "${YELLOW}Set the following environment variable for trialmatch:${RESET}"
echo ""
if [[ -n "$DEPLOYED_ENDPOINT" ]]; then
  echo "  export GCP_PROJECT_ID=\"$PROJECT\""
  echo "  export GCP_REGION=\"$REGION\""
  echo "  export $ENV_VAR_NAME=\"$DEPLOYED_ENDPOINT\""
else
  echo "  export GCP_PROJECT_ID=\"$PROJECT\""
  echo "  export GCP_REGION=\"$REGION\""
  echo "  export $ENV_VAR_NAME=\"<endpoint-id-from-above>\""
fi
echo ""

# Config guidance
CONFIG_FILE="configs/phase0_vertex_${MODEL_SIZE}.yaml"
echo -e "${CYAN}Run the benchmark:${RESET}"
echo "  uv run trialmatch phase0 --config $CONFIG_FILE"
echo ""

# Teardown reminder
echo -e "${YELLOW}To tear down when done (avoid ongoing charges):${RESET}"
if [[ -n "$DEPLOYED_ENDPOINT" ]]; then
  echo "  bash scripts/deploy_vertex.sh --teardown --endpoint-id=$DEPLOYED_ENDPOINT --region=$REGION"
else
  echo "  bash scripts/deploy_vertex.sh --teardown --endpoint-id=<ENDPOINT_ID> --region=$REGION"
fi
echo ""
