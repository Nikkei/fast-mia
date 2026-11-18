#!/bin/bash
# Submit a fast-mia evaluation job to a Google Cloud TPU VM.
#
# Usage:
#   ./gcp/submit_tpu_job.sh \
#     --config config/tpu-sample.yaml \
#     --bucket gs://my-bucket/fast-mia-results \
#     [--project my-gcp-project] \
#     [--zone us-central1-b] \
#     [--accelerator-type v6e-1] \
#     [--runtime-version v2-alpha-tpuv6e] \
#     [--tpu-name fast-mia-tpu] \
#     [--vllm-tpu-spec "vllm-tpu"] \
#     [--spot] \
#     [--extra-args "--seed 42 --detailed-report"] \
#     [--delete-after]
#
# If --tpu-name matches an existing STOPPED or READY TPU VM, it will be
# started/reused (skipping creation and environment setup).
# After the job, the TPU VM is stopped by default (preserving Hugging Face and
# XLA compilation caches). Pass --delete-after to delete it instead.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (with alpha/compute tpus tpu-vm)
#   - Sufficient Cloud TPU quota in the target zone
#   - GCS bucket created
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────
CONFIG=""
GCS_BUCKET=""
PROJECT=""
ZONE="us-central1-b"
ACCELERATOR_TYPE="v6e-1"
RUNTIME_VERSION=""
TPU_NAME="fast-mia-tpu-$(date +%Y%m%d-%H%M%S)"
VLLM_TPU_SPEC="vllm-tpu"
SPOT=false
EXTRA_ARGS=""
DELETE_AFTER=false

# ── Parse arguments ───────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)           CONFIG="$2"; shift 2;;
        --bucket)           GCS_BUCKET="$2"; shift 2;;
        --project)          PROJECT="$2"; shift 2;;
        --zone)             ZONE="$2"; shift 2;;
        --accelerator-type) ACCELERATOR_TYPE="$2"; shift 2;;
        --runtime-version|--version) RUNTIME_VERSION="$2"; shift 2;;
        --tpu-name|--instance-name)  TPU_NAME="$2"; shift 2;;
        --vllm-tpu-spec)    VLLM_TPU_SPEC="$2"; shift 2;;
        --spot)             SPOT=true; shift;;
        --extra-args)       EXTRA_ARGS="$2"; shift 2;;
        --delete-after)     DELETE_AFTER=true; shift;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ── Validate ──────────────────────────────────────────────
if [[ -z "$CONFIG" ]]; then
    echo "ERROR: --config is required" >&2
    exit 1
fi
if [[ -z "$GCS_BUCKET" ]]; then
    echo "ERROR: --bucket is required (e.g. gs://my-bucket/fast-mia-results)" >&2
    exit 1
fi

# Resolve project root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$PROJECT_ROOT/$CONFIG" && ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG" >&2
    exit 1
fi

# Infer default TPU runtime version from accelerator family when omitted
if [[ -z "$RUNTIME_VERSION" ]]; then
    if [[ "$ACCELERATOR_TYPE" == v6e-* ]]; then
        RUNTIME_VERSION="v2-alpha-tpuv6e"
    elif [[ "$ACCELERATOR_TYPE" == v5litepod-* ]]; then
        RUNTIME_VERSION="v2-alpha-tpuv5-lite"
    else
        RUNTIME_VERSION="tpu-ubuntu2204-base"
    fi
fi

PROJECT_FLAG=""
if [[ -n "$PROJECT" ]]; then
    PROJECT_FLAG="--project=$PROJECT"
fi

SPOT_FLAG=""
if [[ "$SPOT" == "true" ]]; then
    SPOT_FLAG="--spot"
fi

# ── Check for existing TPU VM ────────────────────────────
REUSE_TPU=false
# shellcheck disable=SC2086
EXISTING_STATE=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$ZONE" $PROJECT_FLAG \
    --format="value(state)" 2>/dev/null || echo "NOT_FOUND")

if [[ "$EXISTING_STATE" == "STOPPED" || "$EXISTING_STATE" == "READY" ]]; then
    REUSE_TPU=true
fi

echo "============================================"
echo " fast-mia Cloud TPU Job Submission"
echo "============================================"
echo " TPU VM:      $TPU_NAME"
echo " Zone:        $ZONE"
echo " TPU Type:    $ACCELERATOR_TYPE"
echo " Runtime:     $RUNTIME_VERSION"
echo " Spot VM:     $SPOT"
echo " vLLM Spec:   $VLLM_TPU_SPEC"
echo " Config:      $CONFIG"
echo " GCS Bucket:  $GCS_BUCKET"
echo " Extra args:  ${EXTRA_ARGS:-none}"
if [[ "$REUSE_TPU" == "true" ]]; then
    echo " Reusing:     yes (state: $EXISTING_STATE)"
fi
echo "============================================"

# ── Step 1: Create or start TPU VM ───────────────────────
echo ""
if [[ "$REUSE_TPU" == "true" ]]; then
    if [[ "$EXISTING_STATE" == "STOPPED" ]]; then
        echo "[1/5] Starting existing TPU VM..."
        # shellcheck disable=SC2086
        gcloud compute tpus tpu-vm start "$TPU_NAME" \
            --zone="$ZONE" $PROJECT_FLAG
    else
        echo "[1/5] TPU VM already READY."
    fi
else
    echo "[1/5] Creating Cloud TPU VM..."
    # shellcheck disable=SC2086
    gcloud compute tpus tpu-vm create "$TPU_NAME" \
        $PROJECT_FLAG \
        --zone="$ZONE" \
        --accelerator-type="$ACCELERATOR_TYPE" \
        --version="$RUNTIME_VERSION" \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        $SPOT_FLAG
fi

# Wait for SSH to be available
echo "Waiting for TPU VM to be reachable via SSH..."
for i in $(seq 1 60); do
    # shellcheck disable=SC2086
    if gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" $PROJECT_FLAG \
        --command="echo ready" &>/dev/null; then
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: TPU VM not reachable via SSH after 10 minutes." >&2
        exit 1
    fi
    sleep 10
done
echo "TPU VM is ready."

# ── Step 2: Transfer project files ──────────────────────
echo ""
echo "[2/5] Transferring project files..."
TMPTAR=$(mktemp /tmp/fast-mia-tpu-XXXXXX.tar.gz)
trap 'rm -f "$TMPTAR"' EXIT

tar -czf "$TMPTAR" \
    -C "$PROJECT_ROOT" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.venv' \
    --exclude='results' \
    --exclude='data' \
    --exclude='.ruff_cache' \
    .

# shellcheck disable=SC2086
gcloud compute tpus tpu-vm scp "$TMPTAR" "$TPU_NAME":~/fast-mia.tar.gz \
    --zone="$ZONE" $PROJECT_FLAG

# shellcheck disable=SC2086
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" $PROJECT_FLAG \
    --command="mkdir -p ~/fast-mia && tar -xzf ~/fast-mia.tar.gz -C ~/fast-mia"

rm -f "$TMPTAR"

# Transfer local data files if config references a local path
CONFIG_LOCAL_PATH="$PROJECT_ROOT/$CONFIG"
if [[ ! -f "$CONFIG_LOCAL_PATH" ]]; then
    CONFIG_LOCAL_PATH="$CONFIG"
fi
DATA_PATH=$(grep 'data_path:' "$CONFIG_LOCAL_PATH" | head -1 | sed -E 's/.*data_path:[[:space:]]*["'\'']?([^"'\''#]*)["'\'']?.*/\1/' | xargs)
if [[ -n "$DATA_PATH" && "$DATA_PATH" == ./* && -f "$PROJECT_ROOT/$DATA_PATH" ]]; then
    echo "Transferring data file: $DATA_PATH"
    # shellcheck disable=SC2086
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" $PROJECT_FLAG \
        --command="mkdir -p ~/fast-mia/$(dirname "$DATA_PATH")"
    # shellcheck disable=SC2086
    gcloud compute tpus tpu-vm scp "$PROJECT_ROOT/$DATA_PATH" "$TPU_NAME":~/fast-mia/"$DATA_PATH" \
        --zone="$ZONE" $PROJECT_FLAG
fi

# ── Step 3: Setup environment ────────────────────────────
if [[ "$REUSE_TPU" == "true" ]]; then
    echo ""
    echo "[3/5] Skipping base setup (reusing existing TPU VM)..."
else
    echo ""
    echo "[3/5] Setting up environment on TPU VM..."
    # shellcheck disable=SC2086
    gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" $PROJECT_FLAG \
        --command="bash ~/fast-mia/gcp/setup_tpu_instance.sh"
fi

# ── Step 4: Run the job ──────────────────────────────────
echo ""
echo "[4/5] Running fast-mia job on Cloud TPU..."
# Run via nohup with PJRT_DEVICE=TPU and persistent XLA compilation cache
# shellcheck disable=SC2086
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" $PROJECT_FLAG \
    --command="cd ~/fast-mia && rm -f job.done job.failed && export PATH=\$HOME/.local/bin:\$PATH && export PJRT_DEVICE=TPU && export XLA_PERSISTENT_CACHE_PATH=\$HOME/.cache/xla_compile_cache && nohup bash -c 'uv run --with \"$VLLM_TPU_SPEC\" python main.py --config $CONFIG $EXTRA_ARGS > job.log 2>&1 && touch job.done || touch job.failed' > /dev/null 2>&1 & disown"

echo "Waiting for TPU job to complete (polling every 30s; initial XLA compilation may take several minutes)..."
while true; do
    # shellcheck disable=SC2086
    STATUS=$(gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
        --zone="$ZONE" $PROJECT_FLAG \
        --command="if [ -f ~/fast-mia/job.done ]; then echo done; elif [ -f ~/fast-mia/job.failed ]; then echo failed; else echo running; fi" 2>/dev/null || echo "running")
    if [[ "$STATUS" == "done" ]]; then
        echo "Job completed successfully."
        break
    elif [[ "$STATUS" == "failed" ]]; then
        echo "Job failed. Remote log:" >&2
        # shellcheck disable=SC2086
        gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
            --zone="$ZONE" $PROJECT_FLAG \
            --command="tail -80 ~/fast-mia/job.log" 2>/dev/null
        exit 1
    fi
    sleep 30
done

# Show the job log
# shellcheck disable=SC2086
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" $PROJECT_FLAG \
    --command="cat ~/fast-mia/job.log" 2>/dev/null

# ── Step 5: Upload results to GCS ────────────────────────
echo ""
echo "[5/5] Uploading results to GCS..."
# shellcheck disable=SC2086
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --zone="$ZONE" $PROJECT_FLAG \
    --command="gcloud storage cp -r ~/fast-mia/results/* $GCS_BUCKET/ || gsutil -m cp -r ~/fast-mia/results/* $GCS_BUCKET/"

echo ""
echo "============================================"
echo " Results uploaded to: $GCS_BUCKET"
echo "============================================"

# ── Cleanup ──────────────────────────────────────────────
if [[ "$DELETE_AFTER" == "true" ]]; then
    echo ""
    echo "Deleting TPU VM $TPU_NAME..."
    # shellcheck disable=SC2086
    gcloud compute tpus tpu-vm delete "$TPU_NAME" \
        --zone="$ZONE" $PROJECT_FLAG \
        --quiet
    echo "TPU VM deleted."
else
    echo ""
    echo "Stopping TPU VM $TPU_NAME..."
    # shellcheck disable=SC2086
    gcloud compute tpus tpu-vm stop "$TPU_NAME" \
        --zone="$ZONE" $PROJECT_FLAG \
    || echo "WARNING: Could not stop TPU VM (note: Spot TPU VMs may only support delete)."
    echo "TPU VM stopped. To reuse it, pass: --tpu-name $TPU_NAME --zone $ZONE"
    echo "To delete it:"
    echo "  gcloud compute tpus tpu-vm delete $TPU_NAME --zone=$ZONE $PROJECT_FLAG --quiet"
fi

echo ""
echo "Done!"
