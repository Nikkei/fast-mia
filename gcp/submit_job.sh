#!/bin/bash
# Submit a fast-mia GPU job to a Google Cloud Compute Engine instance.
#
# Usage:
#   ./gcp/submit_job.sh \
#     --config config/llama30b-exp.yaml \
#     --bucket gs://my-bucket/fast-mia-results \
#     [--project my-gcp-project] \
#     [--zone us-central1-a] \
#     [--machine-type a2-highgpu-1g] \
#     [--accelerator-type nvidia-tesla-a100] \
#     [--accelerator-count 1] \
#     [--boot-disk-size 200GB] \
#     [--instance-name fast-mia-job] \
#     [--extra-args "--seed 42 --detailed-report"] \
#     [--keep-instance]
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Sufficient GPU quota in the target zone
#   - GCS bucket created
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────
CONFIG=""
GCS_BUCKET=""
PROJECT=""
ZONE="us-central1-a"
MACHINE_TYPE="a2-highgpu-1g"
ACCELERATOR_TYPE="nvidia-tesla-a100"
ACCELERATOR_COUNT=1
BOOT_DISK_SIZE="200GB"
INSTANCE_NAME="fast-mia-job-$(date +%Y%m%d-%H%M%S)"
EXTRA_ARGS=""
KEEP_INSTANCE=false

# ── Parse arguments ───────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)           CONFIG="$2"; shift 2;;
        --bucket)           GCS_BUCKET="$2"; shift 2;;
        --project)          PROJECT="$2"; shift 2;;
        --zone)             ZONE="$2"; shift 2;;
        --machine-type)     MACHINE_TYPE="$2"; shift 2;;
        --accelerator-type) ACCELERATOR_TYPE="$2"; shift 2;;
        --accelerator-count) ACCELERATOR_COUNT="$2"; shift 2;;
        --boot-disk-size)   BOOT_DISK_SIZE="$2"; shift 2;;
        --instance-name)    INSTANCE_NAME="$2"; shift 2;;
        --extra-args)       EXTRA_ARGS="$2"; shift 2;;
        --keep-instance)    KEEP_INSTANCE=true; shift;;
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
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG" >&2
    exit 1
fi

PROJECT_FLAG=""
if [[ -n "$PROJECT" ]]; then
    PROJECT_FLAG="--project=$PROJECT"
fi

# Resolve project root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================"
echo " fast-mia GCP Job Submission"
echo "============================================"
echo " Instance:    $INSTANCE_NAME"
echo " Zone:        $ZONE"
echo " Machine:     $MACHINE_TYPE"
echo " GPU:         ${ACCELERATOR_COUNT}x ${ACCELERATOR_TYPE}"
echo " Config:      $CONFIG"
echo " GCS Bucket:  $GCS_BUCKET"
echo " Extra args:  ${EXTRA_ARGS:-none}"
echo "============================================"

# ── Step 1: Create instance ──────────────────────────────
echo ""
echo "[1/5] Creating GCE instance..."
gcloud compute instances create "$INSTANCE_NAME" \
    $PROJECT_FLAG \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=${ACCELERATOR_TYPE},count=${ACCELERATOR_COUNT}" \
    --maintenance-policy=TERMINATE \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --image-family=common-cu128-ubuntu-2204-nvidia-570 \
    --image-project=deeplearning-platform-release \
    --scopes=storage-full \
    --metadata="install-nvidia-driver=true"

# Wait for SSH to be available
echo "Waiting for instance to be ready..."
for i in $(seq 1 60); do
    if gcloud compute ssh "$INSTANCE_NAME" \
        --zone="$ZONE" $PROJECT_FLAG \
        --command="echo ready" &>/dev/null; then
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Instance not reachable via SSH after 10 minutes."
        exit 1
    fi
    sleep 10
done
echo "Instance is ready."

# ── Step 2: Transfer project files ──────────────────────
echo ""
echo "[2/5] Transferring project files..."
# Create a tarball excluding unnecessary files
TMPTAR=$(mktemp /tmp/fast-mia-XXXXXX.tar.gz)
tar -czf "$TMPTAR" \
    -C "$PROJECT_ROOT" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.venv' \
    --exclude='results' \
    --exclude='.ruff_cache' \
    .

gcloud compute scp "$TMPTAR" "$INSTANCE_NAME":~/fast-mia.tar.gz \
    --zone="$ZONE" $PROJECT_FLAG

# Extract on instance
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" $PROJECT_FLAG \
    --command="mkdir -p ~/fast-mia && tar -xzf ~/fast-mia.tar.gz -C ~/fast-mia"

rm -f "$TMPTAR"

# ── Step 3: Setup environment ────────────────────────────
echo ""
echo "[3/5] Setting up environment on instance..."
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" $PROJECT_FLAG \
    --command="bash ~/fast-mia/gcp/setup_instance.sh"

# ── Step 4: Run the job ──────────────────────────────────
echo ""
echo "[4/5] Running fast-mia job..."
# shellcheck disable=SC2086
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" $PROJECT_FLAG \
    --command="cd ~/fast-mia && export PATH=\$HOME/.local/bin:\$PATH && uv run --with 'vllm==0.15.1' python main.py --config $CONFIG $EXTRA_ARGS"

# ── Step 5: Upload results to GCS ────────────────────────
echo ""
echo "[5/5] Uploading results to GCS..."
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" $PROJECT_FLAG \
    --command="gsutil -m cp -r ~/fast-mia/results/* $GCS_BUCKET/"

echo ""
echo "============================================"
echo " Results uploaded to: $GCS_BUCKET"
echo "============================================"

# ── Cleanup ──────────────────────────────────────────────
if [[ "$KEEP_INSTANCE" == "false" ]]; then
    echo ""
    echo "Deleting instance $INSTANCE_NAME..."
    gcloud compute instances delete "$INSTANCE_NAME" \
        --zone="$ZONE" $PROJECT_FLAG \
        --quiet
    echo "Instance deleted."
else
    echo ""
    echo "Instance kept alive: $INSTANCE_NAME (zone: $ZONE)"
    echo "Don't forget to delete it when done:"
    echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE $PROJECT_FLAG --quiet"
fi

echo ""
echo "Done!"
