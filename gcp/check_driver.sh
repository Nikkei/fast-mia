#!/bin/bash
# Check both new and reused VMs before running the GPU workload.
# The PyPI torch dependency of vLLM 0.29.0 uses CUDA 13 (driver >= 580).
set -euo pipefail

if ! drivers=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader); then
    echo "ERROR: Unable to query NVIDIA drivers with nvidia-smi." >&2
    exit 1
fi

if [[ -z "$drivers" ]]; then
    echo "ERROR: No NVIDIA GPUs found." >&2
    exit 1
fi

while IFS= read -r driver; do
    driver="${driver//[[:space:]]/}"
    major="${driver%%.*}"
    if [[ ! "$major" =~ ^[0-9]+$ ]] || (( 10#$major < 580 )); then
        echo "ERROR: vLLM 0.29.0's CUDA 13 dependencies require NVIDIA driver 580 or newer; found '$driver'." >&2
        echo "Upgrade the VM driver or use a new instance with the default image. The instance remains running; stop it if no longer needed." >&2
        exit 1
    fi
done <<< "$drivers"

echo "NVIDIA driver check passed."
