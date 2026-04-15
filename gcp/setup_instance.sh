#!/bin/bash
# Setup script to run on a GCE GPU instance
# This script installs dependencies and prepares the environment for fast-mia
set -euo pipefail

echo "=== Setting up fast-mia environment ==="

# Wait for NVIDIA drivers (Deep Learning VM images may need a moment)
echo "Waiting for NVIDIA drivers..."
for i in $(seq 1 30); do
    if nvidia-smi &>/dev/null; then
        echo "NVIDIA drivers ready."
        nvidia-smi
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: NVIDIA drivers not available after 5 minutes."
        exit 1
    fi
    sleep 10
done

# Install uv
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd ~/fast-mia

# Create venv and install dependencies
echo "Installing Python dependencies..."
uv sync

echo "=== Setup complete ==="
