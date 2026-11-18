#!/bin/bash
# Setup script to run on a Google Cloud TPU VM
# Prepares uv, persistent XLA compilation cache, and fast-mia Python dependencies
set -euo pipefail

echo "=== Setting up fast-mia Cloud TPU environment ==="

# Verify TPU devices are visible on the host
echo "Checking TPU device nodes..."
if ls /dev/accel* /dev/vfio/* &>/dev/null; then
    echo "TPU device nodes detected:"
    ls -1 /dev/accel* /dev/vfio/* 2>/dev/null || true
else
    echo "WARNING: No /dev/accel* or /dev/vfio/* device nodes detected yet."
fi

# Create persistent XLA compilation cache directory so repeated evaluations
# skip redundant XLA graph compilation.
mkdir -p "$HOME/.cache/xla_compile_cache"

# Install uv if not already installed
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

cd ~/fast-mia

# Create venv and install core dependencies
echo "Installing Python dependencies..."
uv sync

echo "=== Cloud TPU setup complete ==="
