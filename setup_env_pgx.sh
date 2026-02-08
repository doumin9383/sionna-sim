#!/bin/bash
# Script to setup python environment on PGX

echo "=== Setting up PGX Environment ==="

# 1. Install uv on PGX
echo "1. Installing uv on PGX..."
ssh pgx-data "curl -LsSf https://astral.sh/uv/install.sh | sh"

# 2. Sync environment (ARM)
echo "2. Creating .venv-arm on shared storage..."
# Note: We use the shared path /mnt/nas_data/sionna-sim
ssh pgx-data "cd /mnt/nas_data/sionna-sim && UV_PROJECT_ENVIRONMENT=.venv-arm /home/sh-fukue/.local/bin/uv sync"

echo "=== Environment Setup Complete ==="
echo "You can now use './run_pgx.sh' (to be created) to run scripts."
