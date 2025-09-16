#!/bin/bash
#SBATCH --job-name=flask-server
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=flask_%j.out
#SBATCH --error=flask_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Source cluster configuration
source ./cluster_config.sh

echo "Job started on $(hostname) at $(date)"
echo "Working dir: $(pwd)"

# Get Triton server node
TRITON_NODE=$(get_triton_node)
if [[ -z "$TRITON_NODE" ]]; then
    echo "ERROR: Could not detect Triton server node"
    echo "Make sure Triton server is running first"
    exit 1
fi

echo "Detected Triton server on node: $TRITON_NODE"

# Wait for Triton server to be ready
echo "Waiting for Triton server to be ready..."
if ! wait_for_triton 300; then
    echo "Error: Triton server not ready after 300 seconds"
    exit 1
fi

echo "Starting Flask WebSocket server..."

# Run Flask server using uv
uv run python main.py

echo "Flask server stopped"
