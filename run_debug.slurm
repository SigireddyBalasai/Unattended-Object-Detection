#!/bin/bash
#SBATCH --job-name=debug-inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=debug_%j.out
#SBATCH --error=debug_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "Debug job started on $(hostname) at $(date)"
echo "Working dir: $(pwd)"

# Wait for Triton server to be ready
echo "Waiting for Triton server to be ready..."
timeout=300
elapsed=0
while [ $elapsed -lt $timeout ]; do
    if curl -s http://localhost:8000/v2/health/ready >/dev/null 2>&1; then
        echo "Triton server is ready!"
        break
    fi
    echo "Waiting for server... (${elapsed}s elapsed)"
    sleep 5
    elapsed=$((elapsed + 5))
done

if [ $elapsed -ge $timeout ]; then
    echo "Error: Triton server not ready after ${timeout} seconds"
    exit 1
fi

echo ""
echo "Running debug inference..."

# Run debug script
uv run python debug_inference.py

echo "Debug job finished at $(date)"
