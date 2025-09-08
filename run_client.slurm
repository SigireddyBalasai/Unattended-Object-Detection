#!/bin/bash
#SBATCH --job-name=triton-client
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=client_%j.out
#SBATCH --error=client_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "Job started on $(hostname) at $(date)"
echo "Working dir: $(pwd)"

# Wait for Triton server to be ready
echo "Waiting for Triton server to be ready..."
timeout=300  # 5 minutes timeout
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

# Check model status
echo "Checking model status..."
curl -s http://localhost:8000/v2/models/rtdetr

echo ""
echo "Running inference on video1.avi..."

# Run inference using uv
uv run python triton_client.py \
    --video ABODA/video1.avi \
    --model-name rtdetr \
    --url localhost:8000 \
    --output output_video1.mp4 \
    --conf-threshold 0.3

echo "Inference completed!"
echo "Job finished at $(date)"
