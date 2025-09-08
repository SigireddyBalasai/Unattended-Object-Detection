#!/bin/bash
#SBATCH --job-name=triton-inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=triton_%j.out
#SBATCH --error=triton_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "Job started on $(hostname) at $(date)"
echo "Working dir: $(pwd)"

# Try to load rootless docker if available
if command -v module &>/dev/null; then
  module load rootless-docker || true
fi

# Start rootless docker if helper exists
if command -v start_rootless_docker.sh &>/dev/null; then
  echo "Starting rootless docker..."
  start_rootless_docker.sh --quiet || true
fi

MODEL_REPO="$(pwd)/model_repository"
echo "Using model repo: ${MODEL_REPO}"

# Pull Triton container image (non-fatal if fails)
docker pull nvcr.io/nvidia/tritonserver:24.08-py3 || true

echo "Launching Triton container"
docker run --rm \
  --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "${MODEL_REPO}:/models:ro" \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver --model-repository=/models --log-verbose=1

echo "Triton container exited"
