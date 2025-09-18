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

# Source cluster configuration
source ./cluster_config.sh

echo "Job started on $(hostname) at $(date)"
echo "Working dir: $(pwd)"

# Write cluster state with current node
write_cluster_state "${SLURM_JOB_ID}" "$(hostname)" "triton-server"

# Generate random ports for Triton server
echo "Generating random ports for Triton server..."
read HTTP_PORT GRPC_PORT METRICS_PORT <<< $(generate_triton_ports)
echo "Generated ports: HTTP=${HTTP_PORT}, gRPC=${GRPC_PORT}, Metrics=${METRICS_PORT}"

# Update cluster state with port information
write_cluster_state_with_ports "${SLURM_JOB_ID}" "$(hostname)" "triton-server" "$HTTP_PORT" "$GRPC_PORT" "$METRICS_PORT"

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

# Launch Triton container with conversion step
echo "Converting ONNX to TensorRT..."
docker run --rm \
  --gpus all \
  -v "${MODEL_REPO}:/models" \
  -v "$(pwd)/convert_to_tensorrt.sh:/convert_to_tensorrt.sh" \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  bash -c "chmod +x /convert_to_tensorrt.sh && /convert_to_tensorrt.sh"

echo "Starting Triton server..."

# Launch Triton container
docker run --rm \
  --gpus all \
  -p ${HTTP_PORT}:8000 -p ${GRPC_PORT}:8001 -p ${METRICS_PORT}:8002 \
  -v "${MODEL_REPO}:/models:ro" \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver --model-repository=/models --log-verbose=1

echo "Triton container exited"

# Clean up cluster state on exit
cleanup_cluster_state
