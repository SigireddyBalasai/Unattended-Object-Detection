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

# Get Triton server ports from cluster config
TRITON_PORTS=$(get_triton_ports)
if [[ -z "$TRITON_PORTS" ]]; then
    echo "WARNING: Could not retrieve Triton server ports, using defaults"
    HTTP_PORT=8000
    GRPC_PORT=8001
    METRICS_PORT=8002
else
    read HTTP_PORT GRPC_PORT METRICS_PORT <<< "$TRITON_PORTS"
    echo "Retrieved Triton server ports: HTTP=${HTTP_PORT}, gRPC=${GRPC_PORT}, Metrics=${METRICS_PORT}"
fi

# Wait for Triton server to be ready
echo "Waiting for Triton server to be ready..."
if ! wait_for_triton "$HTTP_PORT" 300; then
    echo "Error: Triton server not ready after 300 seconds"
    exit 1
fi

# Check model status
echo "Checking model status..."
curl -s http://${TRITON_NODE}:${HTTP_PORT}/v2/models/rtdetr

echo ""
echo "Running inference on video1.avi..."

# Run inference using uv
uv run python triton_client.py \
    --video ABODA/video1.avi \
    --model-name rtdetr \
    --url ${TRITON_NODE}:${HTTP_PORT} \
    --output output_video1.mp4 \
    --conf-threshold 0.3

echo "Inference completed!"
echo "Job finished at $(date)"
