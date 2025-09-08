#!/bin/bash
#SBATCH --job-name=triton-master
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=master_%j.out
#SBATCH --error=master_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "=== Triton Inference Server Master Job ==="
echo "Job started on $(hostname) at $(date)"
echo "Working dir: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID}"

# Function to cleanup background processes
cleanup() {
    echo "Cleaning up background processes..."
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup EXIT

# Try to load rootless docker if available
if command -v module &>/dev/null; then
    echo "Loading rootless-docker module..."
    module load rootless-docker || true
fi

# Start rootless docker if helper exists
if command -v start_rootless_docker.sh &>/dev/null; then
    echo "Starting rootless docker..."
    start_rootless_docker.sh --quiet || true
fi

MODEL_REPO="$(pwd)/model_repository"
echo "Using model repo: ${MODEL_REPO}"

# Check if model repository exists
if [[ ! -d "${MODEL_REPO}" ]]; then
    echo "Error: Model repository not found at ${MODEL_REPO}"
    exit 1
fi

# Pull Triton container image (non-fatal if fails)
echo "Pulling Triton container image..."
docker pull nvcr.io/nvidia/tritonserver:24.08-py3 || echo "Warning: Failed to pull latest image, using cached version"

# Start Triton server in background
echo "Starting Triton server..."
docker run --rm \
    --gpus all \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v "${MODEL_REPO}:/models:ro" \
    nvcr.io/nvidia/tritonserver:24.08-py3 \
    tritonserver --model-repository=/models --log-verbose=1 &

TRITON_PID=$!
echo "Triton server started with PID: ${TRITON_PID}"

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
    sleep 10
    elapsed=$((elapsed + 10))
done

if [ $elapsed -ge $timeout ]; then
    echo "Error: Triton server not ready after ${timeout} seconds"
    exit 1
fi

# Check model status
echo "Checking model status..."
if curl -s http://localhost:8000/v2/models/rtdetr | grep -q "READY"; then
    echo "Model rtdetr is ready!"
else
    echo "Warning: Model rtdetr may not be ready"
fi

echo ""
echo "=== Running inference tests ==="

# Test with a sample video
if [[ -f "ABODA/video1.avi" ]]; then
    echo "Running inference on video1.avi..."
    uv run python triton_client.py \
        --video ABODA/video1.avi \
        --model-name rtdetr \
        --url localhost:8000 \
        --output output_video1.mp4 \
        --conf-threshold 0.3
    echo "Inference completed!"
else
    echo "Warning: Test video ABODA/video1.avi not found"
fi

# Start Flask WebSocket server in background for further testing
echo ""
echo "=== Starting Flask WebSocket server ==="
uv run python main.py &
FLASK_PID=$!
echo "Flask server started with PID: ${FLASK_PID}"

# Give Flask server time to start
sleep 5

echo ""
echo "=== System is ready ==="
echo "Triton server: http://localhost:8000"
echo "Flask WebSocket server: ws://localhost:6969/ws"
echo "Flask HTTP endpoint: http://localhost:6969"
echo ""
echo "To test WebSocket connection, you can run:"
echo "uv run python test.py"
echo ""
echo "Servers will run until job time limit or manual termination"
echo "Job will keep running... (Ctrl+C to stop)"

# Keep the job alive and monitor the services
while true; do
    # Check if Triton is still running
    if ! kill -0 $TRITON_PID 2>/dev/null; then
        echo "Error: Triton server has stopped unexpectedly"
        exit 1
    fi
    
    # Check if Flask is still running
    if ! kill -0 $FLASK_PID 2>/dev/null; then
        echo "Warning: Flask server has stopped, restarting..."
        uv run python main.py &
        FLASK_PID=$!
    fi
    
    # Check server health
    if ! curl -s http://localhost:8000/v2/health/ready >/dev/null 2>&1; then
        echo "Warning: Triton server health check failed"
    fi
    
    sleep 30
done
