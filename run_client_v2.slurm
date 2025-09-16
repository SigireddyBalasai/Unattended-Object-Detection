#!/bin/bash
#SBATCH --job-name=triton-client-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=client_v2_%j.out
#SBATCH --error=client_v2_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Source cluster configuration
source ./cluster_config.sh

# Enhanced logging function
log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Record job start time
job_start_time=$(date +%s)

log_with_timestamp "=================================="
log_with_timestamp "TRITON CLIENT V2 JOB STARTING"
log_with_timestamp "=================================="
log_with_timestamp "Job ID: ${SLURM_JOB_ID:-N/A}"
log_with_timestamp "Job Name: ${SLURM_JOB_NAME:-N/A}"
log_with_timestamp "Node: $(hostname)"
log_with_timestamp "Working directory: $(pwd)"
log_with_timestamp "CPUs allocated: ${SLURM_CPUS_PER_TASK:-N/A}"
log_with_timestamp "GPUs allocated: ${SLURM_GPUS:-N/A}"
log_with_timestamp "Memory allocated: ${SLURM_MEM_PER_NODE:-N/A}MB"
log_with_timestamp "Time limit: ${SLURM_TIME_LIMIT:-N/A}"
log_with_timestamp "User: ${USER:-N/A}"

# Check system resources
log_with_timestamp "System information:"
log_with_timestamp "  - CPU info: $(nproc) cores available"
log_with_timestamp "  - Memory: $(free -h | grep '^Mem:' | awk '{print $2}') total"
log_with_timestamp "  - Disk space: $(df -h . | tail -1 | awk '{print $4}') available"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    log_with_timestamp "GPU information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | while read line; do
        log_with_timestamp "  - GPU: $line"
    done
else
    log_with_timestamp "WARNING: nvidia-smi not available"
fi

# Check required files
log_with_timestamp "Checking required files..."
if [[ -f "ABODA/video2.avi" ]]; then
    video_size=$(du -h "ABODA/video2.avi" | cut -f1)
    log_with_timestamp "  ✓ Video file found: ABODA/video2.avi (${video_size})"
else
    log_with_timestamp "  ✗ ERROR: Video file ABODA/video2.avi not found"
    exit 1
fi

if [[ -f "triton_client.py" ]]; then
    log_with_timestamp "  ✓ Triton client script found"
else
    log_with_timestamp "  ✗ ERROR: triton_client.py not found"
    exit 1
fi

if [[ -f "model3.py" ]]; then
    log_with_timestamp "  ✓ Model3 script found"
else
    log_with_timestamp "  ✗ ERROR: model3.py not found"
    exit 1
fi

# Check Python environment
log_with_timestamp "Python environment check..."
if command -v uv &> /dev/null; then
    log_with_timestamp "  ✓ UV package manager found: $(uv --version)"
else
    log_with_timestamp "  ✗ ERROR: UV package manager not found"
    exit 1
fi

# Get Triton server node from cluster config
log_with_timestamp "=================================="
log_with_timestamp "DETECTING TRITON SERVER"
log_with_timestamp "=================================="

TRITON_NODE=$(get_triton_node)
if [[ -z "$TRITON_NODE" ]]; then
    log_with_timestamp "✗ ERROR: Could not detect Triton server node"
    log_with_timestamp "Make sure Triton server is running first"
    exit 1
fi

log_with_timestamp "✓ Detected Triton server on node: $TRITON_NODE"

# Wait for Triton server to be ready
log_with_timestamp "=================================="
log_with_timestamp "WAITING FOR TRITON SERVER"
log_with_timestamp "=================================="
log_with_timestamp "Checking Triton server availability at ${TRITON_NODE}:8000..."

if ! wait_for_triton 300; then
    log_with_timestamp "✗ ERROR: Triton server not ready after 300 seconds"
    log_with_timestamp "Troubleshooting information:"
    log_with_timestamp "  - Check if Triton server is running: squeue -u $USER"
    log_with_timestamp "  - Check Triton logs for errors"
    exit 1
fi

# Additional health checks
log_with_timestamp "Performing additional health checks..."

# Check server status
if server_status=$(curl -s http://${TRITON_NODE}:8000/v2/health/live 2>/dev/null); then
    log_with_timestamp "  ✓ Server live check: OK"
else
    log_with_timestamp "  ⚠ Server live check: FAILED"
fi

# Check server metadata
if server_metadata=$(curl -s http://${TRITON_NODE}:8000/v2 2>/dev/null); then
    log_with_timestamp "  ✓ Server metadata accessible"
else
    log_with_timestamp "  ⚠ Server metadata: FAILED"
fi

# Check model status
log_with_timestamp "=================================="
log_with_timestamp "MODEL STATUS CHECK"
log_with_timestamp "=================================="
log_with_timestamp "Checking model 'rtdetr' status..."

if model_status=$(curl -s http://${TRITON_NODE}:8000/v2/models/rtdetr 2>/dev/null); then
    log_with_timestamp "Model status response received:"
    echo "$model_status" | jq '.' 2>/dev/null || echo "$model_status"
    
    # Check if model is ready
    if echo "$model_status" | grep -q "READY"; then
        log_with_timestamp "✓ Model 'rtdetr' is READY"
    else
        log_with_timestamp "⚠ WARNING: Model 'rtdetr' may not be ready"
    fi
else
    log_with_timestamp "✗ ERROR: Could not retrieve model status"
    exit 1
fi

# Get model metadata
log_with_timestamp "Retrieving model metadata..."
if model_metadata=$(curl -s http://${TRITON_NODE}:8000/v2/models/rtdetr/config 2>/dev/null); then
    log_with_timestamp "Model configuration:"
    echo "$model_metadata" | jq '.' 2>/dev/null || echo "$model_metadata"
else
    log_with_timestamp "⚠ WARNING: Could not retrieve model metadata"
fi

log_with_timestamp "=================================="
log_with_timestamp "STARTING VIDEO INFERENCE"
log_with_timestamp "=================================="
log_with_timestamp "Running inference on video2.avi with confidence threshold 0.1..."

# Log inference parameters
log_with_timestamp "Inference parameters:"
log_with_timestamp "  - Input video: ABODA/video2.avi"
log_with_timestamp "  - Model name: rtdetr"
log_with_timestamp "  - Server URL: ${TRITON_NODE}:8000"
log_with_timestamp "  - Output video: output_video2.mp4"
log_with_timestamp "  - Confidence threshold: 0.1"

# Get video information before processing
if command -v ffprobe &> /dev/null; then
    log_with_timestamp "Video file information:"
    ffprobe -v quiet -print_format json -show_format -show_streams "ABODA/video2.avi" 2>/dev/null | jq -r '
        .streams[] | select(.codec_type=="video") | 
        "  - Resolution: \(.width)x\(.height)",
        "  - Frame rate: \(.r_frame_rate)",
        "  - Duration: \(.duration // "N/A") seconds",
        "  - Codec: \(.codec_name)"
    ' 2>/dev/null || log_with_timestamp "  - Could not extract detailed video info"
fi

# Record start time for inference
inference_start_time=$(date +%s)
log_with_timestamp "Inference started at: $(date)"

# Run inference using uv with enhanced error handling
log_with_timestamp "Executing triton_client.py..."
if uv run python triton_client.py \
    --video ABODA/video2.avi \
    --model-name rtdetr \
    --url ${TRITON_NODE}:8000 \
    --output output_video2.mp4 \
    --conf-threshold 0.1; then
    
    inference_end_time=$(date +%s)
    inference_duration=$((inference_end_time - inference_start_time))
    log_with_timestamp "✓ Inference completed successfully in ${inference_duration} seconds"
    
    # Check output file
    if [[ -f "output_video2.mp4" ]]; then
        output_size=$(du -h "output_video2.mp4" | cut -f1)
        log_with_timestamp "✓ Output video created: output_video2.mp4 (${output_size})"
    else
        log_with_timestamp "⚠ WARNING: Output video file not found"
    fi
else
    log_with_timestamp "✗ ERROR: Inference failed"
    exit 1
fi

log_with_timestamp "=================================="
log_with_timestamp "RUNNING MODEL3 POST-PROCESSING"
log_with_timestamp "=================================="
log_with_timestamp "Executing model3.py for post-processing..."

# Record start time for model3
model3_start_time=$(date +%s)
log_with_timestamp "Model3 processing started at: $(date)"

if uv run python model3.py --triton-url ${TRITON_NODE}:8000; then
    model3_end_time=$(date +%s)
    model3_duration=$((model3_end_time - model3_start_time))
    log_with_timestamp "✓ Model3 processing completed successfully in ${model3_duration} seconds"
else
    log_with_timestamp "✗ ERROR: Model3 processing failed"
    exit 1
fi

log_with_timestamp "=================================="
log_with_timestamp "JOB COMPLETION SUMMARY"
log_with_timestamp "=================================="

# Calculate total job duration
job_end_time=$(date +%s)
job_start_time=${job_start_time:-$job_end_time}
total_duration=$((job_end_time - job_start_time))

log_with_timestamp "Job completed successfully!"
log_with_timestamp "Total job duration: ${total_duration} seconds"
log_with_timestamp "Inference duration: ${inference_duration:-N/A} seconds"
log_with_timestamp "Model3 duration: ${model3_duration:-N/A} seconds"
log_with_timestamp "Final output files:"

# List output files with sizes
for file in output_video2.mp4 *.out *.err; do
    if [[ -f "$file" ]]; then
        file_size=$(du -h "$file" | cut -f1)
        log_with_timestamp "  - $file (${file_size})"
    fi
done

log_with_timestamp "Job finished at: $(date)"
log_with_timestamp "=================================="
