#!/bin/bash

# Automated Triton Server and Video Inference Script
# This script checks for existing Triton server, starts it if needed,
# lists available video files, allows user selection, runs inference,
# and saves output with unique naming.

set -euo pipefail

# Default configuration
MODEL_NAME="${1:-rtdetr_tensorrt}"
CONF_THRESHOLD="${2:-0.5}"

# Script directory and paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source cluster configuration and manage jobs functionality
source ./cluster_config.sh

MANAGE_JOBS_SCRIPT="$SCRIPT_DIR/manage_jobs.sh"
OUTPUT_DIR="$SCRIPT_DIR/outputs"
MODEL3_SCRIPT="$SCRIPT_DIR/model3.py"

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
RESET='\033[0m'

# Function to check required commands
check_dependencies() {
    local missing=()
    
    for cmd in curl sed tr expr stat find; do
        if ! command -v "$cmd" &> /dev/null; then
            missing+=("$cmd")
        fi
    done
    
    if [ ${#missing[@]} -ne 0 ]; then
        print_colored "$RED" "❌ Error: Missing required commands: ${missing[*]}"
        print_colored "$YELLOW" "Please install the missing commands and try again."
        exit 1
    fi
    
    # Check for mapfile (bash 4+)
    if ! type mapfile &> /dev/null; then
        print_colored "$RED" "❌ Error: mapfile command not available (requires bash 4.0+)"
        print_colored "$YELLOW" "Please upgrade bash or use a compatible shell."
        exit 1
    fi
}

# Function to test Triton server health using cluster config
test_triton_health() {
    if read_cluster_state; then
        local triton_node="${TRITON_NODE:-}"
        if [[ -n "$triton_node" ]]; then
            curl -s "http://$triton_node:8000/v2/health/ready" >/dev/null 2>&1
            return $?
        fi
    fi
    return 1
}

# Function to test if specific model is ready
test_model_ready() {
    local model_name=$1
    if read_cluster_state; then
        local triton_node="${TRITON_NODE:-}"
        if [[ -n "$triton_node" ]]; then
            curl -s "http://$triton_node:8000/v2/models/$model_name/ready" >/dev/null 2>&1
            return $?
        fi
    fi
    return 1
}

# Function to start Triton server using manage_jobs.sh
start_triton_server() {
    print_colored "$CYAN" "🚀 Starting Triton Inference Server using manage_jobs.sh..."
    
    # Check if manage_jobs.sh exists
    if [ ! -f "$MANAGE_JOBS_SCRIPT" ]; then
        print_colored "$RED" "❌ Error: manage_jobs.sh not found!"
        return 1
    fi
    
    # Check if we're in a SLURM environment
    if ! command -v sbatch &> /dev/null; then
        print_colored "$RED" "❌ Error: SLURM not available. This script requires SLURM environment."
        print_colored "$YELLOW" "Please run this script on a SLURM cluster or use manual Triton server startup."
        return 1
    fi
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    print_colored "$YELLOW" "⏳ Starting Triton server via SLURM..."
    
    # Start Triton server using manage_jobs.sh
    if ! "$MANAGE_JOBS_SCRIPT" start-triton; then
        print_colored "$RED" "❌ Failed to start Triton server"
        return 1
    fi
    
    # Wait for server to be ready
    print_colored "$YELLOW" "⏳ Waiting for Triton server to become ready..."
    if wait_for_triton 300; then
        print_colored "$GREEN" "✅ Triton server is running and healthy!"
        return 0
    else
        print_colored "$RED" "❌ Triton server failed to become ready"
        return 1
    fi
}

# Function to get server URL from cluster state
get_server_url() {
    if read_cluster_state; then
        local triton_node="${TRITON_NODE:-}"
        if [[ -n "$triton_node" ]]; then
            echo "$triton_node:8000"
            return 0
        fi
    fi
    echo "localhost:8000"  # fallback
    return 1
}

# Function to find video files
find_video_files() {
    print_colored "$CYAN" "🔍 Searching for video files..." >&2
    
    # Find video files excluding outputs directory and files starting with "output_"
    find "$SCRIPT_DIR" -type f \( \
        -name "*.mp4" -o \
        -name "*.avi" -o \
        -name "*.mov" -o \
        -name "*.mkv" -o \
        -name "*.wmv" -o \
        -name "*.flv" -o \
        -name "*.webm" -o \
        -name "*.m4v" \
    \) \
    ! -path "*/outputs/*" \
    ! -name "output_*" \
    | sort
}

# Function to show video selection menu
show_video_selection() {
    local video_files=("$@")
    
    print_colored "$MAGENTA" "\n📹 Available Video Files:"
    print_colored "$MAGENTA" "$(printf '=%.0s' {1..60})"
    
    for i in "${!video_files[@]}"; do
        local file="${video_files[$i]}"
        local filename=$(basename "$file")
        local relative_path="${file#$SCRIPT_DIR/}"
        local size_bytes=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "0")
        local size_kb=$(expr $size_bytes / 1024)
        
        printf "${YELLOW}%d.${RESET} ${WHITE}%s${RESET} ${BLUE}(%s, %dKB)${RESET}\n" \
            $((i + 1)) "$filename" "$relative_path" "$size_kb"
    done
    
    print_colored "$MAGENTA" "$(printf '=%.0s' {1..60})"
}

# Function to get user video selection
get_user_selection() {
    local video_files=("$@")
    local video_count=${#video_files[@]}
    
    while true; do
        print_colored "$CYAN" "Enter video number (1-$video_count) or 'q' to quit: "
        if ! read -r -t 60 selection; then
            print_colored "$RED" "❌ Timeout: No input received within 60 seconds"
            return 1
        fi
        
        # Trim whitespace and remove any ANSI escape sequences
        selection=$(printf "%s" "$selection" | sed 's/\x1b\[[0-9;]*m//g' | tr -d '[:space:]')
        
        # Additional validation to ensure selection contains only digits or q/Q
        if [[ -n "$selection" ]]; then
            selection=$(printf "%s" "$selection" | tr -cd '0-9qQ')
        fi
        
        if [[ "$selection" == "q" || "$selection" == "Q" ]]; then
            return 1
        fi
        
        # Check if selection is a valid number
        if [[ "$selection" =~ ^[0-9]+$ ]]; then
            if [ "$selection" -ge 1 ] && [ "$selection" -le "$video_count" ]; then
                # Convert to zero-based index
                local index=$(expr "$selection" - 1)
                echo "$index"
                return 0
            else
                print_colored "$RED" "❌ Invalid selection. Please enter a number between 1 and $video_count"
            fi
        else
            print_colored "$RED" "❌ Invalid input. Please enter a number between 1 and $video_count or 'q' to quit"
        fi
    done
}

# Function to generate unique output path
generate_output_path() {
    local input_file=$1
    local filename=$(basename "$input_file")
    local basename="${filename%.*}"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local output_filename="${basename}_inference_${timestamp}.mp4"
    local output_path="$OUTPUT_DIR/$output_filename"
    
    # Ensure unique filename
    local counter=1
    while [ -f "$output_path" ]; do
        output_filename="${basename}_inference_${timestamp}_${counter}.mp4"
        output_path="$OUTPUT_DIR/$output_filename"
        counter=$(expr $counter + 1)
    done
    
    echo "$output_path"
}

# Function to run inference
run_inference() {
    local video_path=$1
    local output_path=$2
    local model_name=$3
    local conf_threshold=$4
    
    # Get server URL from cluster state
    local server_url=$(get_server_url)
    
    print_colored "$CYAN" "\n🎯 Starting unattended object detection inference..."
    print_colored "$BLUE" "📹 Input: $video_path"
    print_colored "$BLUE" "💾 Output: $output_path"
    print_colored "$BLUE" "🤖 Model: $model_name"
    print_colored "$BLUE" "🌐 Server: $server_url"
    print_colored "$BLUE" "🎚️ Confidence Threshold: $conf_threshold"
    print_colored "$BLUE" "🔍 Features: Unattended object detection with tracking and alerts"
    
    # Check if model repository exists
    local model_repo="$SCRIPT_DIR/model_repository/$MODEL_NAME"
    if [ ! -d "$model_repo" ]; then
        print_colored "$RED" "❌ Error: Model repository not found at $model_repo"
        print_colored "$YELLOW" "Please ensure the model is properly installed."
        return 1
    fi
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        print_colored "$RED" "❌ Error: Python not found!"
        print_colored "$YELLOW" "Please install Python 3 and try again."
        return 1
    fi
    
    # Use python3 if available, otherwise python
    local python_cmd="python3"
    if ! command -v python3 &> /dev/null; then
        python_cmd="python"
        print_colored "$YELLOW" "⚠️ Using 'python' instead of 'python3'. Make sure it's Python 3."
    fi
    
    # Run the inference using model3.py
    print_colored "$YELLOW" "🔄 Running unattended object detection... (This may take a while)"
    print_colored "$YELLOW" "📊 Processing will include: person detection, object tracking, and unattended alerts"
    
    if $python_cmd "$MODEL3_SCRIPT" \
        --input "$video_path" \
        --output "$output_path" \
        --triton-url "$server_url"; then
        print_colored "$GREEN" "✅ Unattended object detection completed successfully!"
        
        # Check if alerts log was created
        if [ -f "alerts.log" ]; then
            local alert_count=$(wc -l < "alerts.log" 2>/dev/null || echo "0")
            print_colored "$BLUE" "📋 Alerts generated: $alert_count (see alerts.log for details)"
        fi
        
        return 0
    else
        print_colored "$RED" "❌ Unattended object detection failed!"
        return 1
    fi
}

# Function to show summary
show_summary() {
    local input_video=$1
    local output_video=$2
    local success=$3
    
    print_colored "$MAGENTA" "\n$(printf '=%.0s' {1..60})"
    print_colored "$MAGENTA" "📊 UNATTENDED OBJECT DETECTION SUMMARY"
    print_colored "$MAGENTA" "$(printf '=%.0s' {1..60})"
    
    print_colored "$BLUE" "📹 Input Video: $input_video"
    
    if [ "$success" = "true" ]; then
        print_colored "$GREEN" "✅ Status: SUCCESS"
        print_colored "$GREEN" "💾 Output Video: $output_video"
        
        if [ -f "$output_video" ]; then
            local output_size=$(stat -f%z "$output_video" 2>/dev/null || stat -c%s "$output_video" 2>/dev/null || echo "0")
            local output_size_mb=$(expr $output_size / 1024 / 1024)
            print_colored "$BLUE" "📏 Output Size: ${output_size_mb} MB"
        fi
        
        # Show alerts summary if alerts.log exists
        if [ -f "alerts.log" ]; then
            local alert_count=$(wc -l < "alerts.log" 2>/dev/null || echo "0")
            if [ "$alert_count" -gt 0 ]; then
                print_colored "$YELLOW" "⚠️ Unattended Object Alerts: $alert_count"
                print_colored "$YELLOW" "📋 Alerts logged to: alerts.log"
                
                # Show last few alerts if any
                if [ "$alert_count" -le 3 ]; then
                    print_colored "$YELLOW" "📄 Recent alerts:"
                    tail -n "$alert_count" "alerts.log" | while read -r line; do
                        print_colored "$YELLOW" "   • $line"
                    done
                else
                    print_colored "$YELLOW" "📄 Last 3 alerts:"
                    tail -n 3 "alerts.log" | while read -r line; do
                        print_colored "$YELLOW" "   • $line"
                    done
                fi
            else
                print_colored "$GREEN" "✅ No unattended object alerts detected"
            fi
        else
            print_colored "$GREEN" "✅ No unattended object alerts detected"
        fi
        
        print_colored "$GREEN" "\n🎉 Your processed video is ready!"
        print_colored "$GREEN" "📁 Location: $output_video"
        print_colored "$BLUE" "🔍 Features processed: Person detection, object tracking, unattended alerts"
    else
        print_colored "$RED" "❌ Status: FAILED"
        print_colored "$YELLOW" "🔍 Check the error messages above for details"
    fi
    
    print_colored "$MAGENTA" "$(printf '=%.0s' {1..60})"
}

# Function to cleanup on exit
cleanup() {
    print_colored "$YELLOW" "\n🧹 Cleaning up..."
    # Don't kill Triton server on exit as it might be used by other processes
}

# Set trap for cleanup
trap cleanup EXIT

# Main function
main() {
    # Check dependencies first
    check_dependencies
    
    print_colored "$CYAN" "🚀 Unattended Object Detection - Inference Runner"
    print_colored "$CYAN" "$(printf '=%.0s' {1..60})"
    
    # Step 1: Check if Triton server is already running
    print_colored "$CYAN" "🔍 Checking Triton server status..."
    
    if test_triton_health; then
        if read_cluster_state; then
            print_colored "$GREEN" "✅ Triton server is already running at ${TRITON_NODE:-unknown}:8000"
            print_colored "$BLUE" "📊 Job ID: ${TRITON_JOB_ID:-unknown}, Type: ${JOB_TYPE:-unknown}"
        else
            print_colored "$GREEN" "✅ Triton server is running"
        fi
        
        # Check if our model is ready
        if test_model_ready "$MODEL_NAME"; then
            print_colored "$GREEN" "✅ Model '$MODEL_NAME' is ready"
        else
            print_colored "$YELLOW" "⚠️ Model '$MODEL_NAME' is not ready. Checking available models..."
            if read_cluster_state && [[ -n "${TRITON_NODE:-}" ]]; then
                if available_models=$(curl -s "http://${TRITON_NODE}:8000/v2/models" 2>/dev/null); then
                    print_colored "$BLUE" "Available models: $available_models"
                else
                    print_colored "$RED" "❌ Could not fetch available models"
                fi
            fi
        fi
    else
        print_colored "$YELLOW" "❌ Triton server not found. Starting new instance..."
        
        if ! start_triton_server; then
            print_colored "$RED" "❌ Failed to start Triton server. Exiting."
            exit 1
        fi
        
        # Verify model is ready
        if ! test_model_ready "$MODEL_NAME"; then
            print_colored "$RED" "❌ Model '$MODEL_NAME' is not ready after server start"
            
            # Show cluster info for debugging
            if [ -f "$MANAGE_JOBS_SCRIPT" ]; then
                print_colored "$YELLOW" "🔍 Checking cluster status..."
                "$MANAGE_JOBS_SCRIPT" cluster-info
            fi
            exit 1
        fi
    fi
    
    # Step 2: Find video files
    mapfile -t video_files < <(find_video_files)
    
    if [ ${#video_files[@]} -eq 0 ]; then
        print_colored "$RED" "❌ No video files found in the repository!"
        print_colored "$YELLOW" "Please add video files to the repository and try again."
        exit 1
    fi
    
    print_colored "$GREEN" "✅ Found ${#video_files[@]} video file(s)"
    
    # Step 3: Show video selection menu
    show_video_selection "${video_files[@]}"
    
    # Step 4: Get user selection
    if ! selected_index=$(get_user_selection "${video_files[@]}"); then
        print_colored "$CYAN" "👋 Goodbye!"
        exit 0
    fi
    
    selected_video="${video_files[$selected_index]}"
    selected_filename=$(basename "$selected_video")
    print_colored "$GREEN" "✅ Selected: $selected_filename"
    
    # Step 5: Generate unique output path
    mkdir -p "$OUTPUT_DIR"
    output_path=$(generate_output_path "$selected_video")
    
    # Step 6: Run inference
    if run_inference "$selected_video" "$output_path" "$MODEL_NAME" "$CONF_THRESHOLD"; then
        success="true"
    else
        success="false"
    fi
    
    # Step 7: Show summary
    show_summary "$selected_video" "$output_path" "$success"
    
    if [ "$success" = "true" ]; then
        exit 0
    else
        exit 1
    fi
}

# Show help if requested
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat << EOF
Unattended Object Detection - Inference Runner

Usage: $0 [MODEL_NAME] [CONF_THRESHOLD]

Arguments:
    MODEL_NAME       Model name to use (default: rtdetr_tensorrt)
    CONF_THRESHOLD   Confidence threshold (default: 0.5)

Examples:
    $0                           # Use defaults
    $0 rtdetr 0.3               # Custom model and threshold
    $0 rtdetr_tensorrt 0.7      # TensorRT model with high threshold

Features:
    - Integration with manage_jobs.sh for SLURM cluster management
    - Automatic Triton server health checking via cluster state
    - Auto-starts Triton server using SLURM if not running
    - Interactive video file selection
    - Unattended object detection with tracking and alerting
    - Person and target object detection (backpack, handbag, suitcase)
    - Real-time tracking of object-person proximity
    - Automated alerts for unattended objects (15-second threshold)
    - Unique output file naming with timestamps
    - Comprehensive error handling and reporting

Detection Capabilities:
    - Person detection and tracking
    - Target objects: backpack, handbag, suitcase
    - Proximity analysis (450-pixel threshold)
    - Unattended object alerts after 15 seconds
    - Alert logging to alerts.log file
    - Visual annotations on output video

Requirements:
    - SLURM cluster environment
    - NVIDIA Triton Inference Server
    - Python with tritonclient, opencv-python, numpy
    - RT-DETR TensorRT model in model_repository/rtdetr_tensorrt/
    - Properly configured manage_jobs.sh and cluster_config.sh

Cluster Management:
    The script uses the existing cluster management system:
    - manage_jobs.sh for starting/stopping services
    - cluster_config.sh for cluster state coordination
    - Automatic server discovery via SLURM job coordination

Output Files:
    - Processed videos: outputs/{original_name}_inference_{timestamp}.mp4
    - Alert logs: alerts.log (JSON format with timestamps and locations)
    - Visual indicators: Bounding boxes, labels, and alert annotations
EOF
    exit 0
fi

# Run main function
main "$@"