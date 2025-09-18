#!/bin/bash

# SLURM Job Management Script for Triton Inference Server

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source cluster configuration
source ./cluster_config.sh

show_help() {
    cat << EOF
SLURM Job Management for Triton Inference Server

Usage: $0 [COMMAND]

Commands:
    start-master    Start the master job (Triton + Flask servers)
    start-triton    Start only Triton server
    start-flask     Start only Flask server  
    start-client    Start client test
    start-client2   Start client 2 (legacy)
    status          Show job status
    logs            Show latest log files
    stop            Cancel all running jobs
    clean           Clean up output files
    cluster-info    Show current cluster state
    help            Show this help

Examples:
    $0 start-master     # Start complete system
    $0 status           # Check job status
    $0 logs             # View recent logs
    $0 cluster-info     # Show Triton server location
    $0 stop             # Stop all jobs
EOF
}

start_master() {
    echo "Starting master job (Triton + Flask)..."
    JOBID=$(sbatch run_master.slurm | grep -o '[0-9]\+')
    echo "Master job submitted: $JOBID"
    echo "Monitor with: watch squeue -u $USER"
    echo "View logs with: tail -f master_${JOBID}.out"
}

start_triton() {
    echo "Starting Triton server..."
    JOBID=$(sbatch run_triton.slurm | grep -o '[0-9]\+')
    echo "Triton job submitted: $JOBID"
    echo "Monitor with: watch squeue -u $USER"
    echo "View logs with: tail -f triton_${JOBID}.out"
    echo "Waiting a few seconds for job to start..."
    sleep 5
    echo "Cluster state will be available once the job starts running."
}

start_flask() {
    # Check if Triton is running first
    if ! is_triton_running; then
        echo "Warning: No Triton server detected. Starting Triton first..."
        start_triton
        echo "Waiting for Triton to start before launching Flask..."
        sleep 10
    fi
    
    echo "Starting Flask server..."
    JOBID=$(sbatch run_flask.slurm | grep -o '[0-9]\+')
    echo "Flask job submitted: $JOBID"
    echo "Monitor with: watch squeue -u $USER"
    echo "View logs with: tail -f flask_${JOBID}.out"
}

start_client() {
    # Check if Triton is running first
    if ! is_triton_running; then
        echo "Error: No Triton server detected. Please start Triton server first:"
        echo "  $0 start-triton"
        echo "  $0 start-master"
        exit 1
    fi
    
    echo "Starting client test..."
    JOBID=$(sbatch run_client.slurm | grep -o '[0-9]\+')
    echo "Client job submitted: $JOBID"
    echo "Monitor with: watch squeue -u $USER"
    echo "View logs with: tail -f client_${JOBID}.out"
}

start_client2() {
    # Check if Triton is running first
    if ! is_triton_running; then
        echo "Error: No Triton server detected. Please start Triton server first:"
        echo "  $0 start-triton"
        echo "  $0 start-master"
        exit 1
    fi
    
    echo "Starting client v2 test..."
    JOBID=$(sbatch run_client_v2.slurm | grep -o '[0-9]\+')
    echo "Client v2 job submitted: $JOBID"
    echo "Monitor with: watch squeue -u $USER"
    echo "View logs with: tail -f client_v2_${JOBID}.out"
}

show_status() {
    echo "=== SLURM Job Status ==="
    squeue -u $USER || echo "No jobs found"
    echo ""
    
    echo "=== Cluster State ==="
    if read_cluster_state; then
        echo "Triton server job ID: ${TRITON_JOB_ID:-Unknown}"
        echo "Triton server node: ${TRITON_NODE:-Unknown}"
        echo "Job type: ${JOB_TYPE:-Unknown}"
        echo "Last updated: ${TIMESTAMP:-Unknown}"
        
        # Test server connectivity
        if [[ -n "${TRITON_NODE:-}" ]]; then
            # Get ports from cluster state
            local triton_ports=$(get_triton_ports)
            if [[ -n "$triton_ports" ]]; then
                read HTTP_PORT GRPC_PORT METRICS_PORT <<< "$triton_ports"
            else
                HTTP_PORT=8000
            fi
            echo "Testing server connectivity..."
            if curl -s "http://${TRITON_NODE}:${HTTP_PORT}/v2/health/ready" >/dev/null 2>&1; then
                echo "✓ Triton server is responding at ${TRITON_NODE}:${HTTP_PORT}"
            else
                echo "✗ Triton server not responding at ${TRITON_NODE}:${HTTP_PORT}"
            fi
        fi
    else
        echo "No cluster state found. Triton server may not be running."
    fi
    
    echo ""
    echo "=== Recent Job History ==="
    sacct --format=JobID,JobName,State,ExitCode,Start,End -u $USER | tail -10 || echo "No job history found"
}

show_cluster_info() {
    echo "=== Cluster Configuration ==="
    if read_cluster_state; then
        echo "Active Triton server:"
        echo "  Job ID: ${TRITON_JOB_ID:-Unknown}"
        echo "  Node: ${TRITON_NODE:-Unknown}"
        echo "  Job Type: ${JOB_TYPE:-Unknown}"
        echo "  Last Updated: ${TIMESTAMP:-Unknown}"
        echo ""
        echo "Server endpoints:"
        # Get ports from cluster state
        local triton_ports=$(get_triton_ports)
        if [[ -n "$triton_ports" ]]; then
            read HTTP_PORT GRPC_PORT METRICS_PORT <<< "$triton_ports"
        else
            HTTP_PORT=8000
            GRPC_PORT=8001
            METRICS_PORT=8002
        fi
        echo "  HTTP: http://${TRITON_NODE:-unknown}:${HTTP_PORT}"
        echo "  gRPC: ${TRITON_NODE:-unknown}:${GRPC_PORT}"
        echo "  Metrics: http://${TRITON_NODE:-unknown}:${METRICS_PORT}"
        echo ""
        
        # Test connectivity
        if [[ -n "${TRITON_NODE:-}" ]]; then
            echo "Connectivity test:"
            if curl -s "http://${TRITON_NODE}:${HTTP_PORT}/v2/health/ready" >/dev/null 2>&1; then
                echo "  ✓ Server is ready and responding"
                
                # Get model info
                if model_info=$(curl -s "http://${TRITON_NODE}:${HTTP_PORT}/v2/models" 2>/dev/null); then
                    echo "  ✓ Models available: $(echo "$model_info" | jq -r '.[].name' 2>/dev/null | tr '\n' ' ' || echo "Could not parse model list")"
                fi
            else
                echo "  ✗ Server not responding"
            fi
        fi
    else
        echo "No active cluster state found."
        echo "This could mean:"
        echo "  - No Triton server is currently running"
        echo "  - Cluster state was cleaned up"
        echo "  - Server was started without cluster coordination"
        echo ""
        echo "To start a coordinated Triton server:"
        echo "  $0 start-triton"
        echo "  $0 start-master"
    fi
}

show_logs() {
    echo "=== Latest Log Files ==="
    ls -lt *.out *.err 2>/dev/null | head -10 || echo "No log files found"
    echo ""
    echo "To view a specific log: tail -f <filename>"
    echo "To follow all outputs: tail -f *.out"
}

stop_jobs() {
    echo "Stopping all jobs for user $USER..."
    JOBS=$(squeue -u $USER -h -o %A || true)
    if [[ -n "$JOBS" ]]; then
        echo "$JOBS" | xargs -r scancel
        echo "All jobs cancelled"
        
        # Clean up cluster state
        cleanup_cluster_state
        echo "Cluster state cleaned up"
    else
        echo "No running jobs found"
    fi
}

clean_files() {
    echo "Cleaning up output files..."
    rm -f *.out *.err *.mp4 2>/dev/null || true
    cleanup_cluster_state
    echo "Cleanup complete"
}

case "${1:-help}" in
    start-master|master)
        start_master
        ;;
    start-triton|triton)
        start_triton
        ;;
    start-flask|flask)
        start_flask
        ;;
    start-client|client)
        start_client
        ;;
    start-client2|client2)
        start_client2
        ;;
    status|st)
        show_status
        ;;
    cluster-info|cluster|info)
        show_cluster_info
        ;;
    logs|log)
        show_logs
        ;;
    stop|cancel)
        stop_jobs
        ;;
    clean|cleanup)
        clean_files
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
