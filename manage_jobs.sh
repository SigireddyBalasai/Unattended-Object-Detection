#!/bin/bash

# SLURM Job Management Script for Triton Inference Server

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

show_help() {
    cat << EOF
SLURM Job Management for Triton Inference Server

Usage: $0 [COMMAND]

Commands:
    start-master    Start the master job (Triton + Flask servers)
    start-triton    Start only Triton server
    start-flask     Start only Flask server  
    start-client    Start client test
    status          Show job status
    logs            Show latest log files
    stop            Cancel all running jobs
    clean           Clean up output files
    help            Show this help

Examples:
    $0 start-master     # Start complete system
    $0 status           # Check job status
    $0 logs             # View recent logs
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
}

start_flask() {
    echo "Starting Flask server..."
    JOBID=$(sbatch run_flask.slurm | grep -o '[0-9]\+')
    echo "Flask job submitted: $JOBID"
    echo "Monitor with: watch squeue -u $USER"
}

start_client() {
    echo "Starting client test..."
    JOBID=$(sbatch run_client.slurm | grep -o '[0-9]\+')
    echo "Client job submitted: $JOBID"
    echo "Monitor with: watch squeue -u $USER"
}

show_status() {
    echo "=== SLURM Job Status ==="
    squeue -u $USER || echo "No jobs found"
    echo ""
    echo "=== Recent Job History ==="
    sacct --format=JobID,JobName,State,ExitCode,Start,End -u $USER | tail -10 || echo "No job history found"
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
    else
        echo "No running jobs found"
    fi
}

clean_files() {
    echo "Cleaning up output files..."
    rm -f *.out *.err *.mp4 2>/dev/null || true
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
    status|st)
        show_status
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
