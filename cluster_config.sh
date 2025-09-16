#!/bin/bash
# Shared cluster configuration for Triton inference system
# This file coordinates resource allocation across all jobs

CLUSTER_CONFIG_DIR="${HOME}/.triton_cluster"
CLUSTER_STATE_FILE="${CLUSTER_CONFIG_DIR}/cluster_state"
TRITON_NODE_FILE="${CLUSTER_CONFIG_DIR}/triton_node"

# Create config directory if it doesn't exist
mkdir -p "$CLUSTER_CONFIG_DIR"

# Function to write cluster state
write_cluster_state() {
    local job_id="$1"
    local node_name="$2"
    local job_type="$3"
    
    echo "TRITON_JOB_ID=${job_id}" > "$CLUSTER_STATE_FILE"
    echo "TRITON_NODE=${node_name}" >> "$CLUSTER_STATE_FILE"
    echo "JOB_TYPE=${job_type}" >> "$CLUSTER_STATE_FILE"
    echo "TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')" >> "$CLUSTER_STATE_FILE"
    
    # Also write just the node name for easy access
    echo "$node_name" > "$TRITON_NODE_FILE"
}

# Function to read cluster state
read_cluster_state() {
    if [[ -f "$CLUSTER_STATE_FILE" ]]; then
        source "$CLUSTER_STATE_FILE"
        return 0
    else
        return 1
    fi
}

# Function to get Triton server node
get_triton_node() {
    if [[ -f "$TRITON_NODE_FILE" ]]; then
        cat "$TRITON_NODE_FILE"
        return 0
    else
        # Try to detect from running jobs
        local node=$(squeue -u $USER -n triton-inference -h -o %N 2>/dev/null | head -1)
        if [[ -n "$node" ]]; then
            echo "$node" > "$TRITON_NODE_FILE"
            echo "$node"
            return 0
        else
            return 1
        fi
    fi
}

# Function to wait for Triton server
wait_for_triton() {
    local triton_node=$(get_triton_node)
    local timeout=${1:-300}
    local check_interval=5
    local elapsed=0
    
    if [[ -z "$triton_node" ]]; then
        echo "ERROR: Triton server node not found"
        return 1
    fi
    
    echo "Waiting for Triton server on node: $triton_node"
    
    while [ $elapsed -lt $timeout ]; do
        if curl -s "http://${triton_node}:8000/v2/health/ready" >/dev/null 2>&1; then
            echo "✓ Triton server is ready on $triton_node"
            return 0
        fi
        
        echo "Waiting for server... (${elapsed}s/${timeout}s elapsed)"
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done
    
    echo "ERROR: Triton server not ready after ${timeout} seconds"
    return 1
}

# Function to clean up cluster state
cleanup_cluster_state() {
    rm -f "$CLUSTER_STATE_FILE" "$TRITON_NODE_FILE"
    echo "Cluster state cleaned up"
}

# Function to check if Triton is running
is_triton_running() {
    local job_running=$(squeue -u $USER -n triton-inference -h -o %i 2>/dev/null | head -1)
    if [[ -n "$job_running" ]]; then
        return 0
    else
        return 1
    fi
}

# Export functions
export -f write_cluster_state
export -f read_cluster_state
export -f get_triton_node
export -f wait_for_triton
export -f cleanup_cluster_state
export -f is_triton_running