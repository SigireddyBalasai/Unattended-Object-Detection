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
    echo "TIMESTAMP='$(date '+%Y-%m-%d %H:%M:%S')'" >> "$CLUSTER_STATE_FILE"
    
    # Also write just the node name for easy access
    echo "$node_name" > "$TRITON_NODE_FILE"
}

# Function to read cluster state
read_cluster_state() {
    if [[ -f "$CLUSTER_STATE_FILE" ]]; then
        # Create a temporary safe version of the cluster state file
        local temp_state_file=$(mktemp)
        
        # Process the file to ensure proper quoting of TIMESTAMP
        while IFS= read -r line; do
            if [[ "$line" =~ ^TIMESTAMP=([^\'\"]*[[:space:]][^\'\"]*) ]]; then
                # If TIMESTAMP contains spaces but isn't quoted, add quotes
                echo "TIMESTAMP='${BASH_REMATCH[1]}'" >> "$temp_state_file"
            else
                echo "$line" >> "$temp_state_file"
            fi
        done < "$CLUSTER_STATE_FILE"
        
        # Source the corrected file
        source "$temp_state_file"
        local result=$?
        
        # Clean up temporary file
        rm -f "$temp_state_file"
        
        return $result
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
    local port=${1:-8000}
    local timeout=${2:-300}
    local check_interval=5
    local elapsed=0
    
    if [[ -z "$triton_node" ]]; then
        echo "ERROR: Triton server node not found"
        return 1
    fi
    
    echo "Waiting for Triton server on node: $triton_node:$port"
    
    while [ $elapsed -lt $timeout ]; do
        if curl -s "http://${triton_node}:${port}/v2/health/ready" >/dev/null 2>&1; then
            echo "✓ Triton server is ready on $triton_node:$port"
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

# Function to generate random port
generate_random_port() {
    local min_port=${1:-8000}
    local max_port=${2:-8999}
    local port
    
    while true; do
        port=$((RANDOM % (max_port - min_port + 1) + min_port))
        # Check if port is available
        if ! lsof -i :$port >/dev/null 2>&1; then
            echo $port
            return 0
        fi
    done
}

# Function to generate unique random ports for Triton
generate_triton_ports() {
    local http_port grpc_port metrics_port
    
    # Generate HTTP port (default 8000)
    http_port=$(generate_random_port 8000 8999)
    # Generate gRPC port (default 8001)
    grpc_port=$(generate_random_port 8000 8999)
    # Generate metrics port (default 8002)
    metrics_port=$(generate_random_port 8000 8999)
    
    # Ensure all ports are unique
    while [ "$grpc_port" = "$http_port" ]; do
        grpc_port=$(generate_random_port 8000 8999)
    done
    
    while [ "$metrics_port" = "$http_port" ] || [ "$metrics_port" = "$grpc_port" ]; do
        metrics_port=$(generate_random_port 8000 8999)
    done
    
    echo "$http_port $grpc_port $metrics_port"
}

# Function to write cluster state with ports
write_cluster_state_with_ports() {
    local job_id="$1"
    local node_name="$2"
    local job_type="$3"
    local http_port="$4"
    local grpc_port="$5"
    local metrics_port="$6"
    
    echo "TRITON_JOB_ID=${job_id}" > "$CLUSTER_STATE_FILE"
    echo "TRITON_NODE=${node_name}" >> "$CLUSTER_STATE_FILE"
    echo "JOB_TYPE=${job_type}" >> "$CLUSTER_STATE_FILE"
    echo "HTTP_PORT=${http_port}" >> "$CLUSTER_STATE_FILE"
    echo "GRPC_PORT=${grpc_port}" >> "$CLUSTER_STATE_FILE"
    echo "METRICS_PORT=${metrics_port}" >> "$CLUSTER_STATE_FILE"
    echo "TIMESTAMP='$(date '+%Y-%m-%d %H:%M:%S')'" >> "$CLUSTER_STATE_FILE"
    
    # Also write just the node name for easy access
    echo "$node_name" > "$TRITON_NODE_FILE"
}

# Function to read port information from cluster state
get_triton_ports() {
    if [[ -f "$CLUSTER_STATE_FILE" ]]; then
        # Source the cluster state file to get port variables
        source "$CLUSTER_STATE_FILE" 2>/dev/null || return 1
        
        if [[ -n "${HTTP_PORT:-}" && -n "${GRPC_PORT:-}" && -n "${METRICS_PORT:-}" ]]; then
            echo "$HTTP_PORT $GRPC_PORT $METRICS_PORT"
            return 0
        fi
    fi
    return 1
}