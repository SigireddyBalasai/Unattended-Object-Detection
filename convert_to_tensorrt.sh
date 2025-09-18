#!/bin/bash
# Script to convert ONNX model to TensorRT plan file inside Docker container

set -e

echo "Starting ONNX to TensorRT conversion..."

# Check if ONNX file exists
if [ ! -f "/models/rtdetr_tensorrt/1/model.onnx" ]; then
    echo "Error: model.onnx not found at /models/rtdetr_tensorrt/1/model.onnx"
    exit 1
fi

# Remove existing plan file if it exists
if [ -f "/models/rtdetr_tensorrt/1/model.plan" ]; then
    echo "Removing existing model.plan..."
    rm -f "/models/rtdetr_tensorrt/1/model.plan"
fi

# Install trtexec if not available
if ! command -v trtexec &> /dev/null; then
    echo "trtexec not found, installing TensorRT tools..."
    apt-get update && apt-get install -y libnvinfer-dev libnvinfer-plugin-dev
fi

# Convert ONNX to TensorRT plan
echo "Converting model.onnx to model.plan..."
uv run exporter.py

if [ -f "/models/rtdetr_tensorrt/1/model.plan" ]; then
    echo "Conversion completed successfully!"
    ls -la /models/rtdetr_tensorrt/1/model.plan
else
    echo "Error: Conversion failed - model.plan was not created"
    exit 1
fi