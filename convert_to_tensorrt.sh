#!/bin/bash
# Script to convert ONNX model to TensorRT plan file inside Docker container

set -e

echo "Starting ONNX to TensorRT conversion..."

# If .engine file exists, do nothing
if [ -f "/models/rtdetr_tensorrt/1/model.engine" ]; then
    echo ".engine file already exists. Skipping conversion."
    exit 0
fi

# Check if ONNX file exists
if [ ! -f "/models/rtdetr_tensorrt/1/model.onnx" ]; then
    echo "Error: model.onnx not found at /models/rtdetr_tensorrt/1/model.onnx"
    exit 1
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