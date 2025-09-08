#!/usr/bin/env python3

import cv2
import numpy as np
import tritonclient.http as httpclient
from triton_client import preprocess_frame, draw_detections
import argparse

def postprocess_output_debug(output, frame_shape, conf_threshold=0.5):
    """Debug version of postprocess_output"""
    detections = []
    
    # Remove batch dimension
    output = output[0]  # Shape: [300, 84]
    
    # Get original frame dimensions
    orig_h, orig_w = frame_shape[:2]
    
    for detection in output:
        # Extract bbox and class scores
        bbox = detection[:4]  # [x_center, y_center, width, height]
        class_scores = detection[4:]  # 80 class scores
        
        # Get max confidence and class
        max_conf = np.max(class_scores)
        class_id = np.argmax(class_scores)
        
        if max_conf > conf_threshold:
            # Convert bbox format (center_x, center_y, w, h) to (x1, y1, x2, y2)
            # Coordinates are normalized, need to scale to original image size
            x_center, y_center, w, h = bbox
            
            # Scale from normalized coordinates to original image size
            x_center *= orig_w
            y_center *= orig_h
            w *= orig_w
            h *= orig_h
            
            # Convert center format to corner format
            x1 = max(0, int(x_center - w / 2))
            y1 = max(0, int(y_center - h / 2))
            x2 = min(orig_w, int(x_center + w / 2))
            y2 = min(orig_h, int(y_center + h / 2))
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(max_conf),
                'class_id': int(class_id)
            })
    
    return detections

def debug_inference():
    """Debug RT-DETR inference to see what detections we're getting"""
    
    # Initialize Triton client
    try:
        triton_client = httpclient.InferenceServerClient(url="localhost:8000")
        print("Connected to Triton server")
    except Exception as e:
        print(f"Error connecting to Triton server: {e}")
        return
    
    # Load a test video
    video_path = "ABODA/video1.avi"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        return
    
    print(f"Frame shape: {frame.shape}")
    
    # Preprocess frame
    input_tensor, scale, new_size = preprocess_frame(frame)
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Scale: {scale}")
    print(f"New size: {new_size}")
    
    # Prepare input for Triton
    inputs = []
    inputs.append(httpclient.InferInput("images", input_tensor.shape, "FP32"))
    inputs[0].set_data_from_numpy(input_tensor)
    
    # Prepare output
    outputs = []
    outputs.append(httpclient.InferRequestedOutput("output0"))
    
    # Run inference
    print("Running inference...")
    results = triton_client.infer("rtdetr", inputs, outputs=outputs)
    
    # Get output
    output_data = results.as_numpy("output0")
    print(f"Output shape: {output_data.shape}")
    print(f"Output dtype: {output_data.dtype}")
    
    # Check output statistics
    print(f"Output min: {np.min(output_data)}")
    print(f"Output max: {np.max(output_data)}")
    print(f"Output mean: {np.mean(output_data)}")
    
    # Check the bbox coordinates (first 4 values of each detection)
    bboxes = output_data[0, :, :4]  # Shape: [300, 4]
    print(f"Bbox coordinates range:")
    print(f"  X_center: {np.min(bboxes[:, 0]):.3f} to {np.max(bboxes[:, 0]):.3f}")
    print(f"  Y_center: {np.min(bboxes[:, 1]):.3f} to {np.max(bboxes[:, 1]):.3f}")
    print(f"  Width: {np.min(bboxes[:, 2]):.3f} to {np.max(bboxes[:, 2]):.3f}")
    print(f"  Height: {np.min(bboxes[:, 3]):.3f} to {np.max(bboxes[:, 3]):.3f}")
    
    # Check the class scores (last 80 values of each detection)
    class_scores = output_data[0, :, 4:]  # Shape: [300, 80]
    max_scores = np.max(class_scores, axis=1)  # Max score for each detection
    print(f"\nClass scores statistics:")
    print(f"  Max scores range: {np.min(max_scores):.6f} to {np.max(max_scores):.6f}")
    print(f"  Number of scores > 0.1: {np.sum(max_scores > 0.1)}")
    print(f"  Number of scores > 0.3: {np.sum(max_scores > 0.3)}")
    print(f"  Number of scores > 0.5: {np.sum(max_scores > 0.5)}")
    
    # Show top detections
    sorted_indices = np.argsort(max_scores)[::-1]  # Sort by confidence descending
    print(f"\nTop 10 detections:")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        conf = max_scores[idx]
        class_id = np.argmax(class_scores[idx])
        bbox = bboxes[idx]
        print(f"  {i+1}: conf={conf:.6f}, class={class_id}, bbox={bbox}")
    
    # Test different confidence thresholds
    for threshold in [0.01, 0.05, 0.1, 0.3, 0.5]:
        detections = postprocess_output_debug(output_data, frame.shape[:2], threshold)
        print(f"\nDetections with threshold {threshold}: {len(detections)}")
        if detections:
            for i, det in enumerate(detections[:3]):  # Show first 3
                print(f"  {i+1}: conf={det['confidence']:.6f}, class={det['class_id']}, bbox={det['bbox']}")
    
    cap.release()
    print("\nDebug completed!")

if __name__ == "__main__":
    debug_inference()
