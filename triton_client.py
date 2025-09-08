#!/usr/bin/env python3

import cv2
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
import argparse
import sys
import time
import os

def preprocess_frame(frame, input_shape=(640, 640)):
    """
    Preprocess frame for RT-DETR model
    """
    # Resize frame
    h, w = frame.shape[:2]
    scale = min(input_shape[0] / h, input_shape[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Pad to target size
    padded = np.full((input_shape[0], input_shape[1], 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    # Convert to RGB and normalize
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    
    # Transpose to CHW format
    input_tensor = np.transpose(normalized, (2, 0, 1))
    
    # Add batch dimension
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    return input_tensor, scale, (new_h, new_w)

def postprocess_output(output, scale, original_size, conf_threshold=0.5):
    """
    Postprocess RT-DETR output
    Output format: [batch, 300, 84] where 84 = 4 (bbox) + 80 (classes)
    """
    detections = []
    
    # Remove batch dimension
    output = output[0]  # Shape: [300, 84]
    
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
            orig_h, orig_w = original_size
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

def draw_detections(frame, detections, class_names=None):
    """
    Draw bounding boxes and labels on frame
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        class_id = det['class_id']
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}: {conf:.2f}"
        else:
            label = f"Class {class_id}: {conf:.2f}"
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return frame

def main():
    parser = argparse.ArgumentParser(description="RT-DETR Triton Inference Client")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model-name", default="rtdetr", help="Model name in Triton")
    parser.add_argument("--url", default="localhost:8000", help="Triton server URL")
    parser.add_argument("--output", help="Output video path (optional)")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found")
        sys.exit(1)
    
    # Initialize Triton client
    try:
        triton_client = httpclient.InferenceServerClient(url=args.url)
        print(f"Connected to Triton server at {args.url}")
    except Exception as e:
        print(f"Error connecting to Triton server: {e}")
        sys.exit(1)
    
    # Check if model is ready
    try:
        model_metadata = triton_client.get_model_metadata(args.model_name)
        print(f"Model '{args.model_name}' is ready")
    except InferenceServerException as e:
        print(f"Error: Model '{args.model_name}' is not available: {e}")
        sys.exit(1)
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{args.video}'")
        sys.exit(1)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup output video if specified
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # COCO class names (80 classes)
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    frame_count = 0
    total_inference_time = 0
    
    print("Starting inference...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}/{total_frames}", end='\r')
            
            # Preprocess frame
            input_tensor, scale, new_size = preprocess_frame(frame)
            
            # Prepare input for Triton
            inputs = []
            inputs.append(httpclient.InferInput("images", input_tensor.shape, "FP32"))
            inputs[0].set_data_from_numpy(input_tensor)
            
            # Prepare output
            outputs = []
            outputs.append(httpclient.InferRequestedOutput("output0"))
            
            # Run inference
            start_time = time.time()
            results = triton_client.infer(args.model_name, inputs, outputs=outputs)
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Get output
            output_data = results.as_numpy("output0")
            
            # Postprocess
            detections = postprocess_output(output_data, scale, (height, width), args.conf_threshold)
            
            # Draw detections
            frame_with_detections = draw_detections(frame.copy(), detections, class_names)
            
            # Add FPS info
            fps_text = f"FPS: {1.0/inference_time:.1f}"
            cv2.putText(frame_with_detections, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save or display frame
            if out:
                out.write(frame_with_detections)
            else:
                cv2.imshow('RT-DETR Detection', frame_with_detections)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError during inference: {e}")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        if frame_count > 0:
            avg_fps = frame_count / total_inference_time
            print(f"\nProcessed {frame_count} frames")
            print(f"Average inference FPS: {avg_fps:.2f}")
            if args.output:
                print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
