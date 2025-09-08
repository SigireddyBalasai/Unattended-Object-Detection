import numpy as np
import cv2
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import tritonclient.http.aio as triton_async


def preprocess_frame(frame):
    # Resize and normalize the frame
    frame = cv2.resize(frame, (640, 640))
    frame = frame.astype(np.float32) / 255.0
    # Change from HWC to CHW format
    frame = np.transpose(frame, (2, 0, 1))
    # Add batch dimension for Triton (BCHW format)
    frame = np.expand_dims(frame, axis=0)
    return frame

def postprocess_output(output, frame_shape, conf_threshold=0.5):
    """
    Postprocess RT-DETR output
    Output format: [batch, 300, 84] where 84 = 4 (bbox) + 80 (classes)
    """
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

def get_video_predictions(video_path):
    # Initialize Triton client
    client = InferenceServerClient(url="localhost:8000")
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process every 5th frame (vid_stride=5)
        if frame_count % 5 == 0:
            # Preprocess frame
            input_data = preprocess_frame(frame)
            
            # Prepare inputs
            inputs = []
            input_tensor = InferInput("images", input_data.shape, "FP32")
            input_tensor.set_data_from_numpy(input_data)
            inputs.append(input_tensor)
            
            # Get predictions from Triton server
            results = client.infer("rtdetr", inputs)
            
            # Convert results to numpy arrays
            outputs = []
            for output in results.get_response()['outputs']:
                outputs.append(np.frombuffer(output['data'], 
                                          dtype=np.float32).reshape(output['shape']))
            
            # Postprocess and yield results with lower confidence threshold
            yield postprocess_output(outputs, frame.shape, conf_threshold=0.3)
            
        frame_count += 1
    
    cap.release()


