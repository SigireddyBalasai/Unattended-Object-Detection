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
    return frame

def postprocess_output(output):
    # Convert raw model output to detection format
    # This needs to be adjusted based on your model's output format
    boxes = output[0]  # Assuming first output is bounding boxes
    scores = output[1]  # Assuming second output is confidence scores
    classes = output[2]  # Assuming third output is class predictions
    
    return {
        'boxes': boxes.tolist(),
        'scores': scores.tolist(),
        'classes': classes.tolist()
    }

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
            input_tensor = InferInput("input", input_data.shape, "FP32")
            input_tensor.set_data_from_numpy(input_data)
            inputs.append(input_tensor)
            
            # Get predictions from Triton server
            results = client.infer("rtdetr", inputs)
            
            # Convert results to numpy arrays
            outputs = []
            for output in results.get_response()['outputs']:
                outputs.append(np.frombuffer(output['data'], 
                                          dtype=np.float32).reshape(output['shape']))
            
            # Postprocess and yield results
            yield postprocess_output(outputs)
            
        frame_count += 1
    
    cap.release()


