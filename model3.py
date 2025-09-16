import numpy as np
import cv2
import json
import logging
import os
import subprocess
from tritonclient.http import InferenceServerClient, InferInput
from tritonclient.utils import InferenceServerException

# === Constants ===
PERSON_CLASS_ID = 0
TARGET_OBJECT_CLASSES = {
    24: 'backpack',
    26: 'handbag',
    28: 'suitcase',
}
PROXIMITY_THRESHOLD = 100  # pixels
UNATTENDED_TIME_SEC = 30
PROCESS_EVERY_N_FRAMES = 5  # process every 5th frame
LOG_FILE = "alerts.log"

# === Logging setup ===
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s - %(message)s",
    level=logging.INFO
)

# === Cluster Configuration ===
def get_triton_server_url():
    """Get Triton server URL from cluster configuration."""
    cluster_config_dir = os.path.expanduser("~/.triton_cluster")
    triton_node_file = os.path.join(cluster_config_dir, "triton_node")
    
    # Try to read from cluster state file
    if os.path.exists(triton_node_file):
        try:
            with open(triton_node_file, 'r') as f:
                triton_node = f.read().strip()
            if triton_node:
                return f"{triton_node}:8000"
        except Exception as e:
            print(f"Warning: Could not read cluster state: {e}")
    
    # Fallback: try to detect from running SLURM jobs
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-n", "triton-inference", "-h", "-o", "%N"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            triton_node = result.stdout.strip().split('\n')[0]
            # Save to cluster state for future use
            os.makedirs(cluster_config_dir, exist_ok=True)
            with open(triton_node_file, 'w') as f:
                f.write(triton_node)
            return f"{triton_node}:8000"
    except Exception as e:
        print(f"Warning: Could not detect Triton server from SLURM: {e}")
    
    # Final fallback to localhost
    print("Warning: Using localhost as fallback. This may not work in cluster environments.")
    return "localhost:8000"

# === Utilities ===
def is_near(bbox1, bbox2, threshold=PROXIMITY_THRESHOLD):
    x1, y1, x2, y2 = bbox1
    cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
    x3, y3, x4, y4 = bbox2
    cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
    dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    return dist < threshold

def iou(bbox1, bbox2):
    """Compute IoU for bbox matching (x1,y1,x2,y2 format)."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def preprocess_frame(frame):
    orig_h, orig_w = frame.shape[:2]
    resized = cv2.resize(frame, (640, 640))
    resized = resized.astype(np.float32) / 255.0
    resized = np.transpose(resized, (2, 0, 1))  # CHW
    resized = np.expand_dims(resized, axis=0)   # BCHW
    return resized, (orig_w, orig_h)

def postprocess_output(output, orig_shape, conf_threshold=0.5):
    detections = []
    output = output[0]  # Shape: [N, 84]
    orig_w, orig_h = orig_shape

    for detection in output:
        bbox = detection[:4]
        class_scores = detection[4:]
        max_conf = np.max(class_scores)
        class_id = np.argmax(class_scores)
        if max_conf > conf_threshold:
            x_center, y_center, w, h = bbox
            # scale back to original resolution
            x_center *= orig_w
            y_center *= orig_h
            w *= orig_w
            h *= orig_h
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

def match_object(obj, tracking_state, iou_thresh=0.5):
    """Try to match obj to an existing tracked object by IoU."""
    for obj_id, state in tracking_state.items():
        if iou(obj['bbox'], state['bbox']) > iou_thresh and obj['class_id'] == state['class_id']:
            return obj_id
    return None

def unattended_object_alert(predictions, current_time, tracking_state):
    persons = [det for det in predictions if det['class_id'] == PERSON_CLASS_ID]
    objects = [det for det in predictions if det['class_id'] in TARGET_OBJECT_CLASSES]
    alerts = []

    for obj in objects:
        obj_id = match_object(obj, tracking_state) or f"new_{len(tracking_state)+1}"
        nearby_persons = [p for p in persons if is_near(obj['bbox'], p['bbox'])]

        if nearby_persons:
            owner = nearby_persons[0]
            tracking_state[obj_id] = {
                'last_seen_with_person': current_time,
                'owner_bbox': owner['bbox'],
                'bbox': obj['bbox'],
                'class_id': obj['class_id'],
                'alerted': False
            }
        else:
            if obj_id in tracking_state:
                last_seen = tracking_state[obj_id]['last_seen_with_person']
                alerted = tracking_state[obj_id].get('alerted', False)
                if (current_time - last_seen) > UNATTENDED_TIME_SEC and not alerted:
                    alert_data = {
                        'object': TARGET_OBJECT_CLASSES[obj['class_id']],
                        'bbox': obj['bbox'],
                        'unattended_since': last_seen,
                        'alert_time': current_time
                    }
                    alerts.append(alert_data)
                    logging.info(json.dumps(alert_data))
                    print(f"⚠️ ALERT: {alert_data}")
                    tracking_state[obj_id]['alerted'] = True
            else:
                tracking_state[obj_id] = {
                    'last_seen_with_person': -1,
                    'owner_bbox': None,
                    'bbox': obj['bbox'],
                    'class_id': obj['class_id'],
                    'alerted': False
                }

    return alerts

# === Main Prediction Loop ===
def get_video_predictions(video_path):
    # Get Triton server URL from cluster configuration
    triton_url = get_triton_server_url()
    print(f"Connecting to Triton server at: {triton_url}")
    
    try:
        client = InferenceServerClient(url=triton_url)
        # Test connection
        if not client.is_server_ready():
            raise Exception(f"Triton server at {triton_url} is not ready")
        print(f"✓ Successfully connected to Triton server at {triton_url}")
    except Exception as e:
        print(f"✗ Failed to connect to Triton server at {triton_url}: {e}")
        print("Make sure the Triton server is running and accessible.")
        raise
    
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_count = 0
    tracking_state = {}

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                input_data, orig_shape = preprocess_frame(frame)
                inputs = []
                input_tensor = InferInput("images", input_data.shape, "FP32")
                input_tensor.set_data_from_numpy(input_data)
                inputs.append(input_tensor)

                try:
                    results = client.infer("rtdetr", inputs)
                    output_names = results.get_response()["outputs"]
                    output_key = output_names[0]["name"] if output_names else "output0"
                    output = results.as_numpy(output_key)
                except InferenceServerException as e:
                    print(f"Inference failed: {e}")
                    break

                current_time = frame_count / video_fps
                predictions = postprocess_output(output, orig_shape, conf_threshold=0.3)
                alerts = unattended_object_alert(predictions, current_time, tracking_state)

                yield {
                    "frame": frame_count,
                    "time_sec": current_time,
                    "frame_img": frame,
                    "alerts": alerts,
                    "predictions": predictions
                }

            frame_count += 1
    finally:
        cap.release()

# === Draw and save video ===
def save_output_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    try:
        for result in get_video_predictions(input_video_path):
            frame = result['frame_img']
            predictions = result['predictions']
            alerts = result['alerts']

            # Draw detections
            for det in predictions:
                x1, y1, x2, y2 = det['bbox']
                class_id = det['class_id']
                color = (255, 0, 0) if class_id == PERSON_CLASS_ID else (0, 255, 0)
                label = TARGET_OBJECT_CLASSES.get(class_id, "person" if class_id==0 else str(class_id))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw unattended alerts
            for alert in alerts:
                x1, y1, x2, y2 = alert['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"UNATTENDED: {alert['object']}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            out.write(frame)
    finally:
        out.release()
        print(f"✅ Saved output video to {output_video_path}")

# === Example usage ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process video for unattended object detection")
    parser.add_argument("--input", default="ABODA/video11.avi", help="Input video path")
    parser.add_argument("--output", default="output_final.mp4", help="Output video path")
    parser.add_argument("--triton-url", default=None, help="Triton server URL (e.g., gpu001:8000)")
    
    args = parser.parse_args()
    
    # Override cluster detection if URL is provided
    if args.triton_url:
        def override_get_triton_server_url():
            return args.triton_url
        get_triton_server_url = override_get_triton_server_url
        print(f"Using provided Triton server URL: {args.triton_url}")
    
    print(f"Processing video: {args.input}")
    print(f"Output will be saved to: {args.output}")
    
    try:
        save_output_video(args.input, args.output)
        print("✅ Video processing completed successfully!")
    except Exception as e:
        print(f"✗ Error during video processing: {e}")
        exit(1)
