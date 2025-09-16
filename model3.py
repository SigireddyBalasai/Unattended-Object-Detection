import numpy as np
import cv2
import json
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
PROCESS_EVERY_N_FRAMES = 5  # analyze every 5th frame

LOG_FILE = "alerts.log"

# === Utilities ===
def is_near(bbox1, bbox2, threshold=PROXIMITY_THRESHOLD):
    x1, y1, x2, y2 = bbox1
    cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
    x3, y3, x4, y4 = bbox2
    cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
    dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    return dist < threshold

def preprocess_frame(frame):
    frame = cv2.resize(frame, (640, 640))
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))  # CHW
    frame = np.expand_dims(frame, axis=0)  # BCHW
    return frame

def postprocess_output(output, frame_shape, conf_threshold=0.5):
    detections = []
    output = output[0]  # Shape: [N, 84] (N detections)
    orig_h, orig_w = frame_shape[:2]

    for detection in output:
        bbox = detection[:4]
        class_scores = detection[4:]
        max_conf = np.max(class_scores)
        class_id = np.argmax(class_scores)
        if max_conf > conf_threshold:
            x_center, y_center, w, h = bbox
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

def make_obj_id(det):
    """Stable ID: class + rounded bbox"""
    x1, y1, x2, y2 = det['bbox']
    return f"{det['class_id']}_{round(x1)}_{round(y1)}_{round(x2)}_{round(y2)}"

def unattended_object_alert(predictions, current_time, tracking_state):
    persons = [det for det in predictions if det['class_id'] == PERSON_CLASS_ID]
    objects = [det for det in predictions if det['class_id'] in TARGET_OBJECT_CLASSES]
    alerts = []

    for obj in objects:
        obj_id = make_obj_id(obj)
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

                    # Log alert
                    with open(LOG_FILE, "a") as log_file:
                        log_file.write(json.dumps(alert_data) + "\n")

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
    client = InferenceServerClient(url="localhost:8000")
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # fallback if 0
    frame_count = 0
    tracking_state = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            input_data = preprocess_frame(frame)
            inputs = []
            input_tensor = InferInput("images", input_data.shape, "FP32")
            input_tensor.set_data_from_numpy(input_data)
            inputs.append(input_tensor)

            try:
                results = client.infer("rtdetr", inputs)
                # safer output extraction (check available outputs)
                output_names = results.get_response()["outputs"]
                output_key = output_names[0]["name"] if output_names else "output0"
                output = results.as_numpy(output_key)
            except InferenceServerException as e:
                print(f"Inference failed: {e}")
                break

            current_time = frame_count / video_fps  # seconds
            predictions = postprocess_output(output, frame.shape, conf_threshold=0.3)
            alerts = unattended_object_alert(predictions, current_time, tracking_state)

            yield {
                "frame": frame_count,
                "time_sec": current_time,
                "alerts": alerts,
                "predictions": predictions
            }

        frame_count += 1

    cap.release()
