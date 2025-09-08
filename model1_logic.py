import numpy as np
import cv2
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput
import tritonclient.http.aio as triton_async

# === Constants and Helper Functions ===
PERSON_CLASS_ID = 0
TARGET_OBJECT_CLASSES = {
    24: 'backpack',
    26: 'handbag',
    28: 'suitcase',
}

PROXIMITY_THRESHOLD = 100  # pixels, tune as needed
UNATTENDED_TIME_SEC = 30
FPS = 5  # processing every 5th frame in 25 FPS video

def is_near(bbox1, bbox2, threshold=PROXIMITY_THRESHOLD):
    x1, y1, x2, y2 = bbox1
    cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
    x3, y3, x4, y4 = bbox2
    cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
    dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    return dist < threshold

def unattended_object_alert(predictions, frame_index, tracking_state):
    """
    Track unattended objects and generate alerts only if owner has NOT returned.
    """
    current_time = frame_index / FPS  # seconds
    persons = [det for det in predictions if det['class_id'] == PERSON_CLASS_ID]
    objects = [det for det in predictions if det['class_id'] in TARGET_OBJECT_CLASSES]

    alerts = []

    for obj in objects:
        obj_id = f"{obj['class_id']}_{obj['bbox']}"

        # Find persons near the object
        nearby_persons = [p for p in persons if is_near(obj['bbox'], p['bbox'])]

        if nearby_persons:
            # Someone is attending the object now - reset or update tracking info
            # Assume first nearby person as owner (could extend with tracking IDs)
            owner = nearby_persons[0]
            tracking_state[obj_id] = {
                'last_seen_with_person': current_time,
                'owner_bbox': owner['bbox'],
                'bbox': obj['bbox'],
                'class_id': obj['class_id'],
                'alerted': False  # reset alert status if owner returned
            }
        else:
            # No person near this object right now
            if obj_id in tracking_state:
                last_seen = tracking_state[obj_id]['last_seen_with_person']
                alerted = tracking_state[obj_id].get('alerted', False)

                # If unattended for longer than threshold and not alerted before
                if (current_time - last_seen) > UNATTENDED_TIME_SEC and not alerted:
                    alerts.append({
                        'object': TARGET_OBJECT_CLASSES[obj['class_id']],
                        'bbox': obj['bbox'],
                        'unattended_since': last_seen,
                        'alert_time': current_time
                    })
                    # Mark as alerted to avoid repeat alerts
                    tracking_state[obj_id]['alerted'] = True

            else:
                # First time seeing this unattended object, track it
                tracking_state[obj_id] = {
                    'last_seen_with_person': -1,  # No owner seen yet
                    'owner_bbox': None,
                    'bbox': obj['bbox'],
                    'class_id': obj['class_id'],
                    'alerted': False
                }

    # Clean up tracking_state to remove objects no longer detected if needed (optional)
    # Could add a timeout for objects missing for many frames

    return alerts

# === Existing functions ===
def preprocess_frame(frame):
    frame = cv2.resize(frame, (640, 640))
    frame = frame.astype(np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    return frame

def postprocess_output(output, frame_shape, conf_threshold=0.5):
    detections = []
    output = output[0]
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

def get_video_predictions(video_path):
    client = InferenceServerClient(url="localhost:8000")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    tracking_state = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:
            input_data = preprocess_frame(frame)
            inputs = []
            input_tensor = InferInput("images", input_data.shape, "FP32")
            input_tensor.set_data_from_numpy(input_data)
            inputs.append(input_tensor)
            results = client.infer("rtdetr", inputs)

            outputs = []
            for output in results.get_response()['outputs']:
                outputs.append(np.frombuffer(output['data'], dtype=np.float32).reshape(output['shape']))

            predictions = postprocess_output(outputs, frame.shape, conf_threshold=0.3)

            alerts = unattended_object_alert(predictions, frame_count // 5, tracking_state)

            for alert in alerts:
                print(f"[ALERT] {alert['object']} unattended since {alert['unattended_since']:.1f}s (Current: {alert['alert_time']:.1f}s)")

            yield predictions

        frame_count += 1

    cap.release()
