# detector.py
import cv2
import math
import time
import csv
import os
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput
import configparser
from typing import List, Dict, Tuple

# ==================== CONFIG ====================
parser = configparser.ConfigParser()
parser.read('./config/config.ini')

video_source = parser['DEFAULT'].get('source', 0)
distance_threshold = parser['DEFAULT'].getint('distance_from_bag', fallback=150)
unattended_time_sec = parser['DEFAULT'].getint('unattended_time', fallback=30)
show_gui = parser['DEFAULT'].getboolean('show_gui', fallback=True)
process_every_n_frames = parser['DEFAULT'].getint('process_every_n_frames', fallback=5)

print(f"Video source: {video_source}")
print(f"Distance threshold: {distance_threshold}")
print(f"Unattended time: {unattended_time_sec} sec")
print(f"Process every {process_every_n_frames} frames")

# ==================== LOGGING ====================
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "unattended_log.csv")

if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "frame_no", "bag_track_id", "status"])

def log_event(frame_no, bag_track_id, status):
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), frame_no, bag_track_id, status])

# ==================== SIMPLE CENTROID TRACKER ====================
class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=1000):
        self.next_id = 0
        self.objects = {}  # id -> bbox [x1,y1,x2,y2]
        self.disappeared = {}  # id -> frames disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, rects: List[List[int]]) -> Dict[int, List[int]]:
        # rects: list of bboxes
        if len(rects) == 0:
            # increment disappeared counters
            keys = list(self.disappeared.keys())
            for objectID in keys:
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    del self.objects[objectID]
                    del self.disappeared[objectID]
            return dict(self.objects)

        # if no existing objects, register all rects
        if len(self.objects) == 0:
            for r in rects:
                self.objects[self.next_id] = r
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return dict(self.objects)

        # otherwise compute pairwise distance between existing centroids and new rects
        objectIDs = list(self.objects.keys())
        objectCentroids = [self._centroid(self.objects[oid]) for oid in objectIDs]
        inputCentroids = [self._centroid(r) for r in rects]

        D = np.zeros((len(objectCentroids), len(inputCentroids)), dtype=np.float32)
        for i, oc in enumerate(objectCentroids):
            for j, ic in enumerate(inputCentroids):
                D[i, j] = math.hypot(oc[0] - ic[0], oc[1] - ic[1])

        # greedy match: for robustness given small numbers, do simple greedy matching
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows, assigned_cols = set(), set()
        new_objects = {}

        for row, col in zip(rows, cols):
            if row in assigned_rows or col in assigned_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            objectID = objectIDs[row]
            self.objects[objectID] = rects[col]
            self.disappeared[objectID] = 0
            assigned_rows.add(row)
            assigned_cols.add(col)

        # unassigned new rects -> register
        for j in range(len(rects)):
            if j not in assigned_cols:
                self.objects[self.next_id] = rects[j]
                self.disappeared[self.next_id] = 0
                self.next_id += 1

        # unassigned old objects -> increase disappeared
        for i in range(len(objectIDs)):
            if i not in assigned_rows:
                objectID = objectIDs[i]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    del self.objects[objectID]
                    del self.disappeared[objectID]

        return dict(self.objects)

# ==================== TRITON HELPERS ====================
INPUT_W, INPUT_H = 640, 640  # model input size used in preprocess

def preprocess_frame(frame):
    resized = cv2.resize(frame, (INPUT_W, INPUT_H))
    arr = resized.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr

def extract_outputs_from_triton_result(result):
    # result.get_response() returns dict-like with 'outputs'
    # Each output entry will have 'name','datatype','shape','data' (flat list) in many triton setups
    resp = result.get_response()
    outputs = []
    for out in resp.get('outputs', []):
        data = np.array(out['data'], dtype=np.float32)
        shape = tuple(out['shape'])
        outputs.append(data.reshape(shape))
    return outputs

def postprocess_output(output_array, orig_shape, conf_threshold=0.3):
    """
    output_array: numpy [N, 84] where first 4 are bbox (x_center,y_center,w,h) normalized to input dims [0-1]
    orig_shape: original frame shape (h,w,channels)
    returns list of detections: {'bbox':[x1,y1,x2,y2],'confidence':float,'class_id':int}
    """
    detections = []
    if output_array is None:
        return detections

    orig_h, orig_w = orig_shape[:2]
    for det in output_array:
        bbox = det[:4]
        class_scores = det[4:]
        max_conf = float(np.max(class_scores))
        class_id = int(np.argmax(class_scores))
        if max_conf > conf_threshold:
            x_center, y_center, w, h = bbox  # assumed normalized [0,1]
            # scale from normalized to model input pixels
            x_center *= INPUT_W
            y_center *= INPUT_H
            w *= INPUT_W
            h *= INPUT_H
            # map from model input to original frame
            scale_x = orig_w / INPUT_W
            scale_y = orig_h / INPUT_H
            x_center *= scale_x
            y_center *= scale_y
            w *= scale_x
            h *= scale_y
            x1 = max(0, int(x_center - w / 2))
            y1 = max(0, int(y_center - h / 2))
            x2 = min(orig_w - 1, int(x_center + w / 2))
            y2 = min(orig_h - 1, int(y_center + h / 2))
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(max_conf),
                'class_id': class_id
            })
    return detections

def calculate_distance(box1, box2):
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    return math.hypot(x2_center - x1_center, y2_center - y1_center)

# ==================== ALERT / TRACKING STATE ====================
bag_classes = [24, 26, 28]  # backpack, handbag, suitcase etc
person_class = 0

# track id -> {"status": bool (with person present), "last_change": timestamp}
bag_status_time = {}

# trackers for each type to maintain stable ids
bag_tracker = CentroidTracker(max_disappeared=30, max_distance=distance_threshold * 3)
person_tracker = CentroidTracker(max_disappeared=30, max_distance=distance_threshold * 3)

# ==================== MAIN LOOP ====================
client = InferenceServerClient(url="localhost:8000")
cap = cv2.VideoCapture(video_source)
frame_no = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        # only do inference every N frames (configurable)
        if frame_no % process_every_n_frames != 0:
            # optionally show existing frame without inference
            if show_gui:
                disp = cv2.resize(frame, (720, 720))
                cv2.imshow("RT-DETR", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            continue

        # Triton inference
        input_data = preprocess_frame(frame)
        input_tensor = InferInput("images", list(input_data.shape), "FP32")
        input_tensor.set_data_from_numpy(input_data)
        results = client.infer("rtdetr", [input_tensor])

        outputs = extract_outputs_from_triton_result(results)  # list of arrays
        if len(outputs) == 0:
            detections = []
        else:
            # many rtdetrs have output[0] as [N,84]
            detections = postprocess_output(outputs[0], frame.shape)

        # Split detections by class
        people = [d for d in detections if d['class_id'] == person_class]
        bags = [d for d in detections if d['class_id'] in bag_classes]

        # Update trackers -> returns dict track_id -> bbox
        bag_tracks = bag_tracker.update([b['bbox'] for b in bags])
        person_tracks = person_tracker.update([p['bbox'] for p in people])

        # For easier distance lookups create list of person track items
        person_items = list(person_tracks.items())  # [(pid,bbox),...]
        person_centroids = {pid: ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0) for pid, b in person_items}

        # Evaluate each bag track
        for bag_track_id, bag_bbox in list(bag_tracks.items()):
            # compute nearest person
            assigned_person = None
            min_distance = float('inf')
            for pid, pbbox in person_tracks.items():
                dist = calculate_distance(bag_bbox, pbbox)
                if dist < min_distance:
                    min_distance = dist
                    assigned_person = pid
            is_unattended_now = True
            if assigned_person is not None and min_distance < distance_threshold:
                is_unattended_now = False

            now = time.time()
            if bag_track_id not in bag_status_time:
                # store True = attended (person near), False = unattended
                bag_status_time[bag_track_id] = {"status": not is_unattended_now, "last_change": now}

            prev_status = bag_status_time[bag_track_id]["status"]
            current_status = not is_unattended_now
            if prev_status != current_status:
                bag_status_time[bag_track_id]["status"] = current_status
                bag_status_time[bag_track_id]["last_change"] = now

            elapsed = now - bag_status_time[bag_track_id]["last_change"]
            final_unattended = (is_unattended_now and elapsed >= unattended_time_sec)

            # Draw
            color = (0, 255, 0) if not final_unattended else (0, 0, 255)
            label = f"BagID:{bag_track_id}"
            if final_unattended:
                label += " UNATTENDED"
                log_event(frame_no, bag_track_id, "Unattended")

            x1, y1, x2, y2 = map(int, bag_bbox)
            # clamp y for putText
            text_y = max(12, y1 - 10)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # show
        if show_gui:
            disp = cv2.resize(frame, (720, 720))
            cv2.imshow("RT-DETR", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
