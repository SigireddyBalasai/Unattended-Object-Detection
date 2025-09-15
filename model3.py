# detector.py
import cv2
import math
import time
import csv
import os
import sys
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput
import configparser
from typing import List, Dict
import logging
from datetime import datetime

# ==================== ENHANCED LOGGING SETUP ====================
def setup_logging():
    """Setup comprehensive logging system"""
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup logger
    logger = logging.getLogger('model3')
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(log_dir, f"model3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

def log_separator(title=""):
    """Log a separator line with optional title"""
    if title:
        logger.info("=" * 50)
        logger.info(f"{title:^50}")
        logger.info("=" * 50)
    else:
        logger.info("=" * 50)

def log_system_info():
    """Log system and environment information"""
    logger.info("System Information:")
    logger.info(f"  - Python version: {sys.version.split()[0]}")
    logger.info(f"  - Working directory: {os.getcwd()}")
    logger.info(f"  - Process ID: {os.getpid()}")
    
    # Check OpenCV version
    try:
        logger.info(f"  - OpenCV version: {cv2.__version__}")
    except Exception as e:
        logger.warning(f"  - OpenCV version: Unknown ({e})")
    
    # Check NumPy version
    try:
        logger.info(f"  - NumPy version: {np.__version__}")
    except Exception as e:
        logger.warning(f"  - NumPy version: Unknown ({e})")

log_separator("MODEL3 UNATTENDED OBJECT DETECTOR STARTING")
logger.info("Enhanced Model3 detector starting...")
log_system_info()

# ==================== ENHANCED CONFIG LOADING ====================
log_separator("CONFIGURATION LOADING")
logger.info("Loading configuration from ./config/config.ini...")

parser = configparser.ConfigParser()
config_file = './config/config.ini'

if not os.path.exists(config_file):
    logger.error(f"Configuration file not found: {config_file}")
    sys.exit(1)

try:
    parser.read(config_file)
    logger.info("✓ Configuration file loaded successfully")
except Exception as e:
    logger.error(f"✗ Failed to parse configuration file: {e}")
    sys.exit(1)

# Load configuration with validation
try:
    video_source = parser['DEFAULT'].get('source', 0)
    distance_threshold = parser['DEFAULT'].getint('distance_from_bag', fallback=150)
    unattended_time_sec = parser['DEFAULT'].getint('unattended_time', fallback=30)
    show_gui = parser['DEFAULT'].getboolean('show_gui', fallback=True)
    process_every_n_frames = parser['DEFAULT'].getint('process_every_n_frames', fallback=5)
    
    logger.info("Configuration parameters loaded:")
    logger.info(f"  - Video source: {video_source}")
    logger.info(f"  - Distance threshold: {distance_threshold} pixels")
    logger.info(f"  - Unattended time threshold: {unattended_time_sec} seconds")
    logger.info(f"  - Show GUI: {show_gui}")
    logger.info(f"  - Process every {process_every_n_frames} frames")
    
    # Validate configuration
    if distance_threshold <= 0:
        logger.warning("Distance threshold should be positive")
    if unattended_time_sec <= 0:
        logger.warning("Unattended time should be positive")
    if process_every_n_frames <= 0:
        logger.error("Process frame interval must be positive")
        sys.exit(1)
        
except Exception as e:
    logger.error(f"✗ Error loading configuration: {e}")
    sys.exit(1)

# ==================== ENHANCED LOGGING ====================
log_separator("LOG SYSTEM INITIALIZATION")
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
logger.info(f"Log directory created/verified: {log_dir}")

log_file = os.path.join(log_dir, "unattended_log.csv")
logger.info(f"CSV log file: {log_file}")

if not os.path.exists(log_file):
    try:
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "frame_no", "bag_track_id", "status"])
        logger.info("✓ CSV log file created with headers")
    except Exception as e:
        logger.error(f"✗ Failed to create CSV log file: {e}")
        sys.exit(1)
else:
    logger.info("✓ CSV log file already exists")

def log_event(frame_no, bag_track_id, status):
    """Enhanced event logging with error handling"""
    try:
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, frame_no, bag_track_id, status])
        logger.info(f"Event logged: Frame {frame_no}, Bag {bag_track_id}, Status: {status}")
    except Exception as e:
        logger.error(f"Failed to log event: {e}")

# ==================== ENHANCED SIMPLE CENTROID TRACKER ====================
class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=1000):
        self.next_id = 0
        self.objects = {}  # id -> bbox [x1,y1,x2,y2]
        self.disappeared = {}  # id -> frames disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        logger.info(f"CentroidTracker initialized: max_disappeared={max_disappeared}, max_distance={max_distance}")

    def _centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, rects: List[List[int]]) -> Dict[int, List[int]]:
        # rects: list of bboxes
        if len(rects) == 0:
            # increment disappeared counters
            keys = list(self.disappeared.keys())
            disappeared_count = 0
            for objectID in keys:
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    logger.debug(f"Removing disappeared object ID {objectID} (disappeared for {self.disappeared[objectID]} frames)")
                    del self.objects[objectID]
                    del self.disappeared[objectID]
                    disappeared_count += 1
            if disappeared_count > 0:
                logger.debug(f"Removed {disappeared_count} disappeared objects")
            return dict(self.objects)

        # if no existing objects, register all rects
        if len(self.objects) == 0:
            for r in rects:
                self.objects[self.next_id] = r
                self.disappeared[self.next_id] = 0
                logger.debug(f"Registered new object ID {self.next_id}")
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

        for row, col in zip(rows, cols):
            if row in assigned_rows or col in assigned_cols:
                continue
            if D[row, col] > self.max_distance:
                logger.debug(f"Distance {D[row, col]:.1f} > {self.max_distance}, not assigning")
                continue
            objectID = objectIDs[row]
            self.objects[objectID] = rects[col]
            self.disappeared[objectID] = 0
            assigned_rows.add(row)
            assigned_cols.add(col)

        # unassigned new rects -> register
        new_objects_count = 0
        for j in range(len(rects)):
            if j not in assigned_cols:
                self.objects[self.next_id] = rects[j]
                self.disappeared[self.next_id] = 0
                logger.debug(f"Registered new unassigned object ID {self.next_id}")
                self.next_id += 1
                new_objects_count += 1

        # unassigned old objects -> increase disappeared
        disappeared_objects_count = 0
        for i in range(len(objectIDs)):
            if i not in assigned_rows:
                objectID = objectIDs[i]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    logger.debug(f"Removing disappeared object ID {objectID}")
                    del self.objects[objectID]
                    del self.disappeared[objectID]
                    disappeared_objects_count += 1

        if new_objects_count > 0 or disappeared_objects_count > 0:
            logger.debug(f"Tracking update: {new_objects_count} new, {disappeared_objects_count} removed, {len(self.objects)} active")

        return dict(self.objects)

# ==================== ENHANCED TRITON HELPERS ====================
INPUT_W, INPUT_H = 640, 640  # model input size used in preprocess
logger.info(f"Model input dimensions: {INPUT_W}x{INPUT_H}")

def preprocess_frame(frame):
    """Enhanced frame preprocessing with validation"""
    if frame is None:
        logger.error("Cannot preprocess None frame")
        return None
    
    original_shape = frame.shape
    try:
        resized = cv2.resize(frame, (INPUT_W, INPUT_H))
        arr = resized.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)
        logger.debug(f"Frame preprocessed: {original_shape} -> {arr.shape}")
        return arr
    except Exception as e:
        logger.error(f"Frame preprocessing failed: {e}")
        return None

def extract_outputs_from_triton_result(result):
    """Enhanced output extraction with error handling"""
    try:
        # result.get_response() returns dict-like with 'outputs'
        # Each output entry will have 'name','datatype','shape','data' (flat list) in many triton setups
        resp = result.get_response()
        outputs = []
        for out in resp.get('outputs', []):
            data = np.array(out['data'], dtype=np.float32)
            shape = tuple(out['shape'])
            outputs.append(data.reshape(shape))
        logger.debug(f"Extracted {len(outputs)} outputs from Triton result")
        return outputs
    except Exception as e:
        logger.error(f"Failed to extract Triton outputs: {e}")
        return []

def postprocess_output(output_array, orig_shape, conf_threshold=0.3):
    """
    Enhanced postprocessing with detailed logging
    output_array: numpy [N, 84] where first 4 are bbox (x_center,y_center,w,h) normalized to input dims [0-1]
    orig_shape: original frame shape (h,w,channels)
    returns list of detections: {'bbox':[x1,y1,x2,y2],'confidence':float,'class_id':int}
    """
    detections = []
    if output_array is None:
        logger.warning("No output array provided for postprocessing")
        return detections

    orig_h, orig_w = orig_shape[:2]
    logger.debug(f"Postprocessing {len(output_array)} detections for frame {orig_w}x{orig_h}")
    
    valid_detections = 0
    for i, det in enumerate(output_array):
        bbox = det[:4]
        class_scores = det[4:]
        max_conf = float(np.max(class_scores))
        class_id = int(np.argmax(class_scores))
        
        if max_conf > conf_threshold:
            valid_detections += 1
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
    
    logger.debug(f"Found {valid_detections} valid detections above confidence {conf_threshold}")
    return detections

def calculate_distance(box1, box2):
    """Enhanced distance calculation with logging"""
    x1_center = (box1[0] + box1[2]) / 2
    y1_center = (box1[1] + box1[3]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    y2_center = (box2[1] + box2[3]) / 2
    distance = math.hypot(x2_center - x1_center, y2_center - y1_center)
    logger.debug(f"Distance calculated: {distance:.1f} pixels")
    return distance

# ==================== ALERT / TRACKING STATE ====================
bag_classes = [24, 26, 28]  # backpack, handbag, suitcase etc
person_class = 0

logger.info(f"Target classes - Bags: {bag_classes}, Person: {person_class}")

# track id -> {"status": bool (with person present), "last_change": timestamp}
bag_status_time = {}

# trackers for each type to maintain stable ids
bag_tracker = CentroidTracker(max_disappeared=30, max_distance=distance_threshold * 3)
person_tracker = CentroidTracker(max_disappeared=30, max_distance=distance_threshold * 3)

logger.info("Trackers initialized successfully")

# ==================== ENHANCED MAIN LOOP ====================
log_separator("TRITON CLIENT CONNECTION")
logger.info("Connecting to Triton Inference Server...")

try:
    client = InferenceServerClient(url="localhost:8000")
    logger.info("✓ Connected to Triton server at localhost:8000")
    
    # Test server health
    if client.is_server_ready():
        logger.info("✓ Triton server is ready")
    else:
        logger.error("✗ Triton server is not ready")
        sys.exit(1)
        
    # Check model availability
    if client.is_model_ready("rtdetr"):
        logger.info("✓ Model 'rtdetr' is ready")
    else:
        logger.error("✗ Model 'rtdetr' is not ready")
        sys.exit(1)
        
except Exception as e:
    logger.error(f"✗ Failed to connect to Triton server: {e}")
    sys.exit(1)

log_separator("VIDEO CAPTURE INITIALIZATION")
logger.info(f"Opening video source: {video_source}")

try:
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"✗ Failed to open video source: {video_source}")
        sys.exit(1)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info("✓ Video capture initialized successfully")
    logger.info(f"  - Resolution: {width}x{height}")
    logger.info(f"  - FPS: {fps:.2f}")
    logger.info(f"  - Total frames: {total_frames}")
    
except Exception as e:
    logger.error(f"✗ Video capture initialization failed: {e}")
    sys.exit(1)

# Initialize performance tracking
frame_no = 0
start_time = time.time()
inference_times = []
detection_counts = []

log_separator("MAIN PROCESSING LOOP STARTING")
logger.info("Starting main processing loop...")
logger.info(f"Processing every {process_every_n_frames} frames")
logger.info(f"GUI display: {'Enabled' if show_gui else 'Disabled'}")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream reached")
            break
        frame_no += 1

        # Progress logging
        if frame_no % 100 == 0:
            elapsed_time = time.time() - start_time
            fps_current = frame_no / elapsed_time if elapsed_time > 0 else 0
            logger.info(f"Progress: Frame {frame_no}/{total_frames} ({frame_no/total_frames*100:.1f}%) - FPS: {fps_current:.1f}")

        # only do inference every N frames (configurable)
        if frame_no % process_every_n_frames != 0:
            # optionally show existing frame without inference
            if show_gui:
                disp = cv2.resize(frame, (720, 720))
                cv2.imshow("RT-DETR", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested quit (q key pressed)")
                    break
            continue

        # Triton inference with timing
        inference_start = time.time()
        try:
            input_data = preprocess_frame(frame)
            if input_data is None:
                logger.warning(f"Frame {frame_no}: Preprocessing failed, skipping")
                continue
                
            input_tensor = InferInput("images", list(input_data.shape), "FP32")
            input_tensor.set_data_from_numpy(input_data)
            results = client.infer("rtdetr", [input_tensor])
            
            inference_time = time.time() - inference_start
            inference_times.append(inference_time)
            
            logger.debug(f"Frame {frame_no}: Inference completed in {inference_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Frame {frame_no}: Inference failed: {e}")
            continue

        # Process results
        try:
            outputs = extract_outputs_from_triton_result(results)  # list of arrays
            if len(outputs) == 0:
                detections = []
                logger.debug(f"Frame {frame_no}: No outputs from Triton")
            else:
                # many rtdetrs have output[0] as [N,84]
                detections = postprocess_output(outputs[0], frame.shape)
                detection_counts.append(len(detections))
                logger.debug(f"Frame {frame_no}: {len(detections)} detections found")

        except Exception as e:
            logger.error(f"Frame {frame_no}: Output processing failed: {e}")
            detections = []

        # Split detections by class
        people = [d for d in detections if d['class_id'] == person_class]
        bags = [d for d in detections if d['class_id'] in bag_classes]
        
        logger.debug(f"Frame {frame_no}: {len(people)} people, {len(bags)} bags detected")

        # Update trackers -> returns dict track_id -> bbox
        bag_tracks = bag_tracker.update([b['bbox'] for b in bags])
        person_tracks = person_tracker.update([p['bbox'] for p in people])

        # For easier distance lookups create list of person track items
        person_items = list(person_tracks.items())  # [(pid,bbox),...]
        person_centroids = {pid: ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0) for pid, b in person_items}

        # Evaluate each bag track
        unattended_bags = 0
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
                logger.debug(f"Frame {frame_no}: Bag {bag_track_id} attended by person {assigned_person} (distance: {min_distance:.1f})")

            now = time.time()
            if bag_track_id not in bag_status_time:
                # store True = attended (person near), False = unattended
                bag_status_time[bag_track_id] = {"status": not is_unattended_now, "last_change": now}
                logger.debug(f"Frame {frame_no}: New bag {bag_track_id} registered")

            prev_status = bag_status_time[bag_track_id]["status"]
            current_status = not is_unattended_now
            if prev_status != current_status:
                bag_status_time[bag_track_id]["status"] = current_status
                bag_status_time[bag_track_id]["last_change"] = now
                status_change = "attended" if current_status else "unattended"
                logger.info(f"Frame {frame_no}: Bag {bag_track_id} status changed to {status_change}")

            elapsed = now - bag_status_time[bag_track_id]["last_change"]
            final_unattended = (is_unattended_now and elapsed >= unattended_time_sec)

            if final_unattended:
                unattended_bags += 1
                logger.warning(f"Frame {frame_no}: Bag {bag_track_id} UNATTENDED for {elapsed:.1f}s")

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

        # Log summary for this frame
        if len(bag_tracks) > 0 or len(person_tracks) > 0:
            logger.debug(f"Frame {frame_no}: {len(bag_tracks)} bags tracked, {len(person_tracks)} people tracked, {unattended_bags} unattended")

        # show
        if show_gui:
            disp = cv2.resize(frame, (720, 720))
            cv2.imshow("RT-DETR", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User requested quit (q key pressed)")
                break

except KeyboardInterrupt:
    logger.info("Processing interrupted by user (Ctrl+C)")
except Exception as e:
    logger.error(f"Unexpected error in main loop: {e}")
    raise

finally:
    # Enhanced cleanup with performance summary
    log_separator("CLEANUP AND PERFORMANCE SUMMARY")
    
    total_time = time.time() - start_time
    logger.info("Cleaning up resources...")
    
    try:
        cap.release()
        logger.info("✓ Video capture released")
    except Exception as e:
        logger.error(f"Error releasing video capture: {e}")
    
    try:
        cv2.destroyAllWindows()
        logger.info("✓ OpenCV windows destroyed")
    except Exception as e:
        logger.error(f"Error destroying windows: {e}")
    
    # Performance summary
    logger.info("Performance Summary:")
    logger.info(f"  - Total processing time: {total_time:.2f} seconds")
    logger.info(f"  - Total frames processed: {frame_no}")
    logger.info(f"  - Average FPS: {frame_no / total_time:.2f}")
    
    if inference_times:
        avg_inference_time = sum(inference_times) / len(inference_times)
        logger.info(f"  - Inference calls: {len(inference_times)}")
        logger.info(f"  - Average inference time: {avg_inference_time:.3f}s")
        logger.info(f"  - Inference FPS: {1.0/avg_inference_time:.1f}")
    
    if detection_counts:
        avg_detections = sum(detection_counts) / len(detection_counts)
        max_detections = max(detection_counts)
        logger.info(f"  - Average detections per frame: {avg_detections:.1f}")
        logger.info(f"  - Maximum detections in a frame: {max_detections}")
    
    # Final tracking summary
    active_bags = len(bag_tracker.objects)
    active_people = len(person_tracker.objects)
    logger.info(f"  - Active bag tracks at end: {active_bags}")
    logger.info(f"  - Active person tracks at end: {active_people}")
    
    logger.info("Processing completed successfully!")
    log_separator("MODEL3 DETECTOR FINISHED")
