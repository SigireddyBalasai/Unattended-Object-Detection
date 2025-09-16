import cv2
import math
import time
import csv
import os
import sys
import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("UnattendedObjectDetection")


# ------------------- Triton Helpers -------------------

def get_model_output_names(client: InferenceServerClient, model_name: str):
    """
    Query Triton for model metadata and return a list of output names.
    """
    try:
        meta = client.get_model_metadata(model_name=model_name)
        outputs = meta.get('outputs', [])
        names = [o.get('name') for o in outputs if 'name' in o]
        logger.info(f"Model '{model_name}' outputs found: {names}")
        return names
    except Exception as e:
        logger.warning(f"Could not fetch model metadata for '{model_name}': {e}")
        return []


def extract_outputs_from_triton_result(result, output_names=None):
    """
    Robust extraction of Triton outputs.
    Tries result.as_numpy(name) first, then falls back to raw response.
    """
    try:
        extracted = []
        if output_names:
            for name in output_names:
                try:
                    arr = result.as_numpy(name)
                    if arr is not None:
                        arr = np.array(arr)
                        extracted.append(arr)
                        logger.debug(f"Extracted output '{name}' via as_numpy with shape {arr.shape}")
                        continue
                except Exception as e:
                    logger.debug(f"as_numpy failed for '{name}': {e}")

                # fallback: get_response
                resp = result.get_response()
                outputs = resp.get('outputs', [])
                matched = [o for o in outputs if o.get('name') == name]
                if matched:
                    out = matched[0]
                    data = np.array(out.get('data', []), dtype=np.float32)
                    shape = tuple(out.get('shape', ()))
                    if shape:
                        try:
                            data = data.reshape(shape)
                        except Exception:
                            pass
                    extracted.append(data)
                    logger.debug(f"Extracted output '{name}' via get_response fallback")
                else:
                    logger.warning(f"No output named '{name}' found in response")
                    extracted.append(np.array([]))
            return extracted

        # No names known → extract everything
        resp = result.get_response()
        outputs = resp.get('outputs', [])
        for out in outputs:
            name = out.get('name', '<unnamed>')
            try:
                arr = result.as_numpy(name)
                if arr is not None:
                    extracted.append(np.array(arr))
                    continue
            except Exception:
                pass
            data = np.array(out.get('data', []), dtype=np.float32)
            shape = tuple(out.get('shape', ()))
            if shape:
                try:
                    data = data.reshape(shape)
                except Exception:
                    pass
            extracted.append(data)
        return extracted

    except Exception as e:
        logger.error(f"Failed to extract Triton outputs: {e}")
        return []


# ------------------- Pre/Post Processing -------------------

def preprocess_frame(frame):
    """Resize and normalize frame for RT-DETR"""
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1]  # BGR → RGB
    img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def postprocess_output(output, frame_shape, conf_threshold=0.5):
    """
    Decode detections.
    Expects output ~ [N, num_classes+4]
    """
    if output is None or len(output) == 0:
        return []

    detections = []
    h, w, _ = frame_shape

    for det in output:
        if len(det) < 5:
            continue
        cx, cy, bw, bh, conf = det[:5]
        if conf < conf_threshold:
            continue
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        detections.append((x1, y1, x2, y2, conf))
    return detections


# ------------------- Main -------------------

def run_inference(video_path, output_csv="detections.csv"):
    client = InferenceServerClient(url="localhost:8001", verbose=False)

    # Check model readiness
    if not client.is_model_ready("rtdetr"):
        logger.error("Model 'rtdetr' not ready on Triton server")
        return

    output_names = get_model_output_names(client, "rtdetr")

    cap = cv2.VideoCapture(video_path)
    frame_no = 0
    detection_counts = []
    inference_times = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        # Preprocess
        input_data = preprocess_frame(frame)

        # Prepare input
        input_tensor = InferInput("images", input_data.shape, "FP32")
        input_tensor.set_data_from_numpy(input_data)

        # Inference
        inference_start = time.time()
        results = client.infer(model_name="rtdetr", inputs=[input_tensor])
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)

        # Extract outputs
        outputs = extract_outputs_from_triton_result(results, output_names)
        if len(outputs) == 0:
            logger.warning(f"Frame {frame_no}: No outputs extracted")
            continue

        detections = postprocess_output(outputs[0], frame.shape)
        detection_counts.append(len(detections))
        logger.debug(f"Frame {frame_no}: {len(detections)} detections in {inference_time:.3f}s")

    cap.release()

    # Save summary
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Detections"])
        for i, c in enumerate(detection_counts, start=1):
            writer.writerow([i, c])

    logger.info(f"Processing complete. Results saved to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model3.py <video_path>")
        sys.exit(1)

    run_inference(sys.argv[1])
