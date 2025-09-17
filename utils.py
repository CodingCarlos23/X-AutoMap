import os
import sys
import re
import time
import copy
import json
import pickle
import threading
import multiprocessing
from collections import Counter
from pathlib import Path
import traceback as trackback
import inspect

import cv2
import numpy as np
import tifffile as tiff

# make if else for rea_state
try:
    from bluesky_queueserver_api import BPlan
    from bluesky_queueserver_api.zmq import REManagerAPI
    RM = REManagerAPI()
except ImportError:
    BPlan = None
    REManagerAPI = None
    RM = None
    print("Warning: bluesky_queueserver_api not found. Bluesky-related functionality will be disabled.")
#  add the db into this here 

from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QLineEdit, QCheckBox, QSlider, QFileDialog, QListWidget, QListWidgetItem,
    QFrame, QMessageBox, QDoubleSpinBox, QProgressBar, QScrollArea, QSizePolicy,
    QGraphicsEllipseItem
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect, QTimer

def save_each_blob_as_individual_scan(json_safe_data, output_dir="scans"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for idx, info in json_safe_data.items():
        cx, cy = info["real_center_um"]
        sx, sy = info["real_size_um"]

        scan_data = {
            idx: {  # Use the union box title as the key
                "cx": cx,
                "cy": cy,
                "num_x": sx,
                "num_y": sy
            }
        }

        file_path = output_dir / f"{idx}.json"
        with open(file_path, "w") as f:
            json.dump(scan_data, f, indent=4)

def headless_send_queue_coarse_scan(beamline_params, coarse_scan_path, real_test):
    """
    Performs coarse scan using only parameters from beamline_params.
    The output directory path is constructed and can be used later.
    No JSON files are read in this function.
    """ 
    dets = beamline_params.get("det_name", "dets_fast")
    x_motor = beamline_params.get("mot1", "zpssx")
    y_motor = beamline_params.get("mot2", "zpssy")

    x_start = beamline_params.get("mot1_s", 0)
    x_end = beamline_params.get("mot1_e", 0)
    y_start = beamline_params.get("mot2_s", 0)
    y_end = beamline_params.get("mot2_e", 0)

    step_size = beamline_params.get("step_size_coarse", 250)
    mot1_n = int(abs(x_end-x_start)/step_size)
    mot2_n = int(abs(y_end-y_start)/step_size)
    exp_time = beamline_params.get("exp_t_coarse", 0.01)

    # Calculate center as midpoint
    cx = (x_start + x_end) / 2
    cy = (y_start + y_end) / 2
    
    roi = {x_motor: cx, y_motor: cy}

    load_and_queue(coarse_scan_path, real_test)

def headless_send_queue_fine_scan(directory_path, beamline_params, scan_ID, real_test):
    """
    Performs fine scan for each blob in the JSON file
    Reads all JSON files in a directory. Each file should contain a single key 
    with scan parameters like cx, cy, num_x, num_y
    
    For each JSON:
    - Move stage to (cx, cy)
    - Perform fly2d scan with the specified image size and resolution
    """
    dets = beamline_params.get("det_name", "dets1")
    x_motor = beamline_params.get("mot1", "zpssx")
    y_motor = beamline_params.get("mot2", "zpssy")
    # mot1_n = beamline_params.get("mot1_n", 100)
    # mot2_n = beamline_params.get("mot2_n", 100)

    exp_t = beamline_params.get("exp_t", 0.01)
    step_size = beamline_params.get("step_size_fine", 100)

    pattern = re.compile(r"scan_\d+_params\.json$")  # matches scan_123_params.json

    for filename in os.listdir(directory_path):
        if not filename.endswith(".json"):
            continue
        if filename in ("unions_output.json", "union_blobs.json"):
            continue
        if pattern.match(filename):
            continue

        json_path = os.path.join(directory_path, filename)
        with open(json_path, "r") as f:
            data = json.load(f)

        for label, info in data.items():
            cx = info["cx"]
            cy = info["cy"] 
            sx = info["num_x"]
            sy = info["num_y"]

            # Define relative scan range around center
            x_start = -sx / 2
            x_end = sx / 2
            y_start = -sy / 2
            y_end = sy / 2

            num_steps_x = int(sx / (step_size)) # Set x_motor to num_steps_x
            num_steps_y = int(sy / (step_size))
            roi = {x_motor: cx, y_motor: cy}

            # Detector names
            # det_names = [d.name for d in eval(dets)]
            # Create ROI dictionary to move motors first

            if real_test == 1:
                RM.item_add(BPlan(
                    "recover_pos_and_scan", #written
                    label, #from folder of jsons
                    roi, #calculated here
                    dets, #from beamline_params
                    x_motor, #from beamline_params
                    x_start, #calculated here
                    x_end, #calculated here
                    num_steps_x, #calculated here
                    y_motor, #from beamline_params
                    y_start, #calculated here
                    y_end, #calculate d here
                    num_steps_y, #calculated here
                    exp_t, #from beamline_params
                    step_size, #from json
                    real_test
                ))

            if scan_ID is not None:
                roi = scan_ID
            else:
                roi = roi

            print("Fine Scan to Queue Server")
            print("\n=== Scan Parameters for JSON: {} ===".format(filename))
            print("recover_pos_and_scan")
            print(f"Label: {label}")
            print(f"ROI (region of interest): {roi}")
            print(f"Detector name (dets): {dets}")
            print(f"X motor: {x_motor} (mot1), mot1_n: {num_steps_x}")
            print(f"Y motor: {y_motor} (mot2), mot2_n: {num_steps_y}")
            print(f"Exposure time (exp_t): {exp_t}")
            print(f"Step size: {step_size}")
            print("--- Scan Ranges ---")
            print(f"  X range: {x_start:.2f} to {x_end:.2f} µm")
            print(f"  Y range: {y_start:.2f} to {y_end:.2f} µm")
            print("------------------------\n")

        print(f"Queued scan(s) from JSON: {filename}") 
        print()
    print("Fine scan is done")

def create_rgb_tiff(tiff_paths, output_dir, element_list):
    """
    Merges the first three element TIFFs into a single RGB TIFF file,
    and draws the union boxes on it.
    """
    if len(element_list) < 3:
        print("⚠️ Not enough elements to create an RGB TIFF (need at least 3).")
        return

    rgb_elements = element_list[:3]
    print(f"Creating RGB TIFF from elements (R, G, B): {rgb_elements[0]}, {rgb_elements[1]}, {rgb_elements[2]}")

    try:
        # Read the three images
        img_r = tiff.imread(tiff_paths[rgb_elements[0]])
        img_g = tiff.imread(tiff_paths[rgb_elements[1]])
        img_b = tiff.imread(tiff_paths[rgb_elements[2]])

        # Determine target shape and resize if needed
        shapes = [img.shape for img in (img_r, img_g, img_b)]
        target_shape = Counter(shapes).most_common(1)[0][0]

        img_r = resize_if_needed(img_r, rgb_elements[0], target_shape)
        img_g = resize_if_needed(img_g, rgb_elements[1], target_shape)
        img_b = resize_if_needed(img_b, rgb_elements[2], target_shape)

        # Normalize each channel to 0-255
        norm_r = cv2.normalize(np.nan_to_num(img_r), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm_g = cv2.normalize(np.nan_to_num(img_g), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm_b = cv2.normalize(np.nan_to_num(img_b), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Merge channels
        merged_rgb = cv2.merge([norm_r, norm_g, norm_b])

        # Draw union boxes
        unions_json_path = Path(output_dir) / "unions_output.json"
        if unions_json_path.exists():
            print(f"Drawing union boxes from {unions_json_path}...")
            with open(unions_json_path, "r") as f:
                unions_data = json.load(f)
            
            for union_info in unions_data.values():
                center = union_info.get("image_center")
                length = union_info.get("image_length")

                if center and length:
                    x, y = center[0], center[1]
                    half_len = length / 2
                    top_left = (int(x - half_len), int(y - half_len))
                    bottom_right = (int(x + half_len), int(y + half_len))
                    cv2.rectangle(merged_rgb, top_left, bottom_right, (255, 255, 255), 2) # White box, thickness 2
        else:
            print(f"⚠️ Could not find {unions_json_path} to draw boxes.")

        # Save the final image
        output_path = Path(output_dir) / "Union of elements.tiff"
        tiff.imwrite(output_path, merged_rgb)
        print(f"✅ Saved merged RGB image with boxes to: {output_path}")

    except KeyError as e:
        print(f"❌ Could not create RGB TIFF. Missing element TIFF: {e}")
    except Exception as e:
        print(f"❌ An error occurred during RGB TIFF creation: {e}")
        trackback.print_exc()


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(i) for i in obj)
    elif isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def detect_blobs(img_norm, img_orig, min_thresh, min_area, color, file_name):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = min_thresh
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = 50000
    params.thresholdStep = 5  #Default was 10

    params.filterByColor = False#True
    # params.blobColor = 255
    
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByConvexity = False
    params.minRepeatability = 1
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img_norm)
    blobs = []

    for idx, kp in enumerate(keypoints, start=1):  # Start from 1
        x, y = int(kp.pt[0]), int(kp.pt[1])
        radius = int(kp.size / 2)
        box_size = 2 * radius
        box_x, box_y = x - radius, y - radius

        x1, y1 = max(0, box_x), max(0, box_y)
        x2, y2 = min(img_orig.shape[1], x + radius), min(img_orig.shape[0], y + radius)
        roi_orig = img_orig[y1:y2, x1:x2]
        roi_dilated = img_norm[y1:y2, x1:x2]

        if roi_orig.size > 0:
            blobs.append({
                'Box': f"{file_name} Box #{idx}",  # <-- Your new label
                'center': (x, y),
                'radius': radius,
                'color': color,
                'file': file_name,
                'max_intensity': roi_orig.max(),
                'mean_intensity': roi_orig.mean(),
                'mean_dilation': float(roi_dilated.mean()),
                'box_x': box_x,
                'box_y': box_y,
                'box_size': box_size
            })
    return blobs

def first_scan_detect_blobs():
    COLOR_ORDER = [
        'red', 'blue', 'green', 'orange', 'purple',
        'cyan', 'olive', 'yellow', 'brown', 'pink'
    ]
    watch_dir = Path(os.getcwd())
    json_path = watch_dir / "first_scan.json"
    precomputed_blobs = {color: {} for color in COLOR_ORDER}

    # --- STEP 1: Load JSON ---
    with open(json_path, "r") as f:
        data = json.load(f)
        if isinstance(data, dict):
            json_items = [[key, value] for key, value in data.items()]
        else:
            json_items = [data]

    print("\n✅ Loaded JSON:")
    for pair in json_items:
        print(pair)

    # --- STEP 2: Get expected number of TIFF files ---
    try:
        expected_tiff_count = int(json_items[6][1])
    except (IndexError, ValueError) as e:
        print(f"❌ Error extracting expected TIFF count from JSON: {e}")
        return None

    if expected_tiff_count > len(COLOR_ORDER):
        print(f"❌ Too many TIFF files requested. Max supported: {len(COLOR_ORDER)}")
        return None

    # --- STEP 3: Wait for TIFF files ---
    print(f"\n🔍 Waiting for {expected_tiff_count} unique .tiff files...")
    while True:
        tiff_files = sorted({f for f in os.listdir(watch_dir) if f.endswith(".tiff")})
        if len(tiff_files) >= expected_tiff_count:
            selected_tiffs = tiff_files[:expected_tiff_count]
            break
        time.sleep(1)

    print("\n✅ Found required TIFF files:")
    for idx, fname in enumerate(selected_tiffs):
        print(f"{COLOR_ORDER[idx].capitalize()}: {fname}")

    # --- STEP 4: Process TIFF files ---
    for idx, tiff_name in enumerate(selected_tiffs):
        color = COLOR_ORDER[idx]
        tiff_path = watch_dir / tiff_name
        try:
            tiff_img = tiff.imread(str(tiff_path)).astype(np.float32)
            norm, dilated = normalize_and_dilate(tiff_img)
            threshold = json_items[0][1]
            min_area = json_items[1][1]
            blobs = detect_blobs(dilated, norm, threshold, min_area, color, tiff_name)
            precomputed_blobs[color][(threshold, min_area)] = blobs
        except Exception as e:
            print(f"❌ Error processing {tiff_name}: {e}")

    return precomputed_blobs

def structure_blob_tooltips(json_path):
    """
    Reads a JSON file containing blobs with HTML tooltips,
    extracts and structures the data, and writes it back to the same file.
    """
    
    def extract_numbers(s):
        """Extract all integers/floats from a string as a list."""
        return [float(x) if '.' in x else int(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", s)]

    with open(json_path, "r") as f:
        original_data = json.load(f)

    structured_data = {}

    for key, blob in original_data.items():
        text = blob.get("text", "")
        tooltip = blob.get("tooltip", "")

        fields = tooltip.replace("<b>", "").replace("</b>", "").split("<br>")
        fields = [line.strip() for line in fields if line.strip()]  # clean empty lines

        structured = {"text": text}

        for line in fields:
            if line.startswith("Center:"):
                structured["image_center"] = extract_numbers(line)
            elif "Length:" in line:
                structured["image_length"] = extract_numbers(line)[0]
            elif "Area:" in line and "px²" in line:
                structured["image_area_px²"] = extract_numbers(line)[0]
            elif "Box area:" in line:
                structured["image_area_px²"] = extract_numbers(line)[0]
            elif "Real Center location" in line or "Real Center:" in line:
                structured["real_center_um"] = extract_numbers(line)
            elif "Real box size" in line or "Real Size:" in line:
                structured["real_size_um"] = extract_numbers(line)
            elif "Real box area" in line or "Real Area:" in line:
                structured["real_area_um²"] = extract_numbers(line)[0]
            elif "Real Top-Left:" in line:
                structured["real_top_left_um"] = extract_numbers(line)
            elif "Real Bottom-Right:" in line:
                structured["real_bottom_right_um"] = extract_numbers(line)
            elif "Max intensity" in line:
                structured["max_intensity"] = extract_numbers(line)[0]
            elif "Mean intensity" in line:
                structured["mean_intensity"] = extract_numbers(line)[0]
            elif "Mean dilation intensity" in line:
                structured["mean_dilation_intensity"] = extract_numbers(line)[0]

        structured_data[key] = structured

    # Overwrite original file with structured data
    with open(json_path, "w") as f:
        json.dump(structured_data, f, indent=4)

def resize_if_needed(img, name, target_shape):
        if img.shape != target_shape:
            # print(f"Resizing {name} from {img.shape} → {target_shape}")
            return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        return img

def normalize_and_dilate(img):
    img = np.nan_to_num(img)
    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dilated = cv2.dilate(norm, np.ones((5, 5), np.uint8), iterations=3)
    return norm, dilated

def boxes_intersect(b1, b2):
    x1_min, y1_min = b1['box_x'], b1['box_y']
    x1_max, y1_max = x1_min + b1['box_size'], y1_min + b1['box_size']

    x2_min, y2_min = b2['box_x'], b2['box_y']
    x2_max, y2_max = x2_min + b2['box_size'], y2_min + b2['box_size']

    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

def union_center(b1, b2, b3):
    x_vals = [b1['center'][0], b2['center'][0], b3['center'][0]]
    y_vals = [b1['center'][1], b2['center'][1], b3['center'][1]]
    return (sum(x_vals) / 3, sum(y_vals) / 3)

def union_box_dimensions(b1, b2, b3):
    xs = [b1['box_x'], b2['box_x'], b3['box_x']]
    ys = [b1['box_y'], b2['box_y'], b3['box_y']]
    sizes = [b1['box_size'], b2['box_size'], b3['box_size']]

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(x + s for x, s in zip(xs, sizes))
    max_y = max(y + s for y, s in zip(ys, sizes))

    width = max_x - min_x
    height = max_y - min_y

    length = max(width, height)  # square side length
    area = length * length

    return float(length), float(area)
    
def union_center(b1, b2, b3):
    x_vals = [b1['center'][0], b2['center'][0], b3['center'][0]]
    y_vals = [b1['center'][1], b2['center'][1], b3['center'][1]]
    return (sum(x_vals) / 3, sum(y_vals) / 3)

def union_box_dimensions(b1, b2, b3):
    xs = [b1['box_x'], b2['box_x'], b3['box_x']]
    ys = [b1['box_y'], b2['box_y'], b3['box_y']]
    sizes = [b1['box_size'], b2['box_size'], b3['box_size']]

    min_x = min(xs)
    min_y = min(ys)
    max_x = max(x + s for x, s in zip(xs, sizes))
    max_y = max(y + s for y, s in zip(ys, sizes))

    width = max_x - min_x
    height = max_y - min_y

    length = max(width, height)  # square side length
    area = length * length

    return float(length), float(area)

def find_union_blobs(blobs, microns_per_pixel_x, microns_per_pixel_y, true_origin_x, true_origin_y):
    blobs_by_color = {color: [] for color in blobs}

    for color, blob_dict in blobs.items():
        for coord_key, blob_list in blob_dict.items():
            blobs_by_color[color].extend(blob_list)
    union_objects = {}
    union_index = 1
    reds = blobs_by_color.get('red', [])
    greens = blobs_by_color.get('green', [])
    blues = blobs_by_color.get('blue', [])
    for r in reds:
        for g in greens:
            if not boxes_intersect(r, g):
                continue
            for b in blues:
                if boxes_intersect(r, b) and boxes_intersect(g, b):
                    cx, cy = union_center(r, g, b)
                    length, area = union_box_dimensions(r, g, b)
                    top_left_x = cx - length // 2
                    top_left_y = cy - length // 2
                    bottom_right_x = top_left_x + length
                    bottom_right_y = top_left_y + length

                    real_cx = (cx * microns_per_pixel_x) + true_origin_x
                    real_cy = (cy * microns_per_pixel_y) + true_origin_y
                    real_length_x = length * microns_per_pixel_x
                    real_length_y = length * microns_per_pixel_y
                    real_area = real_length_x * real_length_y

                    real_top_left = (
                        (top_left_x * microns_per_pixel_x) + true_origin_x,
                        (top_left_y * microns_per_pixel_y) + true_origin_y
                    )
                    real_bottom_right = (
                        (bottom_right_x * microns_per_pixel_x) + true_origin_x,
                        (bottom_right_y * microns_per_pixel_y) + true_origin_y
                    )

                    union_obj = {
                        'center': (cx, cy),
                        'length': length,
                        'area': area,
                        'real_center_um': (real_cx, real_cy),
                        'real_size_um': (real_length_x, real_length_y),
                        'real_area_um\u00b2': real_area,
                        'real_top_left_um': real_top_left,
                        'real_bottom_right_um': real_bottom_right
                    }

                    union_objects[union_index] = union_obj
                    union_index += 1

    return union_objects
 
def wait_for_element_tiffs(element_list, watch_dir):
    tiff_paths = {}
    print(watch_dir)
    print("\nWaiting for TIFF files for all elements:", element_list)
    missing_reported = set()
    while True:
        all_found = True
        tiff_paths.clear()
        missing_now = set()
        for element in element_list:
            pattern = f"scan_*_{element}.tiff"
            watch_dir = Path(watch_dir)
            matches = list(watch_dir.glob(pattern))
            if matches:
                tiff_paths[element] = matches[0]
            else:
                all_found = False
                missing_now.add(element)
        # Only print for elements that are newly missing
        for element in missing_now - missing_reported:
            print(f"Waiting for TIFF file for element: {element}")
        missing_reported = missing_now
        if all_found:
            break
        time.sleep(2)
    print("\n✅ Found TIFF files for all elements:")
    for element in element_list:
        print(f"{element}: {tiff_paths[element].name}")
    return tiff_paths

def _get_flyscan_dimensions(hdr):
    start_doc = hdr.start
    # 2D_FLY_PANDA: prefer 'dimensions', fallback to 'shape'
    if 'scan' in start_doc and start_doc['scan'].get('type') == '2D_FLY_PANDA':
        if 'dimensions' in start_doc:
            return start_doc['dimensions']
        elif 'shape' in start_doc:
            return start_doc['shape']
        else:
            raise ValueError("No dimensions or shape found for 2D_FLY_PANDA scan")
    # rel_scan: use 'shape' or 'num_points'
    elif start_doc.get('plan_name') == 'rel_scan':
        if 'shape' in start_doc:
            return start_doc['shape']
        elif 'num_points' in start_doc:
            return [start_doc['num_points']]
        else:
            raise ValueError("No shape or num_points found for rel_scan")
    else:
        raise ValueError("Unknown scan type for _get_flyscan_dimensions")

def export_xrf_roi_data(scan_id, norm = 'sclr1_ch4', elem_list = [], wd = '.', real_test=0):

    if real_test == 0:
        print("[EXPORT] Skipping XRF ROI data export in test mode.")
        return

    hdr = db[int(scan_id)]
    scan_id = hdr.start["scan_id"]
   
    channels = [1, 2, 3]
    print(f"{elem_list = }")
    print(f"[DATA] fetching XRF ROIs")
    scan_dim = _get_flyscan_dimensions(hdr)
    print(f"[DATA] fetching scalar values")

    scalar = np.array(list(hdr.data(norm))).squeeze()
    print(f"[DATA] fetching scalar {norm} values done")

    for elem in sorted(elem_list):
        roi_keys = [f'Det{chan}_{elem}' for chan in channels]
        spectrum = np.sum([np.array(list(hdr.data(roi)), dtype=np.float32).squeeze() for roi in roi_keys], axis=0)
        if norm !=None:
            spectrum = spectrum/scalar
        xrf_img = spectrum.reshape(scan_dim)
        tiff.imwrite(os.path.join(wd,f"scan_{scan_id}_{elem}.tiff"), xrf_img)


def export_scan_params(sid=-1, zp_flag=True, save_to=None, real_test=0):
    """
    Fetch scan parameters, ROI positions, step size, and the full start_doc
    for scan `sid`.  Optionally write them out as JSON.

    Returns a dict with:
      - scan_id
      - start_doc
      - roi_positions
      - step_size (computed from scan_input for 2D_FLY_PANDA)
    """
    if real_test == 0:
        print("[EXPORT] Skipping scan params export in test mode.")
        return
    # 1) Pull the header
    hdr = db[int(sid)]
    start_doc = dict(hdr.start)  # cast to plain dict

    # 2) Grab the baseline table and build the ROI dict
    tbl = db.get_table(hdr, stream_name='baseline')
    row = tbl.iloc[0]
    if zp_flag:
        roi = {
            "zpssx":    float(row["zpssx"]),
            "zpssy":    float(row["zpssy"]),
            "zpssz":    float(row["zpssz"]),
            "smarx":    float(row["smarx"]),
            "smary":    float(row["smary"]),
            "smarz":    float(row["smarz"]),
            "zp.zpz1":  float(row["zpz1"]),
            "zpsth":    float(row["zpsth"]),
            "zps.zpsx": float(row["zpsx"]),
            "zps.zpsz": float(row["zpsz"]),
        }
    else:
        roi = {
            "dssx":  float(row["dssx"]),
            "dssy":  float(row["dssy"]),
            "dssz":  float(row["dssz"]),
            "dsx":   float(row["dsx"]),
            "dsy":   float(row["dsy"]),
            "dsz":   float(row["dsz"]),
            "sbz":   float(row["sbz"]),
            "dsth":  float(row["dsth"]),
        }

    # 3) Compute unified step_size from scan_input
    scan_info = start_doc.get("scan", {})
    si = scan_info.get("scan_input", [])
    if scan_info.get("type") == "2D_FLY_PANDA" and len(si) >= 3:
        fast_start, fast_end, fast_N = si[0], si[1], si[2]
        step_size = abs(fast_end - fast_start) / fast_N
    else:
        raise ValueError(f"Cannot compute step_size for scan type {scan_info.get('type')}")

    # 4) Assemble the result dict
    result = {
        "scan_id":       int(sid),
        "start_doc":     start_doc,
        "roi_positions": roi,
        "step_size":     float(step_size),
    }

    # 5) Optionally write out JSON
    if save_to:
        if os.path.isdir(save_to):
            filename = os.path.join(save_to, f"scan_{sid}_params.json")
        else:
            filename = save_to if save_to.lower().endswith(".json") else save_to + ".json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(result, f, indent=2)

    return result

def fly2d_qserver_scan_export(label,
                           dets,
                           mot1, mot1_s, mot1_e, mot1_n,
                           mot2, mot2_s, mot2_e, mot2_n,
                           exp_t,
                           roi_positions=None,
                           scan_id=None,
                           zp_move_flag=1,
                           smar_move_flag=1,
                           ic1_count=55000,
                           # **POST-SCAN EXPORTS**
                           elem_list=None,           # list of elements for XRF
                           export_norm='sclr1_ch4',  # channel to normalize by
                           data_wd='.',              # where to write TIFFs
                           pos_save_to=None):        # JSON filename or dir
    """
    1) Optionally recover a previous scan or ROI dict
    2) Do beam/flux checks
    3) Run fly2dpd
    4) Export XRF-ROI data TIFFs
    5) Save final ROI positions JSON
    """
    print(f"{label} starting…")
    RE.md["scan_name"] = str(label)

    # — 1) RECOVERY —
    moved = False
    # If a valid scan_id is provided (truthy), recover from that scan
    if scan_id:
        yield from recover_zp_scan_pos(scan_id,
                                       zp_move_flag=zp_move_flag,
                                       smar_move_flag=smar_move_flag,
                                       move_base=1)
        moved = True

    # Else if ROI positions dict/string provided, and not all values None
    elif roi_positions:
        if isinstance(roi_positions, str):
            roi_positions = json.loads(roi_positions)
        # Filter out keys with None values
        non_null = {k: v for k, v in roi_positions.items() if v is not None}
        if non_null:
            for key, val in non_null.items():
                if key != "zp.zpz1":
                    yield from bps.mov(eval(key), val)
                else:
                    yield from mov_zpz1(val)
                print(f"  → {key} @ {val:.3f}")
            yield from check_for_beam_dump(threshold=5000)
            if sclr2_ch2.get() < ic1_count * 0.9:
                yield from peak_the_flux()
            moved = True

    if not moved:
        print("[RECOVERY] no ROI recovery requested; skipping motor moves.")

    # — 2) FLY SCAN —
    yield from fly2dpd(dets,
                       mot1, mot1_s, mot1_e, mot1_n,
                       mot2, mot2_s, mot2_e, mot2_n,
                       exp_t)
    #produce a zmq message with scan id?

    # # — 3) POST-SCAN EXPORTS —
    # hdr = db[-1]
    # last_id = hdr.start["scan_id"]
    # print(f"[POST] exporting XRF ROI data for scan {last_id}…")
    # export_xrf_roi_data(last_id,
    #                     norm=export_norm,
    #                     elem_list=elem_list or [],
    #                     wd=data_wd)

    # if pos_save_to:
    #     print(f"[POST] saving ROI positions JSON to {pos_save_to}…")
    #     export_scan_params(sid=last_id, zp_flag=True, save_to=pos_save_to)

    # print("[POST] done.")


def send_fly2d_to_queue(label,
                        dets,
                        mot1, mot1_s, mot1_e, mot1_n,
                        mot2, mot2_s, mot2_e, mot2_n,
                        exp_t,
                        roi_positions=None,
                        scan_id=None,
                        zp_move_flag=1,
                        smar_move_flag=1,
                        ic1_count = 55000,
                        elem_list=None,
                        export_norm='sclr1_ch4',
                        data_wd='.',
                        pos_save_to=None,
                        real_test=0):
    # det_names = [d.name for d in eval(dets)]
    det_names = ['fs', 'eiger2', 'xspress3']

    roi_json = ""
    if isinstance(roi_positions, dict):
        roi_json = json.dumps(roi_positions)
    elif isinstance(roi_positions, str):
        roi_json = roi_positions

    print("Coarse scan")
    if real_test == 1:
        RM.item_execute(BPlan("fly2d_qserver_scan_export",
                          label,
                          det_names,
                          mot1, mot1_s, mot1_e, mot1_n,
                          mot2, mot2_s, mot2_e, mot2_n,
                          exp_t,
                          roi_json,
                          scan_id or "",
                          zp_move_flag,
                          smar_move_flag,
                          ic1_count,
                          json.dumps(elem_list or []),
                          export_norm,
                          data_wd,
                          pos_save_to or ""#,
                        #   real_test
                          ))
    print("Coarse scan done")

def wait_for_queue_done(poll_interval=2.0):
    """
    Block until the QServer queue is empty and the manager goes idle.
    """
    print("[WAIT] polling queue status...", end="", flush=True)
    while True:
        st = RM.status()
        if st['items_in_queue'] == 0 and st['manager_state'] == 'idle':
            print(" done.")
            return
        print(".", end="", flush=True)
        time.sleep(poll_interval)



def submit_and_export(**params):
    """
    Enqueue a scan, wait for completion, then export XRF TIFFs and ROI JSON
    into a single folder automap_{scan_id} under data_wd.
    """
    # 1) enqueue
    label = params.get('label', '')
    print(f"[SUBMIT] queueing scan '{label}' …")

    print(f"{params = }")

    valid_keys = inspect.signature(send_fly2d_to_queue).parameters.keys()
    clean_params = {k: v for k, v in params.items() if k in valid_keys}
    print(f" check 1")
    print(f" {clean_params = }")

    send_fly2d_to_queue(**clean_params)

    # 2) wait
    if params.get('real_test', 0) == 1:
        print("[WAIT] waiting for scan to finish…")
        while True:
            st = RM.status()
            if st['items_in_queue'] == 0 and st['manager_state'] == 'idle':
                break
            time.sleep(1.0)
        print("[WAIT] scan complete.")

    # 3) get last scan_id and prepare output folder
    data_wd = params.get('data_wd', '.')
    if params.get('real_test', 0) == 1:
        hdr = db[-1]
        last_id = hdr.start['scan_id']
    else:
        last_id = 365896 # Use a dummy scan ID for testing 

    out_dir = os.path.join(data_wd, f"automap_{last_id}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[EXPORT] saving all outputs to {out_dir}")

    print(f"ID being used {last_id}")

    # 4) export XRF TIFFs
    export_xrf_roi_data(
        last_id,
        norm=params.get('export_norm', 'sclr1_ch4'),
        elem_list=params.get('elem_list', []),
        wd=out_dir,
        real_test=params.get('real_test', 0)
    )

    # 5) export scan parameters JSON
    export_scan_params(
        sid=last_id,
        zp_flag=bool(params.get('zp_move_flag', True)),
        save_to=out_dir,
        real_test=params.get('real_test', 0)
    )

    elem_list = params.get("elem_list", "")
    tiff_paths = wait_for_element_tiffs(elem_list, out_dir)

    COLOR_ORDER = [
    'red', 'blue', 'green', 'orange', 'purple', 'cyan', 'olive', 'yellow'
    ]
    precomputed_blobs = {color: {} for color in COLOR_ORDER}

    # 
    min_thresh = params.get("min_threshold_intensity", "")
    min_area = params.get("min_threshold_area", "")
    microns_per_pixel_x = params.get("microns_per_pixel_x", "")
    microns_per_pixel_y = params.get("microns_per_pixel_y", "")
    true_origin_x = params.get("true_origin_x", "")
    true_origin_y = params.get("true_origin_y", "")

    max_tiffs = min(len(tiff_paths), 8)
    processed_elements = list(tiff_paths.keys())[:max_tiffs]

    for idx, element in enumerate(processed_elements):
        color = COLOR_ORDER[idx]
        tiff_path = tiff_paths[element]
        print(f"Processing {tiff_path.name} as color {color}")
        try:
            tiff_img = tiff.imread(str(tiff_path)).astype(np.float32)
            tiff_norm, tiff_dilated = normalize_and_dilate(tiff_img)
            b = detect_blobs(
                tiff_dilated,
                tiff_norm,
                min_thresh,
                min_area,
                color,
                tiff_path.name
            )
            precomputed_blobs[color][(min_thresh, min_area)] = b
        except Exception as e:
            print(f"❌ Error processing {tiff_path.name}: {e}")
            trackback.print_exc()

    # Only run unions if at least 2 colors are present
    if len(processed_elements) >= 2:
        unions = find_union_blobs(
            precomputed_blobs,
            microns_per_pixel_x,
            microns_per_pixel_y,
            true_origin_x,
            true_origin_y
        )

        formatted_unions = {}

        print("Processsing images now")
        for idx, union in unions.items():
            box_name = f"Union Box #{idx}"
            formatted = {
                "text": box_name,
                "image_center": union["center"],
                "image_length": union["length"],
                "image_area_px²": union["area"],
                "real_center_um": union["real_center_um"],
                "real_size_um": union["real_size_um"],
                "real_area_um²": union["real_area_um\u00b2"],
                "real_top_left_um": union["real_top_left_um"],
                "real_bottom_right_um": union["real_bottom_right_um"]
            }

            formatted_unions[box_name] = formatted

        # Save to file
        out_dir_p = Path(out_dir)  # Convert string to Path object
        output_path = out_dir_p / "unions_output.json"#"data/headless_scan/unions_output.json"
        with open(output_path, "w") as f:
            json.dump(formatted_unions, f, indent=2)
        print(f"\n✅ Union data saved to: {output_path}")
        print()

        save_each_blob_as_individual_scan(formatted_unions, out_dir)


        print("Perform fine scan now")
        headless_send_queue_fine_scan(out_dir, params, last_id, params.get('real_test', 0))

    if tiff_paths:
        create_rgb_tiff(tiff_paths, out_dir, elem_list)

    #
    print("done") 
    print("[DONE] all exports complete.")


def load_and_queue(json_path, real_test):
    """
    Load scan parameters from JSON, compute necessary fields,
    and either enqueue only or enqueue+export based on a flag.

    JSON can include an optional 'block_and_export': true to wait and post-process.
    """
    # 1) Read main params
    with open(json_path, 'r') as f:
        params = json.load(f)

    # 2) Extract blocking flag
    # block_and_export = params.pop('block_and_export', True)

    # 3) Load ROI from separate file if requested
    roi_file = params.pop('roi_positions_file', None)
    if roi_file:
        if not os.path.isfile(roi_file):
            raise FileNotFoundError(f"ROI file not found: {roi_file}")
        with open(roi_file, 'r') as rf:
            params['roi_positions'] = json.load(rf)
    elif isinstance(params.get('roi_positions'), str) and os.path.isfile(params['roi_positions']):
        with open(params['roi_positions'], 'r') as rf:
            params['roi_positions'] = json.load(rf)

    # 4) Compute mot1_n & mot2_n from a single step_size
    if 'step_size' in params:
        step = params.pop('step_size')
        params['mot1_n'] = int(abs(params['mot1_e'] - params['mot1_s']) / step)
        params['mot2_n'] = int(abs(params['mot2_e'] - params['mot2_s']) / step)

    # 5) Ensure dets is a string literal for eval()
    # if isinstance(params.get('dets'), list):
    #     params['dets'] = repr(params['dets'])

    # 6) Dispatch
    params['real_test'] = real_test
    submit_and_export(**params)



def mosaic_overlap_scan(dets = None, ylen = 100, xlen = 100, overlap_per = 15, dwell = 0.05,
                        step_size = 500, plot_elem = ["Cr"],mll = False, beamline_params=None, initial_scan_path=None):
    

    """ Usage <mosaic_overlap_scan([fs, xspress3, eiger2], dwell=0.01, plot_elem=['Au_L'], mll=True)"""

    if dets is None:
        dets = dets_fast

    max_travel = 25

    dsx_i = dsx.position
    dsy_i = dsy.position

    smarx_i = smarx.position
    smary_i = smary.position

    scan_dim = max_travel - round(max_travel*overlap_per*0.01)

    x_tile = round(xlen/scan_dim)
    y_tile = round(ylen/scan_dim)

    xlen_updated = scan_dim*x_tile
    ylen_updated = scan_dim*y_tile

    #print(f"{xlen_updated = }, {ylen_updated=}")


    X_position = np.linspace(0,xlen_updated-scan_dim,x_tile)
    Y_position = np.linspace(0,ylen_updated-scan_dim,y_tile)

    X_position_abs = smarx.position*1000+(X_position)
    Y_position_abs = smary.position*1000+(Y_position)

    #print(X_position_abs)
    #print(Y_position_abs)


    #print(X_position)
    #print(Y_position)

    print(f"{xlen_updated = }")
    print(f"{ylen_updated = }")
    print(f"# of x grids = {x_tile}")
    print(f"# of y grids = {y_tile}")
    print(f"individual grid size in um = {scan_dim} x {scan_dim}")

    num_steps = round(max_travel*1000/step_size)

    unit = "minutes"
    fly_time = (num_steps**2)*dwell*2
    num_flys= len(X_position)*len(Y_position)
    total_time = (fly_time*num_flys)/60


    if total_time>60:
        total_time/=60
        unit = "hours"

    ask = input(f"Optimized scan x and y range = {xlen_updated} by {ylen_updated};\n total time = {total_time} {unit}\n Do you wish to continue? (y/n) ")

    if ask == 'y':

        #yield from bps.sleep(10)
        first_sid = db[-1].start["scan_id"]+1

        if mll:

            yield from bps.movr(dsy, ylen_updated/-2)
            yield from bps.movr(dsx, xlen_updated/-2)
            X_position_abs = dsx.position+(X_position)
            Y_position_abs = dsy.position+(Y_position)


        else:
            yield from bps.movr(smary, ylen_updated*-0.001/2)
            yield from bps.movr(smarx, xlen_updated*-0.001/2)
            X_position_abs = smarx.position+(X_position*0.001)
            Y_position_abs = smary.position+(Y_position*0.001)

            print(X_position_abs)
            print(Y_position_abs)


        for i in tqdm.tqdm(Y_position_abs):
                for j in tqdm.tqdm(X_position_abs):
                    print((i,j))
                    #yield from check_for_beam_dump(threshold=5000)
                    yield from bps.sleep(1) #cbm catchup time

                    fly_dim = scan_dim/2

                    if mll:

                        print(i,j)

                        yield from bps.mov(dsy, i)
                        yield from bps.mov(dsx, j)
                        # yield from fly2dpd(dets,dssx,-1*fly_dim,fly_dim,num_steps,dssy,-1*fly_dim,fly_dim,num_steps,dwell)
                        headless_send_queue_coarse_scan(beamline_params, initial_scan_path)

                        yield from bps.sleep(3)
                        yield from bps.mov(dssx,0,dssy,0)
                        #insert_xrf_map_to_pdf(-1,plot_elem,'dsx')
                        yield from bps.mov(dsx, dsx_i)
                        yield from bps.mov(dsy,dsy_i)

                    else:
                        print(f"{fly_dim = }")
                        yield from bps.mov(smary, i)
                        yield from bps.mov(smarx, j)
                        # yield from fly2dpd(dets, zpssx,-1*fly_dim,fly_dim,num_steps,zpssy, -1*fly_dim,fly_dim,num_steps,dwell)
                        headless_send_queue_coarse_scan(beamline_params, initial_scan_path)

                        yield from bps.sleep(1)
                        yield from bps.mov(zpssx,0,zpssy,0)

                        #try:
                            #insert_xrf_map_to_pdf(-1,plot_elem[0],'smarx')
                        #except:
                            #plt.close()
                            #pass


                        yield from bps.mov(smarx, smarx_i)
                        yield from bps.mov(smary,smary_i)

        save_page()

        # plot_mosiac_overlap(grid_shape = (y_tile,x_tile),
        #                     first_scan_num = int(first_sid),
        #                     elem = plot_elem[0],
        #                     show_scan_num = True)

    else:
        return