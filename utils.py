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

import cv2
import numpy as np
import tifffile as tiff

from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QLineEdit, QCheckBox, QSlider, QFileDialog, QListWidget, QListWidgetItem,
    QFrame, QMessageBox, QDoubleSpinBox, QProgressBar, QScrollArea, QSizePolicy,
    QGraphicsEllipseItem
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect, QTimer

import globals
from globals import update_boxes, union_box_drawer, get_current_blobs

def send_json_boxes_to_queue_with_center_move(json_file_path, dets="dets1", x_motor="zpssx", y_motor="zpssy", exp_t=0.01, px_per_um=1.25, file_save_path="queued_regions.json"):
    """
    For each region in the JSON file:
    - Move stage to real_center_um
    - Perform fly2d scan centered on that position
    """
    with open(json_file_path, "r") as f:
        boxes = json.load(f)

    queued_data = {}

    for label, info in boxes.items():
        cx, cy = info["real_center_um"]         # center in um
        sx, sy = info["real_size_um"]           # size in um
 
        # Define relative scan range around center
 
        x_start = cx - sx / 2
        x_end = cx + sx / 2
        y_start = cy - sy / 2
        y_end = cy + sy / 2

        # # Detector names
        # det_names = [d.name for d in eval(dets)]
 
        # # Create ROI dictionary to move motors first
        # roi = {x_motor: cx, y_motor: cy}
        # RM.item_add(BPlan(
        #     "recover_pos_and_scan",
        #     label,
        #     roi,
        #     det_names,
        #     x_motor,
        #     x_start,
        #     x_end,
        #     sx,
        #     y_motor,
        #     y_start,
        #     y_end,
        #     sy,
        #     exp_t
        # ))
        # print(f"Queued: {label} | center ({cx:.1f}, {cy:.1f}) µm | size ({sx:.1f}, {sy:.1f}) µm")
        # Add to export dictionary
        queued_data[label] = {
            "center_um": [round(cx, 2), round(cy, 2)],
            "size_um": [round(sx, 2), round(sy, 2)],
        }
    # Save metadata to a JSON file
    with open(file_save_path, "w") as f_out:
        json.dump(queued_data, f_out, indent=4)

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

def headless_send_queue_coarse_scan(output_dir, beamline_params):
    """
    Performs coarse scan using only parameters from beamline_params.
    The output directory path is constructed and can be used later.
    No JSON files are read in this function.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    filepath = output_dir  # This can be used later as needed
    print(f"JSON path: {filepath}")

    dets = beamline_params.get("det_name", "dets1")
    x_motor = beamline_params.get("mot1", "zpssx")
    y_motor = beamline_params.get("mot2", "zpssy")
    mot1_n = beamline_params.get("mot1_n", 100)
    mot2_n = beamline_params.get("mot2_n", 100)

    x_start = beamline_params.get("mot1_s", 0)
    x_end = beamline_params.get("mot1_e", 0)
    y_start = beamline_params.get("mot2_s", 0)
    y_end = beamline_params.get("mot2_e", 0)

    # Calculate center as midpoint
    cx = (x_start + x_end) / 2
    cy = (y_start + y_end) / 2
    
    roi = {x_motor: cx, y_motor: cy}

    # RM.item_add(BPlan(
    #     "recover_pos_and_scan",
    #     "coarse_scan", # or another label
    #     roi,
    #     dets,
    #     x_motor,
    #     x_start,
    #     x_end,
    #     mot1_n,
    #     y_motor,
    #     y_start,
    #     y_end,
    #     mot2_n,
    # ))
    #The function ^ would generate the tiff files to headless_scan

    print("In future save Tiff files to the file path")
    print(f"Output directory path: {filepath}")
    print("Coarse Scan to Queue Server")
    print("\n=== Scan Parameters ===")
    print("recover_pos_and_scan")
    print(f"ROI (region of interest): {roi}")
    print(f"Detector name (dets): {dets}")
    print(f"X motor: {x_motor} (mot1), mot1_n: {mot1_n}")
    print(f"Y motor: {y_motor} (mot2), mot2_n: {mot2_n}")
    print("--- Scan Ranges ---")
    print(f"  X range: {x_start:.2f} to {x_end:.2f} µm")
    print(f"  Y range: {y_start:.2f} to {y_end:.2f} µm")
    print("------------------------\n")

def headless_send_queue_fine_scan(directory_path, beamline_params):
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
    mot1_n = beamline_params.get("mot1_n", 100)
    mot2_n = beamline_params.get("mot2_n", 100)

    exp_t = beamline_params.get("exp_t", 0.01)
    step_size = beamline_params.get("step_size", 1)

    for filename in os.listdir(directory_path):
        if not filename.endswith(".json"):
            continue

        json_path = os.path.join(directory_path, filename)
        print(f"JSON path: {json_path}")
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

            num_steps_x = sx / mot1_n  # Set x_motor to num_steps_x
            num_steps_y = sy / mot2_n
            roi = {x_motor: cx, y_motor: cy}

            # Detector names
            # det_names = [d.name for d in eval(dets)]
            # Create ROI dictionary to move motors first
            # RM.item_add(BPlan(
            #     "recover_pos_and_scan", #written
            #     label, #from folder of jsons
            #     roi, #calculated here
            #     dets, #from beamline_params
            #     x_motor, #from beamline_params
            #     x_start, #calculated here
            #     x_end, #calculated here
            #     num_steps_x, #calculated here
            #     y_motor, #from beamline_params
            #     y_start, #calculated here
            #     y_end, #calculate d here
            #     num_steps_y, #calculated here
            #     exp_t #from beamline_params
            #     step_size #from json
            # ))

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


def start_blob_computation(
    element_colors,
    thresholds_range,
    area_range,
    precomputed_blobs,
    dilated,
    img_r,
    img_g,
    img_b,
    file_names,
    progress_bar,
):
    # Prepare shared variables
    progress_lock = threading.Lock()
    
    # Flatten tasks: (color_index, threshold, area)
    task_list = [
        (i, t, a)
        for i, color in enumerate(globals.element_colors)
        for t in thresholds_range
        for a in area_range
    ]

    # Divide tasks into 4 chunks
    num_cores = multiprocessing.cpu_count()
    num_threads = num_cores #4
    chunk_size = len(task_list) // num_threads
    chunks = [task_list[i * chunk_size : (i + 1) * chunk_size] for i in range(num_threads - 1)]
    chunks.append(task_list[(num_threads - 1) * chunk_size:])  # Last chunk takes remainder
    
    # Launch threads
    threads = []
    
    for chunk in chunks:
        def thread_func(ch=chunk):
            for i, t_val, a_val in ch:
                color = globals.element_colors[i]
    
                # Ensure key exists
                with progress_lock:
                    if color not in globals.precomputed_blobs:
                        globals.precomputed_blobs[color] = {}
    
                result = detect_blobs(
                    dilated[i],
                    [img_r, img_g, img_b][i],
                    t_val,
                    a_val,
                    color,
                    globals.file_names[i]
                )
    
                with progress_lock:
                    globals.precomputed_blobs[color][(t_val, a_val)] = result
                    globals.current_iteration += 1
                    progress_bar.setValue(globals.current_iteration)
                    QApplication.processEvents()
    
        t = threading.Thread(target=thread_func)
        threads.append(t)
        t.start()
    
    # Wait for all threads
    for t in threads:
        t.join()

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

    # # --- Done ---
    # print("\n✅ Precomputed blobs:")
    # for color, data in precomputed_blobs.items():
    #     if data:
    #         print(f"{color}: {data}")

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

    # print(f"✅ Structured tooltip data saved to {json_path}")

def resize_if_needed(img, name):
        if img.shape != globals.target_shape:
            # print(f"Resizing {name} from {img.shape} → {globals.target_shape}")
            return cv2.resize(img, (globals.target_shape[1], globals.target_shape[0]), interpolation=cv2.INTER_AREA)
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
    return (sum(x_vals) // 3, sum(y_vals) // 3)

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

    return length, area
    
def union_center(b1, b2, b3):
    x_vals = [b1['center'][0], b2['center'][0], b3['center'][0]]
    y_vals = [b1['center'][1], b2['center'][1], b3['center'][1]]
    return (sum(x_vals) // 3, sum(y_vals) // 3)

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

    return length, area
            

def union_function():
        base_img = globals.graphics_view.current_qimage
        snapped_thresholds = {}
        snapped_areas = {}
        for color in globals.element_colors:
            threshold_val = globals.sliders[color].value()
            area_val = globals.area_sliders[color].value()
            snapped_thresholds[color] = round(threshold_val / 10) * 10
            snapped_areas[color] = round(area_val / 10) * 10
    
        blobs = get_current_blobs()  
    
        blobs_by_color = {color: [] for color in globals.element_colors}
        for blob in blobs:
            blobs_by_color[blob['color']].append(blob)

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
        
                        real_cx = (cx * globals.microns_per_pixel_x) + globals.true_origin_x
                        real_cy = (cy * globals.microns_per_pixel_y) + globals.true_origin_y
                        real_length_x = length * globals.microns_per_pixel_x
                        real_length_y = length * globals.microns_per_pixel_y
                        real_area = real_length_x * real_length_y
        
                        real_top_left = (
                            (top_left_x * globals.microns_per_pixel_x) + globals.true_origin_x,
                            (top_left_y * globals.microns_per_pixel_y) + globals.true_origin_y
                        )
                        real_bottom_right = (
                            (bottom_right_x * globals.microns_per_pixel_x) + globals.true_origin_x,
                            (bottom_right_y * globals.microns_per_pixel_y) + globals.true_origin_y
                        )
        
                        union_obj = {
                            'center': (cx, cy),
                            'length': length,
                            'area': area,
                            'real_center': (real_cx, real_cy),
                            'real_size': (real_length_x, real_length_y),
                            'real_area': real_area,
                            'real_top_left': real_top_left,
                            'real_bottom_right': real_bottom_right
                        }
        
                        union_objects[union_index] = union_obj
                        union_index += 1

        globals.graphics_view.union_objects = list(union_objects.values())
        globals.graphics_view.union_dict = union_objects

        # Update the label
        if globals.graphics_view.union_objects:
            globals.union_list_widget.clear()
        
            for idx, ub in union_objects.items():
                cx, cy = ub['center']
                length = ub['length']
                area = ub['area']
                real_cx, real_cy = ub['real_center']
                real_w, real_h = ub['real_size']
                real_area = ub['real_area']
                real_tl = ub['real_top_left']
                real_br = ub['real_bottom_right']
        
                item_text = (
                    f"Union Box #{idx}"
                )
        
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, idx)
        
                tooltip_text = (
                    f"<b>Union Box #{idx}</b><br>"
                    f"Center: ({cx}, {cy})<br>"
                    f"Length: {length} px<br>"
                    f"Area: {area} px²<br><br>"
                    f"Real Center: ({real_cx:.2f} µm, {real_cy:.2f} µm)<br>"
                    f"Real Size: {real_w:.2f} × {real_h:.2f} µm<br>"
                    f"Real Area: {real_area:.2f} µm²<br><br>"
                    f"Real Top-Left: ({real_tl[0]:.2f}, {real_tl[1]:.2f}) µm<br>"
                    f"Real Bottom-Right: ({real_br[0]:.2f}, {real_br[1]:.2f}) µm"
                )
                item.setToolTip(tooltip_text)
        
                globals.union_list_widget.addItem(item)

            output_path = os.path.join(globals.selected_directory, "union_blobs.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(globals.graphics_view.union_dict, f)
    
            # union_box_drawer(union_dict)
            union_box_drawer(globals.graphics_view.union_dict, base_img=base_img)
            update_boxes()
        else:
            globals.union_list_widget.clear()
            globals.union_list_widget.addItem("No triple overlaps found.")

        return union_objects

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
 