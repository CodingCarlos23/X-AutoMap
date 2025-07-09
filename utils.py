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

def send_json_boxes_to_queue_with_center_move(json_file_path, dets="dets1", x_motor="zpssx", y_motor="zpssy", exp_t=0.01, px_per_um=1.25):
    """
    For each region in the JSON file:
    - Move stage to real_center_um
    - Perform fly2d scan centered on that position
    """
    with open(json_file_path, "r") as f:
        boxes = json.load(f)
 
    for label, info in boxes.items():
        cx, cy = info["real_center_um"]         # center in um
        sx, sy = info["real_size_um"]           # size in um
        num_x = int(sx * px_per_um)
        num_y = int(sy * px_per_um)
 
        # Define relative scan range around center
        x_start = -sx / 2
        x_end = sx / 2
        y_start = -sy / 2
        y_end = sy / 2
 
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
        #     num_x,
        #     y_motor,
        #     y_start,
        #     y_end,
        #     num_y,
        #     exp_t
        # ))
        print(f"Queued: {label} | center ({cx:.1f}, {cy:.1f}) µm | size ({sx:.1f}, {sy:.1f}) µm")

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
