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
from skimage.measure import shannon_entropy

import cv2
import numpy as np
import tifffile as tiff

# make if else for real_state
try:
    from bluesky_queueserver_api import BPlan
    from bluesky_queueserver_api.zmq import REManagerAPI
    RM = REManagerAPI()

    sys.path.insert(0,'/nsls2/data2/hxn/legacy/home/xf03id/src/hxntools')
    from hxntools.CompositeBroker import db
    from hxntools.scan_info import get_scan_positions

except ImportError:
    BPlan = None
    REManagerAPI = None
    RM = None
    print("Warning: bluesky_queueserver_api not found. Bluesky-related functionality will be disabled.")


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

    # yield from piezos_to_zero()
    
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
    dets = beamline_params.get("det_name", "dets_fast")
    #dets = [fs,eiger2,xspress3]
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
        if filename.startswith("unions_output") or filename.startswith("union_blobs"):
            continue
        if pattern.match(filename):
            continue

        json_path = os.path.join(directory_path, filename)
        print(filename)
        with open(json_path, "r") as f:
            data = json.load(f)

        for label, info in data.items():
            time.sleep(1)
            cx = info["cx"]
            cy = info["cy"] 
            sx = info["num_x"]
            sy = info["num_y"]

            # Expand scan size by 25% for padding
            pad_ratio = beamline_params.get("fine_scan_pad_ratio", 0.25)
            sx_padded = sx * (1 + pad_ratio)
            sy_padded = sy * (1 + pad_ratio)

            # Define relative scan range around center
            x_start = -sx_padded / 2
            x_end   =  sx_padded / 2
            y_start = -sy_padded / 2
            y_end   =  sy_padded / 2

            # Step counts based on padded size
            num_steps_x = int(sx_padded / step_size)
            num_steps_y = int(sy_padded / step_size)

            # ROI still centered on original center
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
                    step_size #from json
                ))

            print(f"prev ROI: {roi}")
            print(f"Fine Scan to Queue server {filename}")
            print("BPlan: recover_pos_and_scan")
            print(f"label: {label}")
            print(f"roi: {roi}")
            print(f"dets: {dets}")
            print(f"x_motor: {x_motor}")
            print(f"x_start: {x_start}")
            print(f"x_end: {x_end}")
            print(f"num_steps_x: {num_steps_x}")
            print(f"y_motor: {y_motor}")
            print(f"y_start: {y_start}")
            print(f"y_end: {y_end}")
            print(f"num_steps_y: {num_steps_y}")
            print(f"exp_t: {exp_t}")
            print(f"step_size: {step_size}")
            # print("------------------------")

        print(f"Scan from {filename} completed") 
        print()
    print("Fine scan sent\n")
    # time.sleep(2)
    # RM.queue_start()

def create_rgb_tiff(tiff_paths, output_dir, element_list, group_name=None):
    """
    Merges the first two or three element TIFFs into a single RGB TIFF file,
    and draws the union boxes on it.
    """
    if len(element_list) < 2:
        print("⚠️ Not enough elements to create an RGB TIFF (need at least 2).")
        return

    try:
        # Determine a consistent shape from the first element's tiff
        first_element = element_list[0]
        first_path = tiff_paths.get(first_element)
        if not first_path:
            print(f"⚠️ Cannot find TIFF for base element {first_element}.")
            return
        
        base_img = tiff.imread(first_path)
        target_shape = base_img.shape

        # Prepare channels based on number of elements
        if len(element_list) >= 3:
            elements_to_use = element_list[:3]
            print(f"Creating RGB TIFF from elements (R, G, B): {', '.join(elements_to_use)}")
            img_r = tiff.imread(tiff_paths[elements_to_use[0]])
            img_g = tiff.imread(tiff_paths[elements_to_use[1]])
            img_b = tiff.imread(tiff_paths[elements_to_use[2]])
        else: # 2 elements
            elements_to_use = element_list[:2]
            print(f"Creating RG TIFF from elements (R, G): {', '.join(elements_to_use)}")
            img_r = tiff.imread(tiff_paths[elements_to_use[0]])
            img_g = tiff.imread(tiff_paths[elements_to_use[1]])
            img_b = np.zeros(target_shape, dtype=base_img.dtype)

        # Resize all to target shape
        img_r = resize_if_needed(img_r, 'R channel', target_shape)
        img_g = resize_if_needed(img_g, 'G channel', target_shape)
        img_b = resize_if_needed(img_b, 'B channel', target_shape)

        # Normalize each channel to 0-255
        norm_r = cv2.normalize(np.nan_to_num(img_r), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm_g = cv2.normalize(np.nan_to_num(img_g), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm_b = cv2.normalize(np.nan_to_num(img_b), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Merge channels
        merged_rgb = cv2.merge([norm_r, norm_g, norm_b])

        # Draw union boxes
        unions_json_filename = "unions_output.json"
        if group_name:
            unions_json_filename = f"unions_output_{group_name}.json"
        unions_json_path = Path(output_dir) / unions_json_filename
        
        if unions_json_path.exists():
            merged_unions_path = process_and_save_json(unions_json_path)
            if merged_unions_path and Path(merged_unions_path).exists():
                print(f"Drawing union boxes from {merged_unions_path}...")
                with open(merged_unions_path, "r") as f:
                    unions_data = json.load(f)
                
                for union_info in unions_data.values():
                    center = union_info.get("image_center")
                    length = union_info.get("image_length")

                    if center and length:
                        x, y = center[0], center[1]
                        half_len = length / 2
                        top_left = (int(x - half_len), int(y - half_len))
                        bottom_right = (int(x + half_len), int(y + half_len))
                        cv2.rectangle(merged_rgb, top_left, bottom_right, (255, 255, 255), 1) # White box, thickness 1
            else:
                print(f"⚠️ Could not find merged unions file from {unions_json_path} to draw boxes.")
        else:
            print(f"⚠️ Could not find {unions_json_path} to draw boxes.")

        # Save the final image
        output_filename = "Union of elements.tiff"
        if group_name:
            output_filename = f"Union of elements {group_name}.tiff"
        output_path = Path(output_dir) / output_filename
        tiff.imwrite(output_path, merged_rgb)
        print(f"✅ Saved merged RGB image with boxes to: {output_path}")

    except KeyError as e:
        print(f"❌ Could not create RGB TIFF. Missing element TIFF: {e}")
    except Exception as e:
        print(f"❌ An error occurred during RGB TIFF creation: {e}")
        trackback.print_exc()


def create_all_elements_tiff(tiff_paths, output_dir, element_list, precomputed_blobs, group_name=None):
    """
    Creates a TIFF image with individual blob boxes for each element, named All_of_elements.tiff.
    The base image is an RGB composite of the first up to 3 elements.
    """
    import traceback
    from pathlib import Path
    import tifffile as tiff
    import numpy as np
    import cv2

    try:
        # --- Create a base RGB image ---
        if not element_list or not tiff_paths:
            print("⚠️ Not enough elements or TIFF paths to create an image.")
            return

        # Determine a consistent shape from the first element's tiff
        first_element = element_list[0]
        first_path = tiff_paths.get(first_element)
        if not first_path:
            print(f"⚠️ Cannot find TIFF for base element {first_element}.")
            return
        
        base_img = tiff.imread(first_path)
        target_shape = base_img.shape

        # Prepare channels based on number of elements
        if len(element_list) >= 3:
            elements_to_use = element_list[:3]
            print(f"Creating RGB base from elements (R, G, B): {', '.join(elements_to_use)}")
            img_r = tiff.imread(tiff_paths[elements_to_use[0]])
            img_g = tiff.imread(tiff_paths[elements_to_use[1]])
            img_b = tiff.imread(tiff_paths[elements_to_use[2]])
        elif len(element_list) == 2:
            elements_to_use = element_list[:2]
            print(f"Creating RG base from elements (R, G): {', '.join(elements_to_use)}")
            img_r = tiff.imread(tiff_paths[elements_to_use[0]])
            img_g = tiff.imread(tiff_paths[elements_to_use[1]])
            img_b = np.zeros(target_shape, dtype=base_img.dtype)
        else: # 1 element
            element_to_use = element_list[0]
            print(f"Creating grayscale base from element: {element_to_use}")
            img_r = tiff.imread(tiff_paths[element_to_use])
            img_g = img_r
            img_b = img_r

        # Resize all to target shape
        img_r = resize_if_needed(img_r, 'R channel', target_shape)
        img_g = resize_if_needed(img_g, 'G channel', target_shape)
        img_b = resize_if_needed(img_b, 'B channel', target_shape)

        # Normalize and merge (BGR for OpenCV drawing)
        norm_r = cv2.normalize(np.nan_to_num(img_r), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm_g = cv2.normalize(np.nan_to_num(img_g), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        norm_b = cv2.normalize(np.nan_to_num(img_b), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        merged_bgr = cv2.merge([norm_b, norm_g, norm_r])

        # --- Draw individual blob boxes ---
        color_map = {
            'red':    (0, 0, 255),   # Red
            'green':  (0, 255, 0),   # Green
            'blue':   (255, 0, 0),   # Blue
            'orange': (0, 165, 255),
            'purple': (128, 0, 128),
            'cyan':   (255, 255, 0),
            'olive':  (0, 128, 128),
            'yellow': (0, 255, 255),
            'brown':  (42, 42, 165),
            'pink':   (203, 192, 255)
        }

        print("Drawing individual element boxes...")
        for color_name, blob_data in precomputed_blobs.items():
            if color_name not in color_map:
                continue
            
            box_color = color_map[color_name]
            
            for (thresh, area, max_area), blobs in blob_data.items():
                for blob in blobs:
                    x = blob.get('box_x')
                    y = blob.get('box_y')
                    size = blob.get('box_size')

                    if x is not None and y is not None and size is not None:
                        top_left = (int(x), int(y))
                        bottom_right = (int(x + size), int(y + size))
                        cv2.rectangle(merged_bgr, top_left, bottom_right, box_color, 2)

        # --- Save the final image ---
        merged_rgb_for_save = cv2.cvtColor(merged_bgr, cv2.COLOR_BGR2RGB)
        output_filename = "All_of_elements.tiff"
        if group_name:
            output_filename = f"All_of_elements {group_name}.tiff"
        output_path = Path(output_dir) / output_filename
        tiff.imwrite(str(output_path), merged_rgb_for_save)
        print(f"✅ Saved image with individual boxes to: {output_path}")

    except KeyError as e:
        print(f"❌ Could not create image. Missing element TIFF: {e}")
    except Exception as e:
        print(f"❌ An error occurred during image creation: {e}")
        traceback.print_exc()

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

#merge boxes option 

# ---------- helpers ----------
def box_area(top_left, bottom_right):
    w = bottom_right[0] - top_left[0]
    h = bottom_right[1] - top_left[1]
    return max(0, w) * max(0, h)

def intersection_area(box1, box2):
    x1 = max(box1["real_top_left_um"][0], box2["real_top_left_um"][0])
    y1 = max(box1["real_top_left_um"][1], box2["real_top_left_um"][1])
    x2 = min(box1["real_bottom_right_um"][0], box2["real_bottom_right_um"][0])
    y2 = min(box1["real_bottom_right_um"][1], box2["real_bottom_right_um"][1])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)

def boxes_overlap(box1, box2, overlap_thresh=0.5):
    inter_area = intersection_area(box1, box2)
    if inter_area <= 0:
        return False
    area1 = box_area(box1["real_top_left_um"], box1["real_bottom_right_um"])
    area2 = box_area(box2["real_top_left_um"], box2["real_bottom_right_um"])
    smaller_area = min(area1, area2)
    return (inter_area / smaller_area) >= overlap_thresh

def compute_px_per_um(box):
    """Derive px/um from any one box that contains both image_length and real_size_um."""
    if "image_length" in box and "real_size_um" in box:
        # image_length is pixel side; real_size_um is [w_um, h_um]
        real_len_um = max(float(box["real_size_um"][0]), float(box["real_size_um"][1]))
        img_len_px  = float(box["image_length"])
        if real_len_um > 0:
            return img_len_px / real_len_um
    return None

def add_compatibility_keys(box):
    """Ensure 'center', 'length', 'area' keys exist (pixel-based) and duplicate real_area key."""
    # area key duplication for safety with both encodings
    if "real_area_um²" in box and "real_area_um\u00b2" not in box:
        box["real_area_um\u00b2"] = box["real_area_um²"]
    if "real_area_um\u00b2" in box and "real_area_um²" not in box:
        box["real_area_um²"] = box["real_area_um\u00b2"]

    # provide px/um if computable from the box itself
    px_per_um = compute_px_per_um(box)
    if px_per_um is not None:
        box["px_per_um"] = px_per_um  # optional, can be handy later

    # center (pixels)
    if "center" not in box:
        if "image_center" in box:
            box["center"] = box["image_center"]
        elif px_per_um is not None and "real_center_um" in box:
            rc = box["real_center_um"]
            box["center"] = [int(round(rc[0] * px_per_um)), int(round(rc[1] * px_per_um))]

    # length (pixels)
    if "length" not in box:
        if "image_length" in box:
            box["length"] = box["image_length"]
        elif px_per_um is not None and "real_size_um" in box:
            sx_um, sy_um = box["real_size_um"]
            box["length"] = int(round(max(sx_um, sy_um) * px_per_um))

    # area (pixels^2)
    if "area" not in box:
        if "image_area_px²" in box:
            box["area"] = box["image_area_px²"]
        elif "length" in box:
            L = int(round(box["length"]))
            box["area"] = int(L * L)

    return box

# ---------- merging ----------
def merge_boxes_strict(box1, box2, new_label):
    """Merge two boxes -> union in real units, then recalc image fields via px_per_um if available."""
    # union in real coordinates
    x1 = min(box1["real_top_left_um"][0],  box2["real_top_left_um"][0])
    y1 = min(box1["real_top_left_um"][1],  box2["real_top_left_um"][1])
    x2 = max(box1["real_bottom_right_um"][0], box2["real_bottom_right_um"][0])
    y2 = max(box1["real_bottom_right_um"][1], box2["real_bottom_right_um"][1])

    size_x_um = x2 - x1
    size_y_um = y2 - y1
    center_um = [(x1 + x2) / 2, (y1 + y2) / 2]

    merged = {
        "text": new_label,
        "real_top_left_um": [x1, y1],
        "real_bottom_right_um": [x2, y2],
        "real_center_um": center_um,
        "real_size_um": [size_x_um, size_y_um],
        "real_area_um²": size_x_um * size_y_um,
        "merged_from": [box1.get("text", ""), box2.get("text", "")]
    }
    # duplicate area key with \u00b2 for robustness
    merged["real_area_um\u00b2"] = merged["real_area_um²"]

    # Try to get a px/um from either input
    px_per_um = compute_px_per_um(box1) or compute_px_per_um(box2)

    if px_per_um is not None:
        size_x_px = int(round(size_x_um * px_per_um))
        size_y_px = int(round(size_y_um * px_per_um))
        center_px = [int(round(center_um[0] * px_per_um)),
                     int(round(center_um[1] * px_per_um))]
        merged["image_center"]   = center_px
        merged["image_length"]   = int(max(size_x_px, size_y_px))
        merged["image_area_px²"] = int(size_x_px * size_y_px)
        merged["px_per_um"]      = float(px_per_um)

    # add shorthand compatibility keys
    return add_compatibility_keys(merged)

def merge_overlapping_boxes_dict(data: dict, overlap_thresh=0.5) -> dict:
    """
    Repeatedly merge overlapping boxes; recalc real+image geometry;
    add compatibility keys ('center','length','area').
    """
    boxes = list(data.values())
    merged_any = True
    counter = 1

    while merged_any:
        merged_any = False
        new_boxes = []
        used = set()

        for i in range(len(boxes)):
            if i in used:
                continue
            current = boxes[i]
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                if boxes_overlap(current, boxes[j], overlap_thresh):
                    current = merge_boxes_strict(current, boxes[j], f"Merged Box #{counter}")
                    used.add(j)
                    merged_any = True
            used.add(i)
            new_boxes.append(current)
            counter += 1
        boxes = new_boxes

    # Ensure non-merged boxes also have compat keys
    boxes = [add_compatibility_keys(b) for b in boxes]

    return {f"Final Box #{i+1}": b for i, b in enumerate(boxes)}


# ---------------- File wrapper ----------------
def process_and_save_json(input_path, overlap_thresh=0.5):
    """Load JSON file, merge overlapping boxes, save as *_merged.json."""
    with open(input_path, "r") as f:
        data = json.load(f)

    merged = merge_overlapping_boxes_dict(data, overlap_thresh=overlap_thresh)

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_merged.json"

    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"✅ Merged JSON saved to: {output_path}")
    return output_path





def detect_blobs_(img_norm, img_orig, min_thresh, min_area, color, file_name):
    # --- Threshold the image to binary ---
    _, binary = cv2.threshold(img_norm, min_thresh, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)


    # --- Find external contours ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    height, width = img_orig.shape[:2]

    for idx, cnt in enumerate(contours, start=1):
        area = cv2.contourArea(cnt)
        if area < min_area or area > 200:  # Replace 100 with a max_area if needed
            continue

        # Bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        radius = int(max(w, h) / 2)
        box_size = 2 * radius
        box_x, box_y = cx - radius, cy - radius

        # Clip to image dimensions
        x1, y1 = max(0, box_x), max(0, box_y)
        x2, y2 = min(width, cx + radius), min(height, cy + radius)

        roi_orig = img_orig[y1:y2, x1:x2]
        roi_dilated = img_norm[y1:y2, x1:x2]

        if roi_orig.size > 0:
            blobs.append({
                'Box': f"{file_name} Box #{idx}",
                'center': (cx, cy),
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

def detect_blobs(img_norm, img_orig, min_thresh, min_area, max_area, color, file_name):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = min_thresh
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    params.thresholdStep = 2  #Default was 10

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
        'red', 'green', 'blue', 'orange', 'purple',
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
    params_dict = {item[0]: item[1] for item in json_items if isinstance(item, list) and len(item) == 2}
    dilation_size = int(params_dict.get("dialaiton_size", 5))
    dilation_iterations = int(params_dict.get("dialation_iteration", 3))
    for idx, tiff_name in enumerate(selected_tiffs):
        color = COLOR_ORDER[idx]
        tiff_path = watch_dir / tiff_name
        try:
            tiff_img = tiff.imread(str(tiff_path)).astype(np.float32)
            norm, dilated = normalize_and_dilate(tiff_img, dilation_size, dilation_iterations)
            threshold = json_items[0][1]
            min_area = json_items[1][1]
            max_area = json_items[2][1]
            blobs = detect_blobs(dilated, norm, threshold, min_area, max_area, color, tiff_name)
            precomputed_blobs[color][(threshold, min_area, max_area)] = blobs
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

def is_featureless(img):
    img = np.nan_to_num(img)
    ent = shannon_entropy(img)
    pnr = (img.max() - img.mean()) / (img.std() + 1e-5)
    edge_map = cv2.Canny(img.astype(np.uint8), 50, 150)
    edge_ratio = np.count_nonzero(edge_map) / img.size

    return (ent < 2.5) and (pnr < 2.5) and (edge_ratio < 0.01)



def normalize_and_dilate_(img, kernel_size=(3, 3), iterations=3, blur_kernel=(3, 3),):
    img = np.nan_to_num(img)

    if is_featureless(img):
        print("[normalize_and_dilate] Skipped — no signal detected (entropy+pnr+edges)")
        return np.zeros_like(img, dtype=np.uint8), np.zeros_like(img, dtype=np.uint8)

    if blur_kernel:
        blurred = cv2.GaussianBlur(img, blur_kernel, 0)
    else:
        blurred = img.copy()
    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    dilated = cv2.dilate(norm, kernel, iterations=iterations)
    return norm, dilated

def normalize_and_dilate(img, dilation_size=5, dilation_iterations=3):
    img = np.nan_to_num(img)

    if is_featureless(img):
        print("[normalize_and_dilate] Skipped — no signal detected (entropy+pnr+edges)")
        return np.zeros_like(img, dtype=np.uint8), np.zeros_like(img, dtype=np.uint8)
    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated = cv2.dilate(norm, kernel, iterations=dilation_iterations)
    return norm, dilated

def boxes_intersect(b1, b2):
    x1_min, y1_min = b1['box_x'], b1['box_y']
    x1_max, y1_max = x1_min + b1['box_size'], y1_min + b1['box_size']

    x2_min, y2_min = b2['box_x'], b2['box_y']
    x2_max, y2_max = x2_min + b2['box_size'], y2_min + b2['box_size']

    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

def union_box_dimensions(*blobs):
    """
    Computes the union box of multiple blobs using their box_x, box_y, and box_size.
    The union box is defined by the min bottom-left and max top-right corners.
    Returns:
        center (tuple): (x, y) of union box center
        length (float): side length of union box
        area (float): area of union box
    """
    if not blobs:
        return (0, 0), 0.0, 0.0
    # bottom-left corners
    bl_x = [b['box_x'] for b in blobs]
    bl_y = [b['box_y'] for b in blobs]
   
    # top-right corners
    tr_x = [b['box_x'] + b['box_size'] for b in blobs]
    tr_y = [b['box_y'] + b['box_size'] for b in blobs]
   
    # union box bounds
    min_x = min(bl_x)
    min_y = min(bl_y)
    max_x = max(tr_x)
    max_y = max(tr_y)
   
    # center of union box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
   
    # side length and area
    width = max_x - min_x
    height = max_y - min_y
    length = max(width, height)  # make it square
    area = length * length
   
    return (center_x, center_y), float(length), float(area)


def union_center(*blobs):
    """
    Computes the center of the union box of multiple blobs.
    Uses the union_box_dimensions function to avoid repeating logic.
    """
    center, _, _ = union_box_dimensions(*blobs)
    return center

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

    def create_union_obj(blobs_to_union):
        nonlocal union_index
        cx, cy = union_center(*blobs_to_union)
        _, length, area = union_box_dimensions(*blobs_to_union)
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
            # Original fields (used by formatter)
            'center': [cx, cy],
            'length': length,
            'area': area,

            # Alias for compatibility with merge logic
            'image_center': [cx, cy],
            'image_length': length,
            'image_area_px²': area,

            # Real-world
            'real_center_um': [real_cx, real_cy],
            'real_size_um': [real_length_x, real_length_y],
            'real_area_um\u00b2': real_area,
            'real_top_left_um': list(real_top_left),
            'real_bottom_right_um': list(real_bottom_right),
        }

        union_objects[union_index] = union_obj
        union_index += 1

    if reds and greens and blues:
        for r in reds:
            for g in greens:
                if not boxes_intersect(r, g):
                    continue
                for b in blues:
                    if boxes_intersect(r, b) and boxes_intersect(g, b):
                        create_union_obj([r, g, b])
    elif reds and greens:
        for r in reds:
            for g in greens:
                if boxes_intersect(r, g):
                    create_union_obj([r, g])

    return union_objects


def find_union_blobs_(blobs, microns_per_pixel_x, microns_per_pixel_y, true_origin_x, true_origin_y):
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

                    # union_obj = {
                    #     'center': (cx, cy),
                    #     'length': length,
                    #     'area': area,
                    #     'real_center_um': (real_cx, real_cy),
                    #     'real_size_um': (real_length_x, real_length_y),
                    #     'real_area_um\u00b2': real_area,
                    #     'real_top_left_um': real_top_left,
                    #     'real_bottom_right_um': real_bottom_right
                    # }

                    union_obj = {
                            # Original (pixel-space)
                            'center': (cx, cy),
                            'length': length,
                            'area': area,

                            # Synonyms for downstream merger + formatter
                            'image_center': [cx, cy],
                            'image_length': length,
                            'image_area_px²': area,

                            # Real-world units
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

# def fly2d_qserver_scan_export(label,
#                            dets,
#                            mot1, mot1_s, mot1_e, mot1_n,
#                            mot2, mot2_s, mot2_e, mot2_n,
#                            exp_t,
#                            roi_positions=None,
#                            scan_id=None,
#                            zp_move_flag=1,
#                            smar_move_flag=1,
#                            ic1_count=55000,
#                            # **POST-SCAN EXPORTS**
#                            elem_list=None,           # list of elements for XRF
#                            export_norm='sclr1_ch4',  # channel to normalize by
#                            data_wd='.',              # where to write TIFFs
#                            pos_save_to=None):        # JSON filename or dir
#     """
#     1) Optionally recover a previous scan or ROI dict
#     2) Do beam/flux checks
#     3) Run fly2dpd
#     4) Export XRF-ROI data TIFFs
#     5) Save final ROI positions JSON
#     """
#     print(f"{label} starting…")
#     RE.md["scan_name"] = str(label)

#     # — 1) RECOVERY —
#     moved = False
#     # If a valid scan_id is provided (truthy), recover from that scan

#     if scan_id:
#         yield from recover_zp_scan_pos(scan_id,
#                                        zp_move_flag=zp_move_flag,
#                                        smar_move_flag=smar_move_flag,
#                                        move_base=1)
#         moved = True

#     # Else if ROI positions dict/string provided, and not all values None
#     elif roi_positions:
#         if isinstance(roi_positions, str):
#             roi_positions = json.loads(roi_positions)
#         # Filter out keys with None values
#         non_null = {k: v for k, v in roi_positions.items() if v is not None}
#         if non_null:
#             for key, val in non_null.items():
#                 if key != "zp.zpz1":
#                     # yield from bps.mov(eval(key), val)
#                 else:
#                     # yield from mov_zpz1(val)
#                 print(f"  → {key} @ {val:.3f}")
#             # yield from check_for_beam_dump(threshold=5000)
#             if sclr2_ch2.get() < ic1_count * 0.9:
#                 # yield from peak_the_flux()
#             moved = True

#     if not moved:
#         print("[RECOVERY] no ROI recovery requested; skipping motor moves.")

#     # — 2) FLY SCAN —
#     yield from fly2dpd(dets,
#                        mot1, mot1_s, mot1_e, mot1_n,
#                        mot2, mot2_s, mot2_e, mot2_n,
#                        exp_t)
#     # produce a zmq message with scan id?

#     # — 3) POST-SCAN EXPORTS —
#     hdr = db[-1]
#     last_id = hdr.start["scan_id"]
#     print(f"[POST] exporting XRF ROI data for scan {last_id}…")
#     export_xrf_roi_data(last_id,
#                         norm=export_norm,
#                         elem_list=elem_list or [],
#                         wd=data_wd)

#     if pos_save_to:
#         print(f"[POST] saving ROI positions JSON to {pos_save_to}…")
#         export_scan_params(sid=last_id, zp_flag=True, save_to=pos_save_to)

#     print("[POST] done.")


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

    print("Performing Coarse scan")
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
    print("Coarse scan done\n")

def wait_for_queue_done(poll_interval=5.0, idle_timeout=60, auto_restart=True):
    """
    Wait until QServer queue is empty and manager is idle.
    Optionally restart the queue if stuck in idle with items remaining.

    Args:
        poll_interval (float): Seconds between polls.
        idle_timeout (float): How long to wait in idle with items before triggering restart.
        auto_restart (bool): If True, will automatically call RM.start_queue() after timeout.
    """
    import time

    print("[WAIT] polling queue status...", end="", flush=True)
    idle_stuck_start = None

    while True:
        st = RM.status()
        items = st.get("items_in_queue", 0)
        state = st.get("manager_state", "")

        if items == 0 and state == "idle":
            print(" done.")
            return

        if items > 0 and state == "idle":
            if idle_stuck_start is None:
                idle_stuck_start = time.time()
            elif time.time() - idle_stuck_start > idle_timeout:
                if auto_restart:
                    print("\n⚠️ Queue is idle with items still in queue.")
                    print("🔁 Automatically restarting queue with RM.start_queue()...")
                    RM.start_queue()
                else:
                    print("\n⚠️ Queue is idle with items still in queue.")
                    print("🔁 Consider running: RM.start_queue() to resume.")
                return
        else:
            idle_stuck_start = None  # reset if queue becomes active again

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
    print()

    valid_keys = inspect.signature(send_fly2d_to_queue).parameters.keys()
    clean_params = {k: v for k, v in params.items() if k in valid_keys}
    print(f" {clean_params = }")
    print()

    time.sleep(5)

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
        last_id = 341431 # Use a dummy scan ID for testing 

    out_dir = os.path.join(data_wd, f"automap_{last_id}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[EXPORT] saving all outputs to {out_dir}")

    print(f"ID being used {last_id}")

    all_elem_list = params.get('elem_list', [])

    for elem_list in all_elem_list:

        # 4) export XRF TIFFs
        export_xrf_roi_data(
            last_id,
            norm=params.get('export_norm', 'sclr1_ch4'),
            elem_list=elem_list,
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

    # Read the step_size from the just-created params file
    params_json_path = os.path.join(out_dir, f"scan_{last_id}_params.json")
    if os.path.exists(params_json_path):
        print(f"Reading params from: {params_json_path}")
        with open(params_json_path, 'r') as f:
            params_data = json.load(f)
            step_size = params_data.get('step_size')
            print(f"Step size from params file: {step_size}")
            # Get scan_input and extract x_start and y_start
            scan_input = params_data.get('start_doc', {}).get('scan', {}).get('scan_input', [])
            if len(scan_input) >= 4:
                x_start = scan_input[0]
                y_start = scan_input[3]
                print(f"x_start from params file: {x_start}")
                print(f"y_start from params file: {y_start}")

    elem_list_of_lists = params.get("elem_list", [])
    if not elem_list_of_lists:
        print("elem_list is empty, nothing to process.")
        return

    if isinstance(elem_list_of_lists[0], str):
        elem_list_of_lists = [elem_list_of_lists]

    all_elements = sorted(list(set(elem for sublist in elem_list_of_lists for elem in sublist)))
    tiff_paths = wait_for_element_tiffs(all_elements, out_dir)

    COLOR_ORDER = [
        'red', 'green', 'blue', 'orange', 'purple', 'cyan', 'olive', 'yellow', 'brown', 'pink'
    ]
    precomputed_blobs = {color: {} for color in COLOR_ORDER}

    min_thresh = params.get("min_threshold_intensity", "")
    min_area = params.get("min_threshold_area", "")
    max_area = params.get("max_threshold_area", 1600)
    dilation_size = int(params.get("dialaiton_size", 5))
    dilation_iterations = int(params.get("dialation_iteration", 3))
    microns_per_pixel_x = step_size
    microns_per_pixel_y = step_size
    true_origin_x = x_start
    true_origin_y = y_start

    element_to_color = {element: COLOR_ORDER[i] for i, element in enumerate(all_elements) if i < len(COLOR_ORDER)}

    for element in all_elements:
        if element not in tiff_paths:
            print(f"Skipping element {element} as its TIFF file is missing.")
            continue
        
        color = element_to_color.get(element)
        if not color:
            print(f"Skipping element {element} as it has no assigned color.")
            continue

        tiff_path = tiff_paths[element]
        print(f"Processing {tiff_path.name} for element {element} as color {color}")
        try:
            tiff_img = tiff.imread(str(tiff_path)).astype(np.float32)
            tiff_norm, tiff_dilated = normalize_and_dilate(tiff_img, dilation_size, dilation_iterations)
            b = detect_blobs(
                tiff_dilated,
                tiff_norm,
                min_thresh,
                min_area,
                max_area,
                color,
                tiff_path.name
            )
            precomputed_blobs[color][(min_thresh, min_area, max_area)] = b
        except Exception as e:
            print(f"❌ Error processing {tiff_path.name}: {e}")
            trackback.print_exc()

    for elem_list in elem_list_of_lists:
        group_name = "".join(elem_list)
        
        group_dir = os.path.join(out_dir, group_name)
        os.makedirs(group_dir, exist_ok=True)

        print(f"\n--- Processing element group: {group_name} ({elem_list}) ---")
        print(f"Group output directory: {group_dir}")

        group_blobs_for_union = {}
        for i, element in enumerate(elem_list):
            if i >= 3: # find_union_blobs supports 3 elements
                print(f"Warning: element group has more than 3 elements. Only first 3 will be used for union: {elem_list}")
                break
            
            original_color = element_to_color.get(element)
            if not original_color or not precomputed_blobs.get(original_color):
                print(f"Warning: No blobs found for element {element} in group {group_name}. Skipping it in union.")
                continue

            new_color = ['red', 'green', 'blue'][i]
            group_blobs_for_union[new_color] = precomputed_blobs[original_color]

        if len(group_blobs_for_union) >= 2:
            unions = find_union_blobs(
                group_blobs_for_union,
                microns_per_pixel_x,
                microns_per_pixel_y,
                true_origin_x,
                true_origin_y
            )

            unions = merge_overlapping_boxes_dict(unions, overlap_thresh=0.5)

            formatted_unions = {}
            for idx, union in unions.items():
                box_name = f"Union Box {group_name} #{idx.split('#')[-1].strip()}"
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

            output_path = Path(group_dir) / f"unions_output_{group_name}.json"
            with open(output_path, "w") as f:
                json.dump(formatted_unions, f, indent=2)
            print(f"\n✅ Union data for {group_name} saved to: {output_path}\n")
            #print(formatted_unions)

            save_each_blob_as_individual_scan(formatted_unions, group_dir)

            print("Perform Fine scan now\n")
            headless_send_queue_fine_scan(group_dir, params, last_id, params.get('real_test', 0))
        
        elif len(group_blobs_for_union) == 1:
                    element_name = elem_list[0]
                    print(f"Processing single element: {element_name}")
                    
                    # There's only one color, so just get the first item
                    color, blob_data = next(iter(group_blobs_for_union.items()))
                    
                    formatted_blobs = {}
                    blob_index = 1
                    
                    for (thresh, area, max_area), blobs in blob_data.items():
                        for blob in blobs:
                            cx, cy = blob['center']
                            length = blob['box_size']
                            area = length * length
                            top_left_x = blob['box_x']
                            top_left_y = blob['box_y']
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

                            box_name = f"Blob {element_name} #{blob_index}"
                            formatted = {
                                "text": box_name,
                                "image_center": [cx, cy],
                                "image_length": length,
                                "image_area_px²": area,
                                "real_center_um": [real_cx, real_cy],
                                "real_size_um": [real_length_x, real_length_y],
                                "real_area_um²": real_area,
                                "real_top_left_um": list(real_top_left),
                                "real_bottom_right_um": list(real_bottom_right)
                            }
                            formatted_blobs[box_name] = formatted
                            blob_index += 1

                    if formatted_blobs:
                        output_path = Path(group_dir) / f"unions_output_{group_name}.json"
                        with open(output_path, "w") as f:
                            json.dump(formatted_blobs, f, indent=2)
                        print(f"\n✅ Blob data for {group_name} saved to: {output_path}\n")

                        save_each_blob_as_individual_scan(formatted_blobs, group_dir)

                        print("Perform fine scan now\n")
                        headless_send_queue_fine_scan(group_dir, params, last_id, params.get('real_test', 0))

        if tiff_paths:
            group_blobs_for_all_elements = {}
            for i, element in enumerate(elem_list):
                if i >= len(COLOR_ORDER): break
                original_color = element_to_color.get(element)
                if original_color and original_color in precomputed_blobs:
                    new_color = COLOR_ORDER[i]
                    group_blobs_for_all_elements[new_color] = precomputed_blobs[original_color]

            create_rgb_tiff(tiff_paths, group_dir, elem_list, group_name)
            create_all_elements_tiff(tiff_paths, group_dir, elem_list, group_blobs_for_all_elements, group_name)
    

    print("[DONE] all exports complete.")
    time.sleep(2)

    # st = RM.status()
    # if st['items_in_queue'] != 0 and st['manager_state'] == 'idle':

    #     RM.queue_start()
    #     print('[QSERVER] queue started')

    # else: print('[QSERVER] queue waiting')
    # wait_for_queue_done()


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



# def mosaic_overlap_scan_auto(dets = None, ylen = 100, xlen = 100, overlap_per = 5, dwell = 0.01,
#                         step_size = 250, plot_elem = ["Cr"],mll = False, beamline_params=None, initial_scan_path=None):
    

#     """ Usage <mosaic_overlap_scan([fs, xspress3, eiger2], dwell=0.01, plot_elem=['Au_L'], mll=True)"""

#     if dets is None:
#         dets = dets_fast

#     i0_init = sclr2_ch2.get()

#     max_travel = 25

#     dsx_i = dsx.position
#     dsy_i = dsy.position

#     smarx_i = smarx.position
#     smary_i = smary.position

#     scan_dim = max_travel - round(max_travel*overlap_per*0.01)

#     x_tile = round(xlen/scan_dim)
#     y_tile = round(ylen/scan_dim)

#     xlen_updated = scan_dim*x_tile
#     ylen_updated = scan_dim*y_tile

#     #print(f"{xlen_updated = }, {ylen_updated=}")


#     X_position = np.linspace(0,xlen_updated-scan_dim,x_tile)
#     Y_position = np.linspace(0,ylen_updated-scan_dim,y_tile)

#     X_position_abs = smarx.position+(X_position)
#     Y_position_abs = smary.position+(Y_position)

#     #print(X_position_abs)
#     #print(Y_position_abs)


#     #print(X_position)
#     #print(Y_position)

#     print(f"{xlen_updated = }")
#     print(f"{ylen_updated = }")
#     print(f"# of x grids = {x_tile}")
#     print(f"# of y grids = {y_tile}")
#     print(f"individual grid size in um = {scan_dim} x {scan_dim}")

#     num_steps = round(max_travel*1000/step_size)

#     unit = "minutes"
#     fly_time = (num_steps**2)*dwell*2
#     num_flys= len(X_position)*len(Y_position)
#     total_time = (fly_time*num_flys)/60


#     if total_time>60:
#         total_time/=60
#         unit = "hours"

#     ask = input(f"Optimized scan x and y range = {xlen_updated} by {ylen_updated};\n total time = {total_time} {unit}\n Do you wish to continue? (y/n) ")

#     if ask == 'y':

#         time.sleep(2)
#         first_sid = db[-1].start["scan_id"]+1

#         if sclr2_ch2.get() < i0_init*0.9:
#             yield from peak_the_flux()

#         if mll:

#             yield from bps.movr(dsy, ylen_updated/-2)
#             yield from bps.movr(dsx, xlen_updated/-2)
#             X_position_abs = dsx.position+(X_position)
#             Y_position_abs = dsy.position+(Y_position)


#         else:
#             yield from bps.movr(smary, ylen_updated/-2)
#             yield from bps.movr(smarx, xlen_updated/-2)
#             X_position_abs = smarx.position+(X_position)
#             Y_position_abs = smary.position+(Y_position)

#             print(X_position_abs)
#             print(Y_position_abs)


#         for i in tqdm.tqdm(Y_position_abs):
#                 for j in tqdm.tqdm(X_position_abs):
#                     print((i,j))
#                     #yield from check_for_beam_dump(threshold=5000)
#                     yield from bps.sleep(1) #cbm catchup time

#                     fly_dim = scan_dim/2

#                     if mll:

#                         print(i,j)

#                         yield from bps.mov(dsy, i)
#                         yield from bps.mov(dsx, j)
#                         # yield from fly2dpd(dets,dssx,-1*fly_dim,fly_dim,num_steps,dssy,-1*fly_dim,fly_dim,num_steps,dwell)
#                         yield from headless_send_queue_coarse_scan(beamline_params, initial_scan_path, 1)

#                         yield from bps.sleep(3)
#                         yield from bps.mov(dssx,0,dssy,0)
#                         #insert_xrf_map_to_pdf(-1,plot_elem,'dsx')
#                         yield from bps.mov(dsx, dsx_i)
#                         yield from bps.mov(dsy,dsy_i)

#                     else:
#                         print(f"{fly_dim = }")
#                         yield from bps.mov(smary, i)
#                         yield from bps.mov(smarx, j)
#                         # yield from fly2dpd(dets, zpssx,-1*fly_dim,fly_dim,num_steps,zpssy, -1*fly_dim,fly_dim,num_steps,dwell)
#                         yield from headless_send_queue_coarse_scan(beamline_params, initial_scan_path, 1)

#                         yield from bps.sleep(1)
#                         yield from bps.mov(zpssx,0,zpssy,0)

#                         #try:
#                             #insert_xrf_map_to_pdf(-1,plot_elem[0],'smarx')
#                         #except:
#                             #plt.close()
#                             #pass


#                         yield from bps.mov(smarx, smarx_i)
#                         yield from bps.mov(smary,smary_i)

#         save_page()

#         # plot_mosiac_overlap(grid_shape = (y_tile,x_tile),
#         #                     first_scan_num = int(first_sid),
#         #                     elem = plot_elem[0],
#         #                     show_scan_num = True)

#     else:
#         return