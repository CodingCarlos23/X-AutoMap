import os
import sys
import cv2
import numpy as np
import tifffile as tiff
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QCheckBox, QSlider, QFileDialog, QListWidget, QListWidgetItem, QFrame,
    QMessageBox, QDoubleSpinBox, QProgressBar, QScrollArea, QSizePolicy, QGraphicsEllipseItem
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect, QTimer
from collections import Counter
import time
import copy
import pickle
import os
import re
import json
import threading
import multiprocessing

# Store paths selected by user
img_paths = [None, None, None]  # [R, G, B]
file_names = [None, None, None]
element_colors = ["red", "green", "blue"]

graphics_view = None
controls_widget = None

microns_per_pixel_x = 0.0
microns_per_pixel_y = 0.0

true_origin_x = 0.0
true_origin_y = 0.0

def on_dir_selected():
    directory = QFileDialog.getExistingDirectory(window, "Select Directory")
    if not directory:
        return

    all_files = sorted(os.listdir(directory))  # Alphabetical (case-sensitive)
    file_list_widget.clear()
    file_paths.clear()

    for fname in all_files:
        full_path = os.path.join(directory, fname)
        if os.path.isfile(full_path):
            item = QListWidgetItem(f"{fname} ({os.path.splitext(fname)[1][1:].upper()})")
            item.setCheckState(Qt.Unchecked)
            file_list_widget.addItem(item)
            file_paths.append(full_path)

# Keep these global to track order of selection
selected_files_order = []  # store indices of items checked, in order

def update_selection():
    global selected_files_order

    # Check which items are checked currently
    checked_indices = [i for i in range(file_list_widget.count()) if file_list_widget.item(i).checkState() == Qt.Checked]

    # Find newly checked or unchecked items by comparing to stored order
    # Remove unchecked items from order
    selected_files_order = [i for i in selected_files_order if i in checked_indices]

    # Add newly checked items at the end
    for i in checked_indices:
        if i not in selected_files_order:
            if len(selected_files_order) < 3:
                selected_files_order.append(i)
            else:
                # Too many selected, uncheck this item immediately
                file_list_widget.item(i).setCheckState(Qt.Unchecked)

    # If less than 3 selected, clear img_paths and file_names partially
    for idx in range(3):
        if idx < len(selected_files_order):
            path = file_paths[selected_files_order[idx]]
            img_paths[idx] = path
            file_names[idx] = os.path.basename(path)
        else:
            img_paths[idx] = None
            file_names[idx] = None

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

def on_confirm_clicked():
    global microns_per_pixel_x
    microns_per_pixel_x = float_input_micron_x.value()
    # print(microns_per_pixel_x)
    global microns_per_pixel_y
    microns_per_pixel_y = float_input_micron_y.value()
    # print(microns_per_pixel_y)
    global true_origin_x
    true_origin_x = origin_x_input.value()
    global true_origin_y
    true_origin_y = origin_y_input.value()
    # print(true_origin_x)
    # print(true_origin_y)

    if len(selected_files_order) != 3:
        QMessageBox.warning(window, "Invalid Selection", "Please select exactly 3 items.")
        return
    QApplication.processEvents()
    init_gui()


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

def update_progress(val):
    progress_bar.setValue(val)
    QApplication.processEvents()


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
    global current_iteration
    # Prepare shared variables
    progress_lock = threading.Lock()
    
    # Flatten tasks: (color_index, threshold, area)
    task_list = [
        (i, t, a)
        for i, color in enumerate(element_colors)
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
            global current_iteration
            for i, t_val, a_val in ch:
                color = element_colors[i]
    
                # Ensure key exists
                with progress_lock:
                    if color not in precomputed_blobs:
                        precomputed_blobs[color] = {}
    
                result = detect_blobs(
                    dilated[i],
                    [img_r, img_g, img_b][i],
                    t_val,
                    a_val,
                    color,
                    file_names[i]
                )
    
                with progress_lock:
                    precomputed_blobs[color][(t_val, a_val)] = result
                    current_iteration += 1
                    progress_bar.setValue(current_iteration)
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

app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("Multi-Element Image Analyzer")
window.resize(600, 500)

main_layout = QVBoxLayout()

# Directory selector
dir_button = QPushButton("Choose Directory")
dir_button.clicked.connect(on_dir_selected)
main_layout.addWidget(dir_button)

# List of files in the selected directory
file_list_widget = QListWidget()
file_list_widget.setSelectionMode(QListWidget.NoSelection)
file_list_widget.itemChanged.connect(update_selection)
# Horizontal layout: file list on the left, confirm button on the right
file_confirm_layout = QHBoxLayout()
file_confirm_layout.addWidget(file_list_widget)

# Right side (confirm button)
right_panel = QVBoxLayout()
confirm_button = QPushButton("Confirm Selection")
confirm_button.clicked.connect(on_confirm_clicked)
right_panel.addWidget(confirm_button)
right_panel.addStretch()

float_input_micron_x  = QDoubleSpinBox()
float_input_micron_x.setPrefix("Enter X(µm) per pixel:")
float_input_micron_x.setRange(0.0, 1000.0)         # Adjust range as needed
float_input_micron_x.setSingleStep(0.1)            # Step size
float_input_micron_x.setDecimals(3)                # Number of decimal places
float_input_micron_x.setValue(1.0)                 # Default value
right_panel.addWidget(float_input_micron_x)

float_input_micron_y  = QDoubleSpinBox()
float_input_micron_y.setPrefix("Enter Y(µm) per pixel:")
float_input_micron_y.setRange(0.0, 1000.0)         # Adjust range as needed
float_input_micron_y.setSingleStep(0.1)            # Step size
float_input_micron_y.setDecimals(3)                # Number of decimal places
float_input_micron_y.setValue(1.0)                 # Default value
right_panel.addWidget(float_input_micron_y)

origin_x_input = QDoubleSpinBox()
origin_x_input.setPrefix("Origin X(µm): ")
origin_x_input.setRange(-1e6, 1e6)
origin_x_input.setDecimals(2)
origin_x_input.setValue(0.0)                 # Default value

origin_y_input = QDoubleSpinBox()
origin_y_input.setPrefix("Origin Y(µm): ")
origin_y_input.setRange(-1e6, 1e6)
origin_y_input.setDecimals(2)
origin_x_input.setValue(0.0)                 # Default value

right_panel.addWidget(origin_x_input)
right_panel.addWidget(origin_y_input)

file_confirm_layout.addLayout(right_panel)

main_layout.addLayout(file_confirm_layout)

# Track full file paths
file_paths = []

# Global list to store hoverable union boxes
global_union_boxes = []

# Global Variable for current blob detection 
current_iteration = 0

#Custom Box Counter
custom_box_number = 1

# === Defer the rest of the UI until files are selected ===
def init_gui():
    global current_iteration
    current_iteration = 0
    global precomputed_blobs
    precomputed_blobs = {}
    
    # Remove old progress bars if they exist
    for i in reversed(range(main_layout.count())):
        widget = main_layout.itemAt(i).widget()
        if isinstance(widget, QProgressBar):
            main_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()

    # for btn in buttons:
    #     btn.setEnabled(False)

    # Load and convert
    global graphics_view, controls_widget

    if graphics_view is not None:
        main_layout.removeWidget(graphics_view)
        graphics_view.setParent(None)
        graphics_view.deleteLater()
        graphics_view = None

    if controls_widget is not None:
        main_layout.removeWidget(controls_widget)
        controls_widget.setParent(None)
        controls_widget.deleteLater()
        controls_widget = None
        
    img_r, img_g, img_b = [tiff.imread(p).astype(np.float32) for p in img_paths]
    
    # Resize to majority shape
    shapes = [img_r.shape, img_g.shape, img_b.shape]
    shape_counts = Counter(shapes)
    target_shape = shape_counts.most_common(1)[0][0]
    # print(f"Target (majority) shape: {target_shape}")
    
    def resize_if_needed(img, name):
        if img.shape != target_shape:
            # print(f"Resizing {name} from {img.shape} → {target_shape}")
            return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        return img
    
    img_r = resize_if_needed(img_r, file_names[0])
    img_g = resize_if_needed(img_g, file_names[1])
    img_b = resize_if_needed(img_b, file_names[2])

    def normalize_and_dilate(img):
        img = np.nan_to_num(img)
        norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dilated = cv2.dilate(norm, np.ones((5, 5), np.uint8), iterations=3)
        return norm, dilated

    norm_dilated = [normalize_and_dilate(im) for im in [img_r, img_g, img_b]]
    normalized = [nd[0] for nd in norm_dilated]
    dilated = [nd[1] for nd in norm_dilated]

    thresholds_range = list(range(0, 256, 10))
    area_range = list(range(10, 501, 10))

    QApplication.processEvents()
    progress_bar = QProgressBar()
    progress_bar.setRange(0, len(element_colors) * len(thresholds_range) * len(area_range))
    progress_bar.setValue(0)
    progress_bar.setTextVisible(True)
    progress_bar.setFormat("Computing blobs... %p%")
    main_layout.addWidget(progress_bar)
    QApplication.processEvents() 
    progress_bar.show()
    QApplication.processEvents()
    total_iterations = len(element_colors) * len(thresholds_range) * len(area_range)
    precomputed_blobs = {}
    # print(time.strftime('%H:%M:%S'))
    
    # for i, color in enumerate(element_colors):
    #     precomputed_blobs[color] = {}
    #     for t in thresholds_range:
    #         for a in area_range:
    #             key = (t, a)
    #             result = detect_blobs(dilated[i], [img_r, img_g, img_b][i], t, a, color, file_names[i])
    #             precomputed_blobs[color][key] = result
    
    #             # Update progress
    #             current_iteration += 1
    #             progress_bar.setValue(current_iteration)
    #             QApplication.processEvents()  # Allows UI to update

    QTimer.singleShot(0, lambda: start_blob_computation(
        element_colors,
        thresholds_range,
        area_range,
        precomputed_blobs,
        dilated,
        img_r,
        img_g,
        img_b,
        file_names,
        progress_bar
    )) 
 
    os.makedirs("data", exist_ok=True)

    QApplication.processEvents()

    # Make a deep copy so original data stays untouched
    blobs_to_save = copy.deepcopy(precomputed_blobs)    
    # Add real-world info and tooltip HTML to copied blobs
    for color in blobs_to_save:
        for key in blobs_to_save[color]:
            for blob in blobs_to_save[color][key]:
                cx, cy = blob['center']
    
                # Calculate real-world values
                real_center_x = (cx * microns_per_pixel_x) + true_origin_x
                real_center_y = (cy * microns_per_pixel_y) + true_origin_y
                real_box_size_x = blob['box_size'] * microns_per_pixel_x
                real_box_size_y = blob['box_size'] * microns_per_pixel_y
                real_box_area = (blob['box_size'] ** 2) * microns_per_pixel_x * microns_per_pixel_y
    
                # Store real-world values
                blob['real_center'] = (real_center_x, real_center_y)
                blob['real_box_size'] = (real_box_size_x, real_box_size_y)
                blob['real_box_area'] = real_box_area

                real_w = blob['box_size'] * microns_per_pixel_x
                real_h = blob['box_size'] * microns_per_pixel_y
                real_cx = (cx * microns_per_pixel_x) + true_origin_x
                real_cy = (cy * microns_per_pixel_y) + true_origin_y
                real_tl_x = real_cx - real_w / 2
                real_tl_y = real_cy - real_h / 2
                real_br_x = real_cx + real_w / 2
                real_br_y = real_cy + real_h / 2

                real_w = blob['box_size'] * microns_per_pixel_x
                real_h = blob['box_size'] * microns_per_pixel_y
                real_cx = (cx * microns_per_pixel_x) + true_origin_x
                real_cy = (cy * microns_per_pixel_y) + true_origin_y
                
                # Compose tooltip HTML
                blob['tooltip_html'] = (
                    f"{blob['Box']}<br>"
                    f"Center: ({cx}, {cy})<br>"
                    f"Length: {blob['box_size']} px<br>"
                    f"Box area: {blob['box_size'] * blob['box_size']} px²<br>"
                    f"<br>"
                    f"Real Center location(µm): ({real_center_x:.2f} µm, {real_center_y:.2f} µm)<br>"
                    f"Real box size(µm): ({real_box_size_x:.2f} µm × {real_box_size_y:.2f} µm)<br>"
                    f"Real box area(µm²): {real_box_area:.2f} µm²<br>"
                    f"Real Top-Left: ({real_cx - real_w / 2:.2f}, {real_cy - real_h / 2:.2f}) µm<br>"
                    f"Real Bottom-Right: ({real_cx + real_w / 2:.2f}, {real_cy + real_h / 2:.2f}) µm<br>"
                    f"<br>"
                    f"Max intensity: {blob['max_intensity']:.3f}<br>"
                    f"Mean intensity: {blob['mean_intensity']:.3f}<br>"
                    f"Mean dilation intensity: {blob['mean_dilation']:.1f}"
                )
                
    # Save the augmented copy to disk
    with open("data/precomputed_blobs_with_real_info.pkl", "wb") as f:
        pickle.dump(blobs_to_save, f) 
        
    progress_bar.hide()
    QApplication.processEvents()

    # print(time.strftime('%H:%M:%S'))
    thresholds = {color: 100 for color in element_colors}
    area_thresholds = {color: 200 for color in element_colors}

    def get_current_blobs():
        blobs = []
        for color in element_colors:
            thresh = thresholds[color]
            area = area_thresholds[color]
            
            # Snap area to nearest available area_range value (optional, if mismatch risk exists)
            available_areas = list(range(10, 501, 10))
            snapped_area = min(available_areas, key=lambda a: abs(a - area))
            
            key = (thresh, snapped_area)
            blobs_for_color = precomputed_blobs[color].get(key, [])
            
            filtered = blobs_for_color  # no need to filter again, already done

            blobs.extend(filtered)
        return blobs
    
    merged_rgb = cv2.merge([
        cv2.normalize(img_r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.normalize(img_g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.normalize(img_b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ])

    hover_label = QLabel()
    hover_label.setWindowFlags(Qt.ToolTip)
    hover_label.hide()

    x_label = QLabel("X: 0")
    y_label = QLabel("Y: 0")

    x_micron_label = QLabel("X Real: 0")
    y_micron_label = QLabel("Y Real: 0")
    
    scene = QGraphicsScene()
    q_img = QImage(merged_rgb.data, merged_rgb.shape[1], merged_rgb.shape[0], merged_rgb.shape[1] * 3, QImage.Format_RGB888)

    class ZoomableGraphicsView(QGraphicsView):
        def __init__(self, scene, hover_label, x_label, y_label, x_micron_label, y_micron_label):
            super().__init__(scene)
            self.union_objects = []
            self.union_dict = {}
            self.current_qimage = None
            self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.setMouseTracking(True)
            self.setDragMode(QGraphicsView.NoDrag)
            self._drag_active = False
            self.hover_label = hover_label
            self.blobs = []
            self.visible_colors = set(element_colors)
            self.x_label = x_label
            self.y_label = y_label
            self.x_micron_label = x_micron_label
            self.y_micron_label = y_micron_label
            self.highlighted_union_indices = []
            self.highlight_items = []
            self.union_blobs = [...]  # Indexable list or dict


        def wheelEvent(self, event):
            cursor_pos = event.pos()
            scene_pos = self.mapToScene(cursor_pos)
            zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
            self.scale(zoom_factor, zoom_factor)
            mouse_centered = self.mapFromScene(scene_pos)
            delta = cursor_pos - mouse_centered
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                self.setDragMode(QGraphicsView.ScrollHandDrag)
                self._drag_active = True
            super().mousePressEvent(event)

        def mouseReleaseEvent(self, event):
            if event.button() == Qt.LeftButton:
                self.setDragMode(QGraphicsView.NoDrag)
                self._drag_active = False
            super().mouseReleaseEvent(event)

        def mouseMoveEvent(self, event):
            if self._drag_active:
                super().mouseMoveEvent(event)
                return
            pos = self.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            self.x_label.setText(f"X: {x}")
            self.y_label.setText(f"Y: {y}")
            self.x_micron_label.setText(f"X Real location(µm): {(x * microns_per_pixel_x) + true_origin_x:.2f}")
            self.y_micron_label.setText(f"Y Real location(µm): {(y * microns_per_pixel_y) + true_origin_y:.2f}")
            for blob in self.blobs:
                if blob['color'] not in self.visible_colors:
                    continue
                cx, cy = blob['center']
                r = blob['radius']
                
                real_w = blob['box_size'] * microns_per_pixel_x
                real_h = blob['box_size'] * microns_per_pixel_y
                real_cx = (cx * microns_per_pixel_x) + true_origin_x
                real_cy = (cy * microns_per_pixel_y) + true_origin_y

                if abs(x - cx) <= r and abs(y - cy) <= r:
                    html = (
                        f"{blob['Box']}<br>"
                        f"Center: ({cx}, {cy})<br>"
                        f"Length: {blob['box_size']} px<br>"
                        f"Box area: {blob['box_size'] * blob['box_size']} px²<br>"
                        f"<br>"
                        f"Real Center location(µm): ({(cx * microns_per_pixel_x) + true_origin_x:.2f} µm, {(cy * microns_per_pixel_y) + true_origin_y:.2f} µm)<br>"
                        f"Real box size(µm): ({blob['box_size'] * microns_per_pixel_x:.2f} µm × {blob['box_size'] * microns_per_pixel_y:.2f} µm)<br>"
                        f"Real box area(µm): ({blob['box_size']**2 * microns_per_pixel_x * microns_per_pixel_y:.2f} µm²)<br>"
                        f"Real Top-Left: ({real_cx - real_w / 2:.2f}, {real_cy - real_h / 2:.2f}) µm<br>"
                        f"Real Bottom-Right: ({real_cx + real_w / 2:.2f}, {real_cy + real_h / 2:.2f}) µm<br>"
                        f"<br>"
                        f"Max intensity: {blob['max_intensity']:.3f}<br>"
                        f"Mean intensity: {blob['mean_intensity']:.3f}<br>"
                        f"Mean dilation intensity: {blob['mean_dilation']:.1f}"
                    )
                    self.hover_label.setText(html)
                    self.hover_label.adjustSize()
                    self.hover_label.move(event.x() + 15, event.y() - 30)
                    self.hover_label.setStyleSheet(
                        f"background-color: {blob['color']}; color: white; border: 1px solid black; padding: 4px;"
                    )
                    self.hover_label.show()
                    return 
                    
            if checkboxes['union'].isChecked():  # actual union box visibility check
                for ub in global_union_boxes:
                    if ub['rect'].contains(pos.toPoint()):
                        self.hover_label.setText(ub['text'])
                        self.hover_label.adjustSize()
                        self.hover_label.move(event.x() + 15, event.y() - 30)
                        self.hover_label.setStyleSheet(
                            "background-color: white; color: black; border: 1px solid black; padding: 4px;"
                        )
                        self.hover_label.show()
                        return


            
            self.hover_label.hide()

        def update_blobs(self, blobs, visible_colors):
            self.blobs = blobs
            self.visible_colors = visible_colors

        def highlight_selected_union_boxes(self, selected_items):
            # Remove previous highlight circles
            for item in self.highlight_items:
                self.scene().removeItem(item)
            self.highlight_items.clear()
        
            for item in selected_items:
                text = item.toolTip()  # or item.toolTip() if needed
        
                # Extract pixel center from: "Center: (428, 447)"
                center_match = re.search(r"Center: \((\d+), (\d+)\)", text)
        
                # Extract length from: "Length: 18 px"
                length_match = re.search(r"Length: (\d+)\s*px", text)
        
                if center_match and length_match:
                    x = int(center_match.group(1))
                    y = int(center_match.group(2))
                    length = int(length_match.group(1))
                    radius = length / 2 + 5  # slightly larger than the box
        
                    circle = QGraphicsEllipseItem(x - radius, y - radius, radius * 2, radius * 2)
                    circle.setPen(QPen(QColor("yellow"), 2, Qt.DashLine))
                    circle.setZValue(100)
                    self.scene().addItem(circle)
                    self.highlight_items.append(circle)
  
    graphics_view = ZoomableGraphicsView(scene, hover_label, x_label, y_label, x_micron_label, y_micron_label)
    graphics_view.current_qimage = q_img
    pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(graphics_view.current_qimage))
    scene.addItem(pixmap_item)
    main_layout.addWidget(graphics_view)
    
    def redraw_boxes(blobs, selected_colors, onto_img=None):
        updated_img = onto_img or graphics_view.current_qimage.copy()
        painter = QPainter(updated_img)
        painter.setRenderHint(QPainter.Antialiasing, False)
        for blob in blobs:
            if blob['color'] in selected_colors:
                cx, cy, r = *blob['center'], blob['radius']
                painter.setPen(QPen(QColor(blob['color']), 1))
                painter.drawRect(cx - r, cy - r, 2 * r, 2 * r)
        painter.end()

        if onto_img is None:
            pixmap_item.setPixmap(QPixmap.fromImage(updated_img))  # if standalone call

    checkboxes = {}
    selected_colors = set(element_colors)

    def union_box_drawer(union_dict, base_img=None, clear_only=False):
        global global_union_boxes
        global_union_boxes = []  # Clear old ones

        valid_img = base_img or graphics_view.current_qimage

        if clear_only:
            pixmap_item.setPixmap(QPixmap.fromImage(valid_img))
            return              
            
        updated_img = valid_img.copy()
        painter = QPainter(updated_img)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(QPen(QColor('white'), 1))

        for idx, ub in union_dict.items():
            cx, cy = ub['center']
            length = ub['length']
            x = int(cx - length / 2)
            y = int(cy - length / 2)
            w = h = int(length)
    
            area = ub['area']
            real_cx, real_cy = ub['real_center']
            real_w, real_h = ub['real_size']
            real_area = ub['real_area']
            real_tl = ub['real_top_left']
            real_br = ub['real_bottom_right']
 
            # Draw box
            painter.drawRect(x, y, w, h)
    
            # Exact hover text format
            hover_text = (
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
            
            # Store for hover lookup
            global_union_boxes.append({
                'rect': QRect(x, y, w, h),
                'text': hover_text
            })
    
        painter.end()
        pixmap_item.setPixmap(QPixmap.fromImage(updated_img))
    
    # def update_boxes():
    #     nonlocal selected_colors
    #     selected_colors = {c for c, cb in checkboxes.items() if cb.isChecked()}
        
    #     # graphics_view.visible_colors = selected_colors
    #     # graphics_view.union_visible = union_checkbox.isChecked()
        
    #     blobs = [b for b in get_current_blobs() if b['color'] in selected_colors]
    #     graphics_view.update_blobs(blobs, selected_colors)
    #     redraw_boxes(blobs, selected_colors)

    #     # Conditionally draw union boxes
    #     if union_checkbox.isChecked(): 
    #         union_box_drawer(graphics_view.union_objects, base_img=graphics_view.current_qimage)
    #     else:
    #         union_box_drawer([], clear_only=True)
    #         pixmap_item.setPixmap(QPixmap.fromImage(graphics_view.current_qimage))
            
    #     hover_label.hide() 
    def update_boxes():
        nonlocal selected_colors
        selected_colors = {c for c, cb in checkboxes.items() if cb.isChecked() and c != 'union'}
    
        # Start from a shared copy of the current base image
        base_img = graphics_view.current_qimage.copy()
    
        # Get blobs for selected colors
        blobs = [b for b in get_current_blobs() if b['color'] in selected_colors]
        graphics_view.update_blobs(blobs, selected_colors)
    
        # Draw element boxes on base_img
        redraw_boxes(blobs, selected_colors, onto_img=base_img)
    
        # Conditionally draw union boxes on the same image
        if union_checkbox.isChecked():
            union_box_drawer(graphics_view.union_dict, base_img=base_img)
        else:
            pixmap_item.setPixmap(QPixmap.fromImage(base_img))  # just show element boxes
    
        hover_label.hide()

    legend_layout = QVBoxLayout()
    legend_label = QLabel("Legend")
    legend_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
    legend_layout.addWidget(legend_label)
    for i, color in enumerate(element_colors):
        cb = QCheckBox(file_names[i])
        cb.setChecked(True)
        cb.setStyleSheet(f"color: {color}")
        cb.stateChanged.connect(update_boxes)
        checkboxes[color] = cb
        legend_layout.addWidget(cb)

    union_checkbox = QCheckBox("Union Boxes")
    union_checkbox.setChecked(True)
    union_checkbox.setStyleSheet("color: black")
    union_checkbox.stateChanged.connect(update_boxes)
    legend_layout.addWidget(union_checkbox)
    checkboxes['union'] = union_checkbox


    legend_layout.addStretch()

    sliders = {}
    slider_labels = {}
    slider_layout = QHBoxLayout()

    def on_slider_change(value, color):
        snapped = round(value / 10) * 10
        snapped = max(0, min(250, snapped))
        if thresholds[color] != snapped:
            thresholds[color] = snapped
            sliders[color].blockSignals(True)
            sliders[color].setValue(snapped)
            sliders[color].blockSignals(False)
            slider_labels[color].setText(f"{checkboxes[color].text()}_threshold: {snapped}")
            update_boxes()
        
    for color in element_colors:
        i = element_colors.index(color)
        vbox = QVBoxLayout()
        label = QLabel(f"{file_names[i]}_threshold: {thresholds[color]}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(255)
        slider.setTickInterval(10)
        slider.setValue(thresholds[color])
        slider.setTickPosition(QSlider.TicksBelow)
        slider.valueChanged.connect(lambda val, c=color: on_slider_change(val, c))
        sliders[color] = slider
        slider_labels[color] = label
        vbox.addWidget(label)
        vbox.addWidget(slider)
        slider_layout.addLayout(vbox)

    area_sliders = {}
    area_slider_labels = {}
    
    def on_area_slider_change(value, color):
        valid_areas = list(range(10, 401, 10)) 
        snapped = min(valid_areas, key=lambda a: abs(a - value))
        if area_thresholds[color] != snapped:
            area_thresholds[color] = snapped
            area_sliders[color].blockSignals(True)
            area_sliders[color].setValue(snapped)
            area_sliders[color].blockSignals(False)
            area_slider_labels[color].setText(f"{checkboxes[color].text()}_min_area: {snapped}")
            update_boxes()


    area_slider_layout = QHBoxLayout()
    
    for color in element_colors:
        i = element_colors.index(color)
        vbox = QVBoxLayout()
        label = QLabel(f"{file_names[i]}_min_area: {area_thresholds[color]}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(10)
        slider.setMaximum(400)
        slider.setTickInterval(10)
        slider.setValue(area_thresholds[color])
        slider.setTickPosition(QSlider.TicksBelow)
        slider.valueChanged.connect(lambda val, c=color: on_area_slider_change(val, c))
        area_sliders[color] = slider
        area_slider_labels[color] = label
        vbox.addWidget(label)
        vbox.addWidget(slider)
        area_slider_layout.addLayout(vbox)

    exit_btn = QPushButton("Exit")
    exit_btn.clicked.connect(window.close)
    reset_btn = QPushButton("Reset View")
    reset_btn.clicked.connect(lambda: graphics_view.resetTransform())

    # def union_function():
    #     for color in element_colors:
    #         threshold_val = sliders[color].value()
    #         area_val = area_sliders[color].value()
    #         thresh_snap = round(threshold_val / 10) * 10
    #         area_snap = round(area_val / 10) * 10
    #         print(thresh_snap)
    #         print(area_snap)
    #         print(sliders)
    #         union_label.set_text("clicked")

    def union_function():
        base_img = graphics_view.current_qimage
        # Step 1: Get snapped slider values
        snapped_thresholds = {}
        snapped_areas = {}
        for color in element_colors:
            threshold_val = sliders[color].value()
            area_val = area_sliders[color].value()
            snapped_thresholds[color] = round(threshold_val / 10) * 10
            snapped_areas[color] = round(area_val / 10) * 10
    
        # print("Current snapped slider settings:")
        # for color in element_colors:
        #     print(f"{color}: threshold={snapped_thresholds[color]}, area={snapped_areas[color]}")
    
        # Step 2: Get blobs based on current threshold & area settings
        blobs = get_current_blobs()  # Assume this uses the snapped values already
    
        # Step 3: Group blobs by color
        blobs_by_color = {color: [] for color in element_colors}
        for blob in blobs:
            blobs_by_color[blob['color']].append(blob)
    
        # Step 4: Find regions where blobs from all 3 colors overlap
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
                            'real_center': (real_cx, real_cy),
                            'real_size': (real_length_x, real_length_y),
                            'real_area': real_area,
                            'real_top_left': real_top_left,
                            'real_bottom_right': real_bottom_right
                        }
        
                        union_objects[union_index] = union_obj
                        union_index += 1
        
        graphics_view.union_objects = list(union_objects.values())
        graphics_view.union_dict = union_objects

        # Update the label
        if graphics_view.union_objects:
            union_list_widget.clear()
        
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
        
                union_list_widget.addItem(item)
            
            with open("data/union_blobs.pkl", "wb") as f:
                pickle.dump(graphics_view.union_dict, f)
    
            # union_box_drawer(union_dict)
            union_box_drawer(graphics_view.union_dict, base_img=base_img)
            update_boxes()
        else:
            union_list_widget.clear()
            union_list_widget.addItem("No triple overlaps found.")

    def add_box():
        # Notify user
        QMessageBox.information(window, "Add Union Box", "Click and drag to define a new union box.")
    
        temp_state = {'start': None}
    
        def on_press(event):
            temp_state['start'] = graphics_view.mapToScene(event.pos()).toPoint()
    
        def on_release(event):
            if temp_state['start'] is None:
                return
        
            end = graphics_view.mapToScene(event.pos()).toPoint()
            start = temp_state['start']
            temp_state['start'] = None
        
            x1, y1 = start.x(), start.y()
            x2, y2 = end.x(), end.y()
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            length = max(abs(x2 - x1), abs(y2 - y1))
            area = length * length
        
            real_cx = (cx * microns_per_pixel_x) + true_origin_x
            real_cy = (cy * microns_per_pixel_y) + true_origin_y
            real_length_x = length * microns_per_pixel_x
            real_length_y = length * microns_per_pixel_y
            real_area = real_length_x * real_length_y
        
            real_top_left = (
                ((cx - length // 2) * microns_per_pixel_x) + true_origin_x,
                ((cy - length // 2) * microns_per_pixel_y) + true_origin_y
            )
            real_bottom_right = (
                ((cx + length // 2) * microns_per_pixel_x) + true_origin_x,
                ((cy + length // 2) * microns_per_pixel_y) + true_origin_y
            )
        
            new_union = {
                'center': (cx, cy),
                'length': length,
                'area': area,
                'real_center': (real_cx, real_cy),
                'real_size': (real_length_x, real_length_y),
                'real_area': real_area,
                'real_top_left': real_top_left,
                'real_bottom_right': real_bottom_right
            }
        
            # Assign next available index in the union dict
            current_union_dict = graphics_view.union_dict
            next_index = max(current_union_dict.keys(), default=0) + 1
            current_union_dict[next_index] = new_union
        
            # Update both object list and dictionary
            graphics_view.union_objects = list(current_union_dict.values())
            graphics_view.union_dict = current_union_dict

            # Add new item to QListWidget with consistent style/tooltip
            ub = new_union  # The most recently added one
            idx = next_index
            cx, cy = ub['center']
            length = ub['length']
            area = ub['area']
            real_cx, real_cy = ub['real_center']
            real_w, real_h = ub['real_size']
            real_area = ub['real_area']
            real_tl = ub['real_top_left']
            real_br = ub['real_bottom_right']

            global custom_box_number

            item_text = f"Custom Box #{custom_box_number}"

            
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, custom_box_number)
            
            tooltip_text = (
                f"<b>Custom Box #{custom_box_number}</b><br>"
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

            custom_box_number = custom_box_number + 1
            
            union_list_widget.addItem(item)
  
            with open("data/union_blobs.pkl", "wb") as f:
                pickle.dump(graphics_view.union_dict, f)
                
            # Restore original mouse events and update display
            graphics_view.mousePressEvent = original_mouse_press
            graphics_view.mouseReleaseEvent = original_mouse_release
            update_boxes()
    
        # Temporarily override mouse events
        original_mouse_press = graphics_view.mousePressEvent
        original_mouse_release = graphics_view.mouseReleaseEvent
        graphics_view.mousePressEvent = on_press
        graphics_view.mouseReleaseEvent = on_release

    def on_union_item_selected():
        selected_items = union_list_widget.selectedItems()
        graphics_view.highlight_selected_union_boxes(selected_items)


    def send_to_list():
        # Collect all existing item texts in queue_server_list
        existing_texts = {queue_server_list.item(i).text() for i in range(queue_server_list.count())}
    
        for item in union_list_widget.selectedItems():
            if item.text() not in existing_texts:
                new_item = QListWidgetItem(item.text())
                new_item.setToolTip(item.toolTip())
                new_item.setData(Qt.UserRole, item.data(Qt.UserRole))  # preserve index if needed
                queue_server_list.addItem(new_item)

    def send_to_queue_server():
        data = {}
        for i in range(queue_server_list.count()):
            item = queue_server_list.item(i)
            text = item.text()
            tooltip = item.toolTip()
            index = item.data(Qt.UserRole)
    
            # Extract "Union Box #X" part as the key
            key = text.split("|")[0].strip()  # e.g., "Union Box #1"
    
            data[key] = {
                "text": text,
                "tooltip": tooltip,
                "index": index  # optional
            }

        json_safe_data = make_json_serializable(data)

        with open("data/selected_blobs_to_queue_server.json", "w") as f:
            json.dump(json_safe_data, f, indent=2)

        structure_blob_tooltips("data/selected_blobs_to_queue_server.json")
        
        queue_server_list.clear()
        queue_server_list.addItem("✅ Data sent and saved")

    def get_elements_list():    
        # Collect all visible blobs across elements
        union_list_widget.clear()
        all_blobs = get_current_blobs()
        for blob in all_blobs:
            cx, cy = blob['center']
            box_size = blob['box_size']

            real_cx = (cx * microns_per_pixel_x) + true_origin_x
            real_cy = (cy * microns_per_pixel_y) + true_origin_y
            real_length_x = box_size * microns_per_pixel_x
            real_length_y = box_size * microns_per_pixel_y
            real_area = real_length_x * real_length_y
    
            # HTML tooltip
            html = (
                f"{blob['Box']}<br>"
                f"Center: ({cx}, {cy})<br>"
                f"Length: {box_size} px<br>"
                f"Box area: {box_size * box_size} px²<br>"
                f"<br>"
                f"Real Center location (µm): ({real_cx:.2f} µm, {real_cy:.2f} µm)<br>"
                f"Real box size (µm): ({real_length_x:.2f} µm × {real_length_y:.2f} µm)<br>"
                f"Real box area (µm²): ({real_area:.2f} µm²)<br>"
                f"Real Top-Left: ({real_cx - real_length_x / 2:.2f}, {real_cy - real_length_y / 2:.2f}) µm<br>"
                f"Real Bottom-Right: ({real_cx + real_length_x / 2:.2f}, {real_cy + real_length_y / 2:.2f}) µm<br>"
                f"<br>"
                f"Max intensity: {blob['max_intensity']:.3f}<br>"
                f"Mean intensity: {blob['mean_intensity']:.3f}<br>"
                f"Mean dilation intensity: {blob['mean_dilation']:.1f}"
            )
    
            # List item
            title = blob['Box']
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, blob)  # Store full blob data if needed later
            item.setData(Qt.ToolTipRole, html)  # So we can show it in a custom hover
    
            union_list_widget.addItem(item)

    def clear_queue_server_list():    
        queue_server_list.clear()

        
    add_btn = QPushButton("Add custom box")
    add_btn.clicked.connect(add_box)
    
    union_btn = QPushButton("Get current unions")
    union_btn.clicked.connect(union_function)

    union_list_widget = QListWidget()
    union_list_widget.setSelectionMode(QListWidget.MultiSelection)  # Allow multiple selection
    union_list_widget.setMinimumHeight(500)
    union_list_widget.itemSelectionChanged.connect(on_union_item_selected)
    union_list_widget.setLineWidth(1)
    union_list_widget.setStyleSheet("""
        QListWidget {
            border: 1px solid #aaa;
            border-radius: 4px;
            background-color: #fdfdfd;
        }
        QListWidget::item {
            padding: 8px;
            border-bottom: 1px solid #ccc;  /* ← This creates the line */
        }
        QListWidget::item:selected {
            background-color: #007acc;
            color: white;
            font-weight: bold;
            border: 1px solid #005b9f;
        }
        QListWidget::item:hover {
            background-color: #e0f0ff;
        }
    """)
    union_list_widget.setMouseTracking(True)  # Required for hover without clicking
    union_list_widget.setEnabled(True)

    queue_server_list = QListWidget()
    queue_server_list.setMinimumHeight(500)
    queue_server_list.setLineWidth(1)

    send_to_list_btn = QPushButton("Add to Queue server List")
    send_to_list_btn.clicked.connect(send_to_list)

    send_elements_to_list_btn = QPushButton("Add all individual element scans to list")
    send_elements_to_list_btn.clicked.connect(get_elements_list)

    send_to_queue_server_btn = QPushButton("Send to Queue server List")    
    send_to_queue_server_btn.clicked.connect(send_to_queue_server)

    clear_queue_server_btn = QPushButton("Clear")    
    clear_queue_server_btn.clicked.connect(clear_queue_server_list)

    union_list_layout = QVBoxLayout()
    union_list_layout.addWidget(union_list_widget)
    union_list_layout.addWidget(send_to_list_btn)
    union_list_layout.addWidget(send_elements_to_list_btn)
    union_list_layout.addWidget(union_btn)
    union_list_layout.addWidget(add_btn)

    queue_server_list_layout = QVBoxLayout()
    queue_server_list_layout.addWidget(queue_server_list)
    queue_server_list_layout.addWidget(send_to_queue_server_btn)
    queue_server_list_layout.addWidget(clear_queue_server_btn)

    dual_list_layout = QHBoxLayout()
    dual_list_layout.addLayout(union_list_layout)
    dual_list_layout.addLayout(queue_server_list_layout)

    controls = QVBoxLayout()
    controls.addWidget(exit_btn)
    controls.addWidget(reset_btn)
    
    controls.addLayout(dual_list_layout)
    # controls.addWidget(union_list_widget)
    controls.addWidget(send_to_list_btn)
    controls.addWidget(send_to_queue_server_btn)
    controls.addLayout(legend_layout)
    controls.addLayout(slider_layout)
    controls.addLayout(area_slider_layout)
    controls.addWidget(x_label)
    controls.addWidget(y_label)
    controls.addWidget(x_micron_label)
    controls.addWidget(y_micron_label)

    layout = QHBoxLayout()
    layout.addWidget(graphics_view)
    side_panel = QWidget()
    side_panel.setLayout(controls)
    controls_widget = side_panel
    layout.addWidget(side_panel)

    main_layout.addLayout(layout)
    hover_label.setParent(window)

    blobs = get_current_blobs()
    graphics_view.update_blobs(blobs, selected_colors)
    redraw_boxes(blobs, selected_colors)
    window.resize(1200, 800)

window.setLayout(main_layout)
window.show()
sys.exit(app.exec_())
