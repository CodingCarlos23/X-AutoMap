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
from utils import *
from globals import update_boxes, union_box_drawer, get_current_blobs, redraw_boxes
# from globals import *

def on_dir_selected():
    globals.selected_directory = QFileDialog.getExistingDirectory(globals.window, "Select Directory") 
    if not globals.selected_directory:
        return

    all_files = sorted(os.listdir(globals.selected_directory))  # Alphabetical (case-sensitive)
    globals.file_list_widget.clear()
    globals.file_paths.clear()

    for fname in all_files:
        full_path = os.path.join(globals.selected_directory, fname)
        ext = os.path.splitext(fname)[1].lower()  # Get file extension, lowercase
        if os.path.isfile(full_path) and ext in ['.tif', '.tiff']:
            item = QListWidgetItem(f"{fname} ({ext[1:].upper()})")
            item.setCheckState(Qt.Unchecked)
            globals.file_list_widget.addItem(item)
            globals.file_paths.append(full_path) 

def update_selection(): 
    # Check which items are checked currently
    checked_indices = [i for i in range(globals.file_list_widget.count()) if globals.file_list_widget.item(i).checkState() == Qt.Checked]

    # Find newly checked or unchecked items by comparing to stored order
    # Remove unchecked items from order
    globals.selected_files_order = [i for i in globals.selected_files_order if i in checked_indices]

    # Add newly checked items at the end
    for i in checked_indices:
        if i not in globals.selected_files_order:
            if len(globals.selected_files_order) < 3:
                globals.selected_files_order.append(i)
            else:
                # Too many selected, uncheck this item immediately
                globals.file_list_widget.item(i).setCheckState(Qt.Unchecked)

    # If less than 3 selected, clear globals.img_paths and globals.file_names partially
    for idx in range(3):
        if idx < len(globals.selected_files_order):
            path = globals.file_paths[globals.selected_files_order[idx]]
            globals.img_paths[idx] = path
            globals.file_names[idx] = os.path.basename(path)
        else:
            globals.img_paths[idx] = None
            globals.file_names[idx] = None

def on_confirm_clicked():
    globals.microns_per_pixel_x = globals.float_input_micron_x.value()
    # print(globals.microns_per_pixel_x)
    globals.microns_per_pixel_y = globals.float_input_micron_y.value()
    # print(globals.microns_per_pixel_y)
    globals.true_origin_x = globals.origin_x_input.value()
    globals.true_origin_y = globals.origin_y_input.value()
    # print(globals.true_origin_x)
    # print(globals.true_origin_y)

    if len(globals.selected_files_order) != 3:
        QMessageBox.warning(globals.window, "Invalid Selection", "Please select exactly 3 items.")
        return
    QApplication.processEvents()
    init_gui()


def update_progress(val):
    progress_bar.setValue(val)
    QApplication.processEvents()


def create_manual_scan_tab():
    data_fields = {
        "min_threshold_intensity": None,
        "min_threshold_area": None,
        "microns_per_pixel_x": None,
        "microns_per_pixel_y": None,
        "true_origin_x": None,
        "true_origin_y": None,
        "number_of_scans": None,
        "debt_names": None,
        "x_motor": None,
        "x_start": None,
        "x_end": None,
        "y_motor": None,
        "y_start": None,
        "y_end": None,
        "dwell_time": None,
    }

    # Dictionary to store references to input fields
    input_fields = {}
    manual_scan_widget = QWidget()
    manual_scan_layout = QVBoxLayout()

    input_rows = []

    # Create input fields
    for key in data_fields:
        row = QHBoxLayout()
        label = QLabel(key)
        line_edit = QLineEdit()
        input_fields[key] = line_edit
        row.addWidget(label)
        row.addWidget(line_edit)
        manual_scan_layout.addLayout(row)
        input_rows.append((label, line_edit))  # Save widgets to hide later

    # Create the send button
    send_first_scan_btn = QPushButton("Send First Scan")
    manual_scan_layout.addWidget(send_first_scan_btn)

    # Waiting label (initially hidden)
    waiting_label = QLabel("Waiting for files to be generated...")
    waiting_label.setAlignment(Qt.AlignCenter)
    waiting_label.setStyleSheet("font-size: 16px; color: gray;")
    waiting_label.hide()
    manual_scan_layout.addWidget(waiting_label)

    # Setup timer
    timer = QTimer()
    timer.setInterval(1000)  # Check every 1 second

    def check_for_tiffs():
        expected_files = {"file1.tiff", "file2.tiff", "file3.tiff"}
        found_files = set(os.listdir(os.getcwd()))
        if expected_files.issubset(found_files):
            timer.stop()
            waiting_label.setText("✅ Files generated!")

    def check_for_tiffs():
        all_files = os.listdir(os.getcwd())
        tiff_files = {f for f in all_files if f.lower().endswith('.tiff')}
        if len(tiff_files) >= 3:
            timer.stop()
            waiting_label.setText("✅ 3 TIFF files detected!")

    def handle_send_scan():
        scan_data = {}
        for key, line_edit in input_fields.items():
            text = line_edit.text().strip()
            if text == "":
                scan_data[key] = None
            else:
                try:
                    value = float(text)
                    if value.is_integer():
                        value = int(value)
                    scan_data[key] = value
                except ValueError:
                    scan_data[key] = text

        file_path = os.path.join(os.getcwd(), "first_scan.json")
        with open(file_path, "w") as f:
            json.dump(scan_data, f, indent=4)
        print(f"Scan parameters saved to: {file_path}")

        # Hide all inputs
        for label, edit in input_rows:
            label.hide()
            edit.hide()
        send_first_scan_btn.hide()

        # Show waiting label and start timer
        waiting_label.show()
        timer.start()
        blobs = first_scan_detect_blobs()


        watch_dir = Path(os.getcwd())
        json_path = watch_dir / "first_scan.json"
        with open(json_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                json_items = [[key, value] for key, value in data.items()]
            else:
                json_items = [data]

        blobs = find_union_blobs(blobs,json_items[2][1],json_items[3][1],json_items[4][1],json_items[5][1])

        json_safe_data = make_json_serializable(blobs)
        print()
        for idx, blob_data in json_safe_data.items():
            print(f"Blob #{idx}:")
            for key, value in blob_data.items():
                print(f"  {key}: {value}")

            print("-" * 40)  # separator line between blobs

        save_each_blob_as_individual_scan(json_safe_data, px_per_um=1.25)



        # output_path = os.path.join(globals.selected_directory, "selected_blobs_to_queue_server.json")
        # with open(output_path, "w") as f:
        #     json.dump(json_safe_data, f, indent=2)
        # structure_blob_tooltips(output_path)
        # send_json_boxes_to_queue_with_center_move(output_path) 

    send_first_scan_btn.clicked.connect(handle_send_scan)
    timer.timeout.connect(check_for_tiffs)
    manual_scan_widget.setLayout(manual_scan_layout)
    return manual_scan_widget


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
        self.hover_label = globals.hover_label
        self.blobs = []
        self.visible_colors = set(globals.element_colors)
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
        self.x_micron_label.setText(f"X Real location(µm): {(x * globals.microns_per_pixel_x) + globals.true_origin_x:.2f}")
        self.y_micron_label.setText(f"Y Real location(µm): {(y * globals.microns_per_pixel_y) + globals.true_origin_y:.2f}")
        for blob in self.blobs:
            if blob['color'] not in self.visible_colors:
                continue
            cx, cy = blob['center']
            r = blob['radius']
            
            real_w = blob['box_size'] * globals.microns_per_pixel_x
            real_h = blob['box_size'] * globals.microns_per_pixel_y
            real_cx = (cx * globals.microns_per_pixel_x) + globals.true_origin_x
            real_cy = (cy * globals.microns_per_pixel_y) + globals.true_origin_y

            if abs(x - cx) <= r and abs(y - cy) <= r:
                html = (
                    f"{blob['Box']}<br>"
                    f"Center: ({cx}, {cy})<br>"
                    f"Length: {blob['box_size']} px<br>"
                    f"Box area: {blob['box_size'] * blob['box_size']} px²<br>"
                    f"<br>"
                    f"Real Center location(µm): ({(cx * globals.microns_per_pixel_x) + globals.true_origin_x:.2f} µm, {(cy * globals.microns_per_pixel_y) + globals.true_origin_y:.2f} µm)<br>"
                    f"Real box size(µm): ({blob['box_size'] * globals.microns_per_pixel_x:.2f} µm × {blob['box_size'] * globals.microns_per_pixel_y:.2f} µm)<br>"
                    f"Real box area(µm): ({blob['box_size']**2 * globals.microns_per_pixel_x * globals.microns_per_pixel_y:.2f} µm²)<br>"
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
                
        if globals.checkboxes['union'].isChecked():  # actual union box visibility check
            for ub in globals.global_union_boxes:
                if ub['rect'].contains(pos.toPoint()):
                    self.hover_label.setText(ub['text'])
                    self.hover_label.adjustSize()
                    self.hover_label.move(event.x() + 15, event.y() - 30)
                    self.hover_label.setStyleSheet(
                        "background-color: white; color: black; border: 1px solid black; padding: 4px;"
                    )
                    self.globals.hover_label.show()
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

def on_slider_change(value, color):
        snapped = round(value / 10) * 10
        snapped = max(0, min(250, snapped))
        if globals.thresholds[color] != snapped:
            globals.thresholds[color] = snapped
            globals.sliders[color].blockSignals(True)
            globals.sliders[color].setValue(snapped)
            globals.sliders[color].blockSignals(False)
            globals.slider_labels[color].setText(f"{globals.checkboxes[color].text()}_threshold: {snapped}")
            update_boxes()


def on_area_slider_change(value, color):
    valid_areas = list(range(10, 401, 10)) 
    snapped = min(valid_areas, key=lambda a: abs(a - value))
    if globals.area_thresholds[color] != snapped:
        globals.area_thresholds[color] = snapped
        globals.area_sliders[color].blockSignals(True)
        globals.area_sliders[color].setValue(snapped)
        globals.area_sliders[color].blockSignals(False)
        globals.area_slider_labels[color].setText(f"{globals.checkboxes[color].text()}_min_area: {snapped}")
        update_boxes()


    
def add_box():
    # Notify user
    QMessageBox.information(globals.window, "Add Union Box", "Click and drag to define a new union box.")

    temp_state = {'start': None}

    def on_press(event):
        temp_state['start'] = globals.graphics_view.mapToScene(event.pos()).toPoint()

    def on_release(event):
        if temp_state['start'] is None:
            return
    
        end = globals.graphics_view.mapToScene(event.pos()).toPoint()
        start = temp_state['start']
        temp_state['start'] = None
    
        x1, y1 = start.x(), start.y()
        x2, y2 = end.x(), end.y()
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = max(abs(x2 - x1), abs(y2 - y1))
        area = length * length
    
        real_cx = (cx * globals.microns_per_pixel_x) + globals.true_origin_x
        real_cy = (cy * globals.microns_per_pixel_y) + globals.true_origin_y
        real_length_x = length * globals.microns_per_pixel_x
        real_length_y = length * globals.microns_per_pixel_y
        real_area = real_length_x * real_length_y
    
        real_top_left = (
            ((cx - length // 2) * globals.microns_per_pixel_x) + globals.true_origin_x,
            ((cy - length // 2) * globals.microns_per_pixel_y) + globals.true_origin_y
        )
        real_bottom_right = (
            ((cx + length // 2) * globals.microns_per_pixel_x) + globals.true_origin_x,
            ((cy + length // 2) * globals.microns_per_pixel_y) + globals.true_origin_y
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
        current_union_dict = globals.graphics_view.union_dict
        next_index = max(current_union_dict.keys(), default=0) + 1
        current_union_dict[next_index] = new_union
    
        # Update both object list and dictionary
        globals.graphics_view.union_objects = list(current_union_dict.values())
        globals.graphics_view.union_dict = current_union_dict

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

        item_text = f"Custom Box #{globals.custom_box_number}"

        
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, globals.custom_box_number)
        
        tooltip_text = (
            f"<b>Custom Box #{globals.custom_box_number}</b><br>"
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

        globals.custom_box_number = globals.custom_box_number + 1
        
        globals.union_list_widget.addItem(item)

        output_path = os.path.join(globals.selected_directory, "union_blobs.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(globals.graphics_view.union_dict, f)
            
        # Restore original mouse events and update display
        globals.graphics_view.mousePressEvent = original_mouse_press
        globals.graphics_view.mouseReleaseEvent = original_mouse_release
        update_boxes()

    # Temporarily override mouse events
    original_mouse_press = globals.graphics_view.mousePressEvent
    original_mouse_release = globals.graphics_view.mouseReleaseEvent
    globals.graphics_view.mousePressEvent = on_press
    globals.graphics_view.mouseReleaseEvent = on_release

def on_union_item_selected():
    selected_items = globals.union_list_widget.selectedItems()
    globals.graphics_view.highlight_selected_union_boxes(selected_items)


def send_to_list():
    # Collect all existing item texts in globals.queue_server_list
    existing_texts = {globals.queue_server_list.item(i).text() for i in range(globals.queue_server_list.count())}

    for item in globals.union_list_widget.selectedItems():
        if item.text() not in existing_texts:
            new_item = QListWidgetItem(item.text())
            new_item.setToolTip(item.toolTip())
            new_item.setData(Qt.UserRole, item.data(Qt.UserRole))  # preserve index if needed
            globals.queue_server_list.addItem(new_item)

def send_to_queue_server():
    data = {}
    for i in range(globals.queue_server_list.count()):
        item = globals.queue_server_list.item(i)
        text = item.text()

        if "✅" in text or "Data sent and saved" in text:
            continue

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

    output_path = os.path.join(globals.selected_directory, "selected_blobs_to_queue_server.json")
    with open(output_path, "w") as f:
        json.dump(json_safe_data, f, indent=2)

    structure_blob_tooltips(output_path)
    send_json_boxes_to_queue_with_center_move(output_path)

    globals.queue_server_list.clear()
    globals.queue_server_list.addItem("✅ Data sent and saved")

def get_elements_list():    
    # Collect all visible blobs across elements
    globals.union_list_widget.clear()
    all_blobs = get_current_blobs()
    for blob in all_blobs:
        cx, cy = blob['center']
        box_size = blob['box_size']

        real_cx = (cx * globals.microns_per_pixel_x) + globals.true_origin_x
        real_cy = (cy * globals.microns_per_pixel_y) + globals.true_origin_y
        real_length_x = box_size * globals.microns_per_pixel_x
        real_length_y = box_size * globals.microns_per_pixel_y
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

        globals.union_list_widget.addItem(item)

def clear_queue_server_list():    
    globals.queue_server_list.clear() 


def init_gui():
    globals.current_iteration = 0
    globals.precomputed_blobs = {}
    
    # Remove old progress bars if they exist
    for i in reversed(range(globals.main_layout.count())):
        widget = globals.main_layout.itemAt(i).widget()
        if isinstance(widget, QProgressBar):
            globals.main_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()

    # for btn in buttons:
    #     btn.setEnabled(False)

    # Load and convert
    globals.graphics_view, globals.controls_widget

    if globals.graphics_view is not None:
        globals.main_layout.removeWidget(globals.graphics_view)
        globals.graphics_view.setParent(None)
        globals.graphics_view.deleteLater()
        globals.graphics_view = None

    if globals.controls_widget is not None:
        globals.main_layout.removeWidget(globals.controls_widget)
        globals.controls_widget.setParent(None)
        globals.controls_widget.deleteLater()
        globals.controls_widget = None
        
    img_r, img_g, img_b = [tiff.imread(p).astype(np.float32) for p in globals.img_paths]
    
    # Resize to majority shape
    shapes = [img_r.shape, img_g.shape, img_b.shape]
    shape_counts = Counter(shapes)
    globals.target_shape = shape_counts.most_common(1)[0][0]
    # print(f"Target (majority) shape: {globals.target_shape}")
    
    img_r = resize_if_needed(img_r, globals.file_names[0])
    img_g = resize_if_needed(img_g, globals.file_names[1])
    img_b = resize_if_needed(img_b, globals.file_names[2])

    norm_dilated = [normalize_and_dilate(im) for im in [img_r, img_g, img_b]]
    normalized = [nd[0] for nd in norm_dilated]
    dilated = [nd[1] for nd in norm_dilated]

    thresholds_range = list(range(0, 256, 10))
    area_range = list(range(10, 501, 10))

    QApplication.processEvents()
    progress_bar = QProgressBar()
    progress_bar.setRange(0, len(globals.element_colors) * len(thresholds_range) * len(area_range))
    progress_bar.setValue(0)
    progress_bar.setTextVisible(True)
    progress_bar.setFormat("Computing blobs... %p%")
    globals.main_layout.addWidget(progress_bar)
    QApplication.processEvents() 
    progress_bar.show()
    QApplication.processEvents()
    total_iterations = len(globals.element_colors) * len(thresholds_range) * len(area_range)
    globals.precomputed_blobs = {}
    
    QTimer.singleShot(0, lambda: start_blob_computation(
        globals.element_colors,
        thresholds_range,
        area_range,
        globals.precomputed_blobs,
        dilated,
        img_r,
        img_g,
        img_b,
        globals.file_names,
        progress_bar
    )) 
 
    os.makedirs("data", exist_ok=True)

    QApplication.processEvents()

    # Make a deep copy so original data stays untouched
    blobs_to_save = copy.deepcopy(globals.precomputed_blobs)    
    # Add real-world info and tooltip HTML to copied blobs
    for color in blobs_to_save:
        for key in blobs_to_save[color]:
            for blob in blobs_to_save[color][key]:
                cx, cy = blob['center']
    
                # Calculate real-world values
                real_center_x = (cx * globals.microns_per_pixel_x) + globals.true_origin_x
                real_center_y = (cy * globals.microns_per_pixel_y) + globals.true_origin_y
                real_box_size_x = blob['box_size'] * globals.microns_per_pixel_x
                real_box_size_y = blob['box_size'] * globals.microns_per_pixel_y
                real_box_area = (blob['box_size'] ** 2) * globals.microns_per_pixel_x * globals.microns_per_pixel_y
    
                # Store real-world values
                blob['real_center'] = (real_center_x, real_center_y)
                blob['real_box_size'] = (real_box_size_x, real_box_size_y)
                blob['real_box_area'] = real_box_area

                real_w = blob['box_size'] * globals.microns_per_pixel_x
                real_h = blob['box_size'] * globals.microns_per_pixel_y
                real_cx = (cx * globals.microns_per_pixel_x) + globals.true_origin_x
                real_cy = (cy * globals.microns_per_pixel_y) + globals.true_origin_y
                real_tl_x = real_cx - real_w / 2
                real_tl_y = real_cy - real_h / 2
                real_br_x = real_cx + real_w / 2
                real_br_y = real_cy + real_h / 2

                real_w = blob['box_size'] * globals.microns_per_pixel_x
                real_h = blob['box_size'] * globals.microns_per_pixel_y
                real_cx = (cx * globals.microns_per_pixel_x) + globals.true_origin_x
                real_cy = (cy * globals.microns_per_pixel_y) + globals.true_origin_y
                
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
    output_path = os.path.join(globals.selected_directory, "precomputed_blobs_with_real_info.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(blobs_to_save, f) 
        
    progress_bar.hide()
    QApplication.processEvents()

    # print(time.strftime('%H:%M:%S'))
    globals.thresholds = {color: 100 for color in globals.element_colors}
    globals.area_thresholds = {color: 200 for color in globals.element_colors}
    
    merged_rgb = cv2.merge([
        cv2.normalize(img_r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.normalize(img_g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.normalize(img_b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ])

    globals.hover_label = QLabel()
    globals.hover_label.setWindowFlags(Qt.ToolTip)
    globals.hover_label.hide()

    x_label = QLabel("X: 0")
    y_label = QLabel("Y: 0")

    x_micron_label = QLabel("X Real: 0")
    y_micron_label = QLabel("Y Real: 0")
    
    scene = QGraphicsScene()
    q_img = QImage(merged_rgb.data, merged_rgb.shape[1], merged_rgb.shape[0], merged_rgb.shape[1] * 3, QImage.Format_RGB888)

    globals.graphics_view = ZoomableGraphicsView(scene, globals.hover_label, x_label, y_label, x_micron_label, y_micron_label)
    globals.graphics_view.current_qimage = q_img
    globals.pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(globals.graphics_view.current_qimage))
    scene.addItem(globals.pixmap_item)
    globals.main_layout.addWidget(globals.graphics_view)

    globals.checkboxes = {}
    selected_colors = set(globals.element_colors)
 
    legend_layout = QVBoxLayout()
    legend_label = QLabel("Legend")
    legend_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
    legend_layout.addWidget(legend_label)
    for i, color in enumerate(globals.element_colors):
        cb = QCheckBox(globals.file_names[i])
        cb.setChecked(True)
        cb.setStyleSheet(f"color: {color}")
        cb.stateChanged.connect(update_boxes)
        globals.checkboxes[color] = cb
        legend_layout.addWidget(cb)

    globals.union_checkbox = QCheckBox("Union Boxes")
    globals.union_checkbox.setChecked(True)
    globals.union_checkbox.setStyleSheet("color: black")
    globals.union_checkbox.stateChanged.connect(update_boxes)
    legend_layout.addWidget(globals.union_checkbox)
    globals.checkboxes['union'] = globals.union_checkbox


    legend_layout.addStretch()

    globals.sliders = {}
    globals.slider_labels = {}
    slider_layout = QHBoxLayout()
        
    for color in globals.element_colors:
        i = globals.element_colors.index(color)
        vbox = QVBoxLayout()
        label = QLabel(f"{globals.file_names[i]}_threshold: {globals.thresholds[color]}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(255)
        slider.setTickInterval(10)
        slider.setValue(globals.thresholds[color])
        slider.setTickPosition(QSlider.TicksBelow)
        slider.valueChanged.connect(lambda val, c=color: on_slider_change(val, c))
        globals.sliders[color] = slider
        globals.slider_labels[color] = label
        vbox.addWidget(label)
        vbox.addWidget(slider)
        slider_layout.addLayout(vbox)

    globals.area_sliders = {}
    globals.area_slider_labels = {}

    area_slider_layout = QHBoxLayout()
    
    for color in globals.element_colors:
        i = globals.element_colors.index(color)
        vbox = QVBoxLayout()
        label = QLabel(f"{globals.file_names[i]}_min_area: {globals.area_thresholds[color]}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(10)
        slider.setMaximum(400)
        slider.setTickInterval(10)
        slider.setValue(globals.area_thresholds[color])
        slider.setTickPosition(QSlider.TicksBelow)
        slider.valueChanged.connect(lambda val, c=color: on_area_slider_change(val, c))
        globals.area_sliders[color] = slider
        globals.area_slider_labels[color] = label
        vbox.addWidget(label)
        vbox.addWidget(slider)
        area_slider_layout.addLayout(vbox)

    exit_btn = QPushButton("Exit")
    exit_btn.clicked.connect(lambda: (globals.window.close(), globals.app.quit()))
    reset_btn = QPushButton("Reset View")
    reset_btn.clicked.connect(lambda: globals.graphics_view.resetTransform())
        
    add_btn = QPushButton("Add custom box")
    add_btn.clicked.connect(add_box)
    
    union_btn = QPushButton("Get current unions")
    union_btn.clicked.connect(union_function)

    globals.union_list_widget = QListWidget()
    globals.union_list_widget.setSelectionMode(QListWidget.MultiSelection)  # Allow multiple selection
    globals.union_list_widget.setMinimumHeight(200)
    globals.union_list_widget.itemSelectionChanged.connect(on_union_item_selected)
    globals.union_list_widget.setLineWidth(1)
    globals.union_list_widget.setStyleSheet("""
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
    globals.union_list_widget.setMouseTracking(True)  # Required for hover without clicking
    globals.union_list_widget.setEnabled(True)

    globals.queue_server_list = QListWidget()
    globals.queue_server_list.setMinimumHeight(200)
    globals.queue_server_list.setLineWidth(1)

    send_to_list_btn = QPushButton("Add to Queue server List")
    send_to_list_btn.clicked.connect(send_to_list)

    send_elements_to_list_btn = QPushButton("Add all individual element scans to list")
    send_elements_to_list_btn.clicked.connect(get_elements_list)

    send_to_queue_server_btn = QPushButton("Send to Queue server List")    
    send_to_queue_server_btn.clicked.connect(send_to_queue_server)

    clear_queue_server_btn = QPushButton("Clear")    
    clear_queue_server_btn.clicked.connect(clear_queue_server_list)

    union_list_layout = QVBoxLayout()
    union_list_layout.addWidget(globals.union_list_widget)
    union_list_layout.addWidget(send_to_list_btn)
    union_list_layout.addWidget(send_elements_to_list_btn)
    union_list_layout.addWidget(union_btn)
    union_list_layout.addWidget(add_btn)

    queue_server_list_layout = QVBoxLayout()
    queue_server_list_layout.addWidget(globals.queue_server_list)
    queue_server_list_layout.addWidget(send_to_queue_server_btn)
    queue_server_list_layout.addWidget(clear_queue_server_btn)

    dual_list_layout = QHBoxLayout()
    dual_list_layout.addLayout(union_list_layout)
    dual_list_layout.addLayout(queue_server_list_layout)

    controls = QVBoxLayout()
    controls.addWidget(exit_btn)
    controls.addWidget(reset_btn)
    
    controls.addLayout(dual_list_layout)
    # controls.addWidget(globals.union_list_widget)
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
    layout.addWidget(globals.graphics_view)
    side_panel = QWidget()
    side_panel.setLayout(controls)
    globals.controls_widget = side_panel
    layout.addWidget(side_panel)

    globals.main_layout.addLayout(layout)
    globals.hover_label.setParent(globals.window)

    blobs = get_current_blobs()
    globals.graphics_view.update_blobs(blobs, selected_colors)
    redraw_boxes(blobs, selected_colors)
    globals.window.resize(1900, 1000)
