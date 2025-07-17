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


window = None
file_list_widget = None
file_paths = None
selected_files_order = None
img_paths = [None, None, None]  # [R, G, B]
file_names = [None, None, None]
float_input_micron_x = None
float_input_micron_y = None
origin_x_input = None
origin_y_input = None
true_origin_x = None
true_origin_y = None
main_layout = None

graphics_view = None

controls_widget = None

target_shape = None

element_colors = None

thresholds = None

area_thresholds = None

precomputed_blobs = None

pixmap_item = None

checkboxes = None

global_union_boxes = None

current_iteration = None

sliders = None

area_sliders = None

microns_per_pixel_x = None

microns_per_pixel_y = None

union_list_widget = None

queue_server_list = None

selected_directory = None

union_checkbox = None

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

def update_boxes():
    # nonlocal selected_colors
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



def union_box_drawer(union_dict, base_img=None, clear_only=False):
    import globals
    globals.global_union_boxes = []  # Clear old ones

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
        globals.global_union_boxes.append({
            'rect': QRect(x, y, w, h),
            'text': hover_text
        })

    painter.end()
    pixmap_item.setPixmap(QPixmap.fromImage(updated_img))

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

custom_box = None

custom_box_number = None

slider_labels = None

area_slider_labels = None

hover_label = None
