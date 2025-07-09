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

# from globals import *
from gui import *
# from utils import *

import globals
import utils

globals.element_colors = ["red", "green", "blue"]

globals.graphics_view = None
globals.controls_widget = None

globals.microns_per_pixel_x = 0.0
globals.microns_per_pixel_y = 0.0

globals.true_origin_x = 0.0
globals.true_origin_y = 0.0

globals.selected_directory = None

globals.selected_files_order = []


app = QApplication(sys.argv)
globals.window = QWidget()
globals.window.setWindowTitle("Multi-Element Image Analyzer")
globals.window.resize(600, 500)


globals.main_layout = QVBoxLayout()

# Directory selector
dir_button = QPushButton("Choose Directory")
dir_button.clicked.connect(on_dir_selected)
globals.main_layout.addWidget(dir_button)

# List of files in the selected directory
globals.file_list_widget = QListWidget()
globals.file_list_widget.setSelectionMode(QListWidget.NoSelection)
globals.file_list_widget.itemChanged.connect(update_selection)
# Horizontal layout: file list on the left, confirm button on the right
file_confirm_layout = QHBoxLayout()
file_confirm_layout.addWidget(globals.file_list_widget)

# Right side (confirm button)
right_panel = QVBoxLayout()
confirm_button = QPushButton("Confirm Selection")
confirm_button.clicked.connect(on_confirm_clicked)
right_panel.addWidget(confirm_button)
right_panel.addStretch()

globals.float_input_micron_x  = QDoubleSpinBox()
globals.float_input_micron_x.setPrefix("Enter X(µm) per pixel:")
globals.float_input_micron_x.setRange(0.0, 1000.0)         # Adjust range as needed
globals.float_input_micron_x.setSingleStep(0.1)            # Step size
globals.float_input_micron_x.setDecimals(3)                # Number of decimal places
globals.float_input_micron_x.setValue(1.0)                 # Default value
right_panel.addWidget(globals.float_input_micron_x)

globals.float_input_micron_y  = QDoubleSpinBox()
globals.float_input_micron_y.setPrefix("Enter Y(µm) per pixel:")
globals.float_input_micron_y.setRange(0.0, 1000.0)         # Adjust range as needed
globals.float_input_micron_y.setSingleStep(0.1)            # Step size
globals.float_input_micron_y.setDecimals(3)                # Number of decimal places
globals.float_input_micron_y.setValue(1.0)                 # Default value
right_panel.addWidget(globals.float_input_micron_y)

globals.origin_x_input = QDoubleSpinBox()
globals.origin_x_input.setPrefix("Origin X(µm): ")
globals.origin_x_input.setRange(-1e6, 1e6)
globals.origin_x_input.setDecimals(2)
globals.origin_x_input.setValue(0.0)                 # Default value

globals.origin_y_input = QDoubleSpinBox()
globals.origin_y_input.setPrefix("Origin Y(µm): ")
globals.origin_y_input.setRange(-1e6, 1e6)
globals.origin_y_input.setDecimals(2)
globals.origin_x_input.setValue(0.0)                 # Default value

right_panel.addWidget(globals.origin_x_input)
right_panel.addWidget(globals.origin_y_input)

file_confirm_layout.addLayout(right_panel)

globals.main_layout.addLayout(file_confirm_layout)

# Track full file paths
globals.file_paths = []

# Global list to store hoverable union boxes
globals.global_union_boxes = []

# Global Variable for current blob detection 
globals.current_iteration = 0

#Custom Box Counter
globals.custom_box_number = 1












layout_wrapper = QWidget()
layout_wrapper.setLayout(globals.main_layout)
tabs = QTabWidget()
tabs.addTab(layout_wrapper, "Home")

tabs.addTab(create_manual_scan_tab(), "Manual Scan")

screen = QVBoxLayout()
screen.addWidget(tabs)

globals.window.setLayout(screen)
globals.window.show()
sys.exit(app.exec_())