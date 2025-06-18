# Testing multithread code 
import os
import sys
import cv2
import numpy as np
import tifffile as tiff
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QCheckBox, QSlider, QFileDialog, QListWidget, QListWidgetItem, 
    QMessageBox, QDoubleSpinBox, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen
from PyQt5.QtCore import Qt
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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
 
def on_confirm_clicked():
    global microns_per_pixel_x
    microns_per_pixel_x = float_input_micron_x.value()
    print(microns_per_pixel_x)
    global microns_per_pixel_y
    microns_per_pixel_y = float_input_micron_y.value()
    print(microns_per_pixel_y)
    global true_origin_x
    true_origin_x = origin_x_input.value()
    global true_origin_y
    true_origin_y = origin_y_input.value()
    print(true_origin_x)
    print(true_origin_y)

    if len(selected_files_order) != 3:
        QMessageBox.warning(window, "Invalid Selection", "Please select exactly 3 items.")
        return

    init_gui()
        
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

# === Defer the rest of the UI until files are selected ===
def init_gui():       
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
    print(f"Target (majority) shape: {target_shape}")
    
    def resize_if_needed(img, name):
        if img.shape != target_shape:
            print(f"Resizing {name} from {img.shape} → {target_shape}")
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
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2)
            box_size = 2 * radius
            box_x, box_y = x - radius, y - radius
            mask = np.zeros(img_orig.shape, dtype=np.uint8)
            cv2.rectangle(mask, (box_x, box_y), (x + radius, y + radius), 255, thickness=-1)
            vals = img_orig[mask == 255]
            vals_dilated = img_norm[mask == 255]
            if vals.size > 0:
                blobs.append({
                    'center': (x, y),
                    'radius': radius,
                    'color': color,
                    'file': file_name,
                    'max_intensity': vals.max(),
                    'mean_intensity': vals.mean(),
                    'mean_dilation': float(vals_dilated.mean()),
                    'box_x': box_x,
                    'box_y': box_y,
                    'box_size': box_size
                })
        return blobs

    norm_dilated = [normalize_and_dilate(im) for im in [img_r, img_g, img_b]]
    normalized = [nd[0] for nd in norm_dilated]
    dilated = [nd[1] for nd in norm_dilated]

    thresholds_range = list(range(0, 256, 10))
    area_range = list(range(10, 501, 10))

    progress_bar = QProgressBar()

    
    # progress_bar.setRange(0, len(element_colors) * len(thresholds_range) * len(area_range))
    # progress_bar.setValue(0)
    # progress_bar.setTextVisible(True)
    # progress_bar.setFormat("Computing blobs... %p%")
    # main_layout.addWidget(progress_bar)
    # total_iterations = len(element_colors) * len(thresholds_range) * len(area_range)
    # current_iteration = 0
    
    # precomputed_blobs = {}
    
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

    # progress_bar.hide()

    precomputed_blobs = {color: {} for color in element_colors}
    print(time.strftime('%H:%M:%S'))
    # Build tasks
    tasks = []
    for i, color in enumerate(element_colors):
        for t in thresholds_range:
            for a in area_range:
                tasks.append((
                    i, color, t, a,
                    dilated[i],
                    [img_r, img_g, img_b][i],
                    file_names[i]
                ))
    
    # Set up progress bar
    progress_bar.setRange(0, len(tasks))
    progress_bar.setValue(0)
    progress_bar.setTextVisible(True)
    progress_bar.setFormat("Computing blobs... %p%")
    main_layout.addWidget(progress_bar)
    
    # Run in threads
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                detect_blobs, img_norm, img_orig, t, a, color, file_name
            ): (color, t, a)
            for i, color, t, a, img_norm, img_orig, file_name in tasks
        }
    
        for i, future in enumerate(as_completed(futures)):
            color, t, a = futures[future]
            result = future.result()
            precomputed_blobs[color][(t, a)] = result
    
            # Update progress
            progress_bar.setValue(i + 1)
            QApplication.processEvents()
    
    progress_bar.hide()
    print(time.strftime('%H:%M:%S'))
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
    pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(q_img))
    scene.addItem(pixmap_item)

    class ZoomableGraphicsView(QGraphicsView):
        def __init__(self, scene, hover_label, x_label, y_label, x_micron_label, y_micron_label):
            super().__init__(scene)
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
                if abs(x - cx) <= r and abs(y - cy) <= r:
                    html = (
                        f"<b>{blob['color'].capitalize()} Element</b><br>"
                        f"<i>{blob['file']}</i><br>"
                        f"Center: ({cx}, {cy})<br>"
                        f"Top-left: ({blob['box_x']}, {blob['box_y']})<br>"
                        f"Box size: {blob['box_size']} x {blob['box_size']} px<br>"
                        f"Box area: {blob['box_size'] * blob['box_size']} px²<br>"
                        f"<br>"
                        f"Real Center location(µm): ({(cx * microns_per_pixel_x) + true_origin_x:.2f} µm, {(cy * microns_per_pixel_y) + true_origin_y:.2f} µm)<br>"
                        f"Real box size(µm): ({blob['box_size'] * microns_per_pixel_x:.2f} µm × {blob['box_size'] * microns_per_pixel_y:.2f} µm)<br>"
                        f"Real box area(µm): ({blob['box_size']**2 * microns_per_pixel_x * microns_per_pixel_y:.2f} µm²)<br>"
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
            self.hover_label.hide()

        def update_blobs(self, blobs, visible_colors):
            self.blobs = blobs
            self.visible_colors = visible_colors

    graphics_view = ZoomableGraphicsView(scene, hover_label, x_label, y_label, x_micron_label, y_micron_label)
    main_layout.addWidget(graphics_view)
    
    def redraw_boxes(blobs, selected_colors):
        updated_img = QImage(q_img)
        painter = QPainter(updated_img)
        for blob in blobs:
            if blob['color'] in selected_colors:
                cx, cy, r = *blob['center'], blob['radius']
                painter.setPen(QPen(QColor(blob['color']), 2))
                painter.drawRect(cx - r, cy - r, 2 * r, 2 * r)
        painter.end()
        pixmap_item.setPixmap(QPixmap.fromImage(updated_img))

    checkboxes = {}
    selected_colors = set(element_colors)

    def update_boxes():
        nonlocal selected_colors
        selected_colors = {c for c, cb in checkboxes.items() if cb.isChecked()}
        blobs = [b for b in get_current_blobs() if b['color'] in selected_colors]
        graphics_view.update_blobs(blobs, selected_colors)
        redraw_boxes(blobs, selected_colors)
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

    controls = QVBoxLayout()
    controls.addWidget(exit_btn)
    controls.addWidget(reset_btn)
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
