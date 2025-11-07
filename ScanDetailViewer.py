import sys
import os
import shutil
import json
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QSizePolicy, QPushButton, QFileDialog, QShortcut, QCheckBox
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap, QImage, QPainter, QKeySequence, QFont, QPen

# --- File Path Constants ---
PROCESSED_SCANS_DIR = "/home/codingcarlos/Documents/github/SULI-2025-Summer/data/scans_grouped"
RAW_PRIMARY_SCAN_DIR = "/home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3"
RAW_SECONDARY_SCAN_DIR = os.path.join(RAW_PRIMARY_SCAN_DIR, "all_xrf")
INFO_PATH = "/home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3/data/user_macros"

# --- Helper Classes ---
class XRFScan:
    """A data class to hold information about a single coarse scan."""
    def __init__(self, scan_id, group_config):
        self.id = scan_id
        self.group_config = group_config
        self.fine_scan_ids = []

    @property
    def group_name(self):
        return self.group_config["name"]

class SquareLabel(QLabel):
    def __init__(self, main_window, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_window = main_window
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.union_data = None
        self.show_text_annotations = False # Renamed and changed purpose

    def setAnnotations(self, data, show_text): # Changed parameter name
        self.union_data = data
        self.show_text_annotations = show_text # Updated variable
        self.update() # Trigger a repaint

    def paintEvent(self, event):
        super().paintEvent(event)
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return
            
        widget_width = self.width()
        widget_height = self.height()
        size = min(widget_width, widget_height)
        x_offset = (widget_width - size) // 2
        y_offset = (widget_height - size) // 2
        target_rect = QRect(x_offset, y_offset, size, size)
        
        # Maintain aspect ratio of the source pixmap
        source_size = pix.size()
        scaled_size = source_size.scaled(target_rect.size(), Qt.KeepAspectRatio)
        final_rect = QRect(0, 0, scaled_size.width(), scaled_size.height())
        final_rect.moveCenter(target_rect.center())

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.HighQualityAntialiasing, True) # This might help
        painter.drawPixmap(final_rect, pix) # Draw original pixmap, letting drawPixmap handle scaling

        if self.union_data: # Boxes always drawn if data exists
            source_rect = pix.rect()
            x_scale = final_rect.width() / source_rect.width()
            y_scale = final_rect.height() / source_rect.height()

            pen = QPen(Qt.white)
            pen.setWidth(self.main_window.BOX_BORDER_THICKNESS) # Use variable
            painter.setPen(pen)
            
            font = QFont()
            font.setPointSize(self.main_window.ANNOTATION_FONT_SIZE) # Use variable
            painter.setFont(font)

            tile_w, tile_h = source_rect.width(), source_rect.height()

            for key, item in self.union_data.items():
                if 'image_center' in item and 'image_length' in item:
                    x, y = item['image_center']
                    length = item['image_length']
                    if x < 0 or y < 0:
                        x += tile_w / 2
                        y += tile_h / 2
                    
                    half_length = length / 2
                    box_x = x - half_length
                    box_y = y - half_length

                    # Scale to target coordinates
                    target_box_x = final_rect.x() + box_x * x_scale
                    target_box_y = final_rect.y() + box_y * y_scale
                    target_box_w = length * x_scale
                    target_box_h = length * y_scale
                    
                    painter.drawRect(int(target_box_x), int(target_box_y), int(target_box_w), int(target_box_h))

                    # Text drawing is now conditional on self.show_text_annotations
                    if self.show_text_annotations and 'real_center_um' in item and 'real_size_um' in item:
                        real_center = item['real_center_um']
                        real_size = item['real_size_um']
                        text = f"Center: ({real_center[0]:.1f}, {real_center[1]:.1f})\nSize: ({real_size[0]:.1f}, {real_size[1]:.1f})"
                        
                        text_x = target_box_x - 10
                        text_y = target_box_y + target_box_h + self.main_window.ANNOTATION_TEXT_OFFSET_Y # Use variable
                        
                        # Draw text with a slight shadow for readability
                        painter.setPen(Qt.black)
                        painter.drawText(int(text_x+1), int(text_y+1), text)
                        painter.setPen(pen) # Use the same pen as the box for text
                        painter.drawText(int(text_x), int(text_y), text)

class ScansGroupedViewer(QMainWindow):
    BOX_BORDER_THICKNESS = 4 # For on-screen display
    EXPORT_BOX_BORDER_THICKNESS = 1 # For export
    ANNOTATION_FONT_SIZE = 12
    ANNOTATION_TEXT_OFFSET_Y = 18
    SCAN_GROUPS_CONFIG = {
        "CuCaFe": {
            "name": "Cu, Ca, Fe Group",
            "ids": [
                367582, 367589, 367592, 367596, 367600, 367609, 367614, 367622,
                367630, 367634, 367638, 367641, 367646, 367653, 367658, 367663,
                367667, 367671, 367675, 367680, 367686, 367692, 367698, 367703,
                367710, 367715, 367720, 367726, 367733, 367741, 367744, 367748,
                367754, 367760, 367767, 367772, 367780, 367786, 367789, 367795,
                367798, 367803, 367807, 367813, 367816, 367819, 367825, 367830,
                367837, 367846, 367851, 367857, 367862, 367870, 367873, 367880,
                367885, 367890, 367897, 367899, 367903, 367910, 367915, 367921
            ],
            "elements": ["Cu", "Ca", "Fe"],
            "elements_map": {"r": "Cu", "g": "Ca", "b": "Fe"},
            "json_name": "unions_output.json",
            "json_path_template": os.path.join(INFO_PATH, "automap_{scan_id}"),
            "coarse_file_template": "detsum_{element}_K_norm.tiff",
            "coarse_path_template": os.path.join(RAW_SECONDARY_SCAN_DIR, "output_tiff_scan2D_{scan_id}"),
            "fine_file_template": "detsum_{element}_K_norm.tiff",
            "fine_path_template": os.path.join(RAW_SECONDARY_SCAN_DIR, "output_tiff_scan2D_{fine_id}"),
            "fine_id_logic": "range_between_coarse"
        },
        "FeCaSi": {
            "name": "Fe, Ca, Si Group",
            "ids": [
                368343, 368362, 368370, 368383, 368412, 368442, 368454, 368464, 
                368472, 368490, 368499, 368513, 368525, 368530, 368549, 368604, 
                368612, 368620, 368643, 368653, 368662, 368671, 368683, 368695, 
                368701, 368709, 368715, 368722, 368729, 368743, 368763, 368772, 
                368782, 368792, 368801, 368826, 368836, 368851, 368857, 368876, 
                368890, 368950, 368961, 368984, 369001, 369009, 369017, 369025, 
                369042, 369058, 369068, 369089, 369098, 369116, 369127, 369139, 
                369155, 369447, 369449, 369453, 369462, 369467
            ],
            "elements": ["Ca", "Fe", "Si"],
            "elements_map": {"r": "Ca", "g": "Fe", "b": "Si"},
            "json_name": "unions_output_FeCaSi.json",
            "coarse_file_template": "scan_{scan_id}_{element}.tiff",
            "coarse_path_template": os.path.join(RAW_PRIMARY_SCAN_DIR, "automap_{scan_id}"),
            "fine_file_template": "detsum_{element}_K_norm.tiff",
            "fine_path_template": os.path.join(RAW_SECONDARY_SCAN_DIR, "output_tiff_scan2D_{fine_id}"),
            "fine_id_logic": "json_boxes"
        }
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("XRF Scan Viewer")
        self.setGeometry(100, 100, 1200, 600)
        self.setStyleSheet(
            "QMainWindow{background-color:black}QLabel{color:white}QPushButton{background-color:#444;color:white;border:1px solid #666;padding:5px;border-radius:3px}QPushButton:hover{background-color:#555}QPushButton:disabled{background-color:#222;color:#888;border:1px solid #444}"
        )

        self.merged_images = {}
        self.initial_scan_id = 368304
        self.current_base_image = None
        self.current_union_data = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        header_label = QLabel("XRF Scan Viewer")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        main_layout.addWidget(header_label)

        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        main_layout.addWidget(nav_widget)
        self.prev_button = QPushButton("<- Previous")
        self.scan_id_label = QLabel("Scan ID: N/A")
        self.scan_id_label.setAlignment(Qt.AlignCenter)
        self.next_button = QPushButton("Next ->")
        nav_layout.addWidget(self.prev_button)
        nav_layout.addStretch()
        nav_layout.addWidget(self.scan_id_label)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_button)

        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        main_layout.addWidget(content_widget, 1)
        self.large_image_label = SquareLabel(self, "Large Image Placeholder")
        self.large_image_label.setAlignment(Qt.AlignCenter)
        self.large_image_label.setStyleSheet("background-color: #222; border: 1px solid #444;")
        content_layout.addWidget(self.large_image_label, 1)
        right_widget = QWidget()
        right_layout = QGridLayout(right_widget)
        self.small_image_labels = []
        for i in range(4):
            for j in range(2):
                label = SquareLabel(self, f"Small Image {i*2 + j + 1}")
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("background-color: #222; border: 1px solid #444;")
                right_layout.addWidget(label, i, j)
                self.small_image_labels.append(label)
        content_layout.addWidget(right_widget, 1)

        footer_widget = QWidget()
        footer_layout = QHBoxLayout(footer_widget)
        main_layout.addWidget(footer_widget)
        export_button = QPushButton("Export Merged Images")
        export_button.clicked.connect(self.export_images)

        self.show_box_info_checkbox = QCheckBox("Show Union Box Info")
        self.show_box_info_checkbox.setStyleSheet("color: white;")
        self.show_box_info_checkbox.toggled.connect(self.update_main_image_annotations)

        footer_layout.addStretch()
        footer_layout.addWidget(self.show_box_info_checkbox)
        footer_layout.addWidget(export_button)
        footer_layout.addStretch()

        self.prev_button.clicked.connect(self.show_previous_scan)
        self.next_button.clicked.connect(self.show_next_scan)

        # Add keyboard shortcuts for navigation
        self.shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(lambda: self.prev_button.isEnabled() and self.show_previous_scan())
        self.shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(lambda: self.next_button.isEnabled() and self.show_next_scan())

        # --- REFACTORED Data Processing and Display ---
        all_scans_dict = {}
        group_order = ["CuCaFe", "FeCaSi"]

        for group_key in group_order:
            group_config = self.SCAN_GROUPS_CONFIG[group_key]
            coarse_ids = group_config["ids"]
            if group_config["fine_id_logic"] == "range_between_coarse":
                for i, coarse_id in enumerate(coarse_ids):
                    if coarse_id in all_scans_dict: continue
                    scan_obj = XRFScan(coarse_id, group_config)
                    next_coarse_id = coarse_ids[i + 1] if i + 1 < len(coarse_ids) else coarse_id + 8
                    scan_obj.fine_scan_ids = list(range(coarse_id + 1, next_coarse_id))
                    all_scans_dict[coarse_id] = scan_obj
            else:
                for coarse_id in coarse_ids:
                    if coarse_id in all_scans_dict: continue
                    scan_obj = XRFScan(coarse_id, group_config)
                    all_scans_dict[coarse_id] = scan_obj
        
        self.available_scans = []
        self.failed_scans = []
        sorted_ids = sorted(all_scans_dict.keys())
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

        for scan_id in sorted_ids:
            scan_obj = all_scans_dict[scan_id]
            if self.gather_data_for_scan(scan_obj):
                self.available_scans.append(scan_obj)
            else:
                self.failed_scans.append(scan_id)

        self.current_scan_index = 0
        if self.available_scans:
            self.display_scan(self.available_scans[self.current_scan_index])
            if len(self.available_scans) > 1:
                self.prev_button.setEnabled(True)
                self.next_button.setEnabled(True)
        else:
            self.large_image_label.setText("No valid scan data found to display.")
            self.scan_id_label.setText("Scan ID: None")

        if self.failed_scans:
            print("\n--- Data Gathering Summary ---")
            print("Could not find source data for the following scan IDs:")
            for scan_id in self.failed_scans:
                print(f"- {scan_id}")

    def display_scan(self, scan_obj):
        self.scan_id_label.setText(f"Scan ID: {scan_obj.id} ({scan_obj.group_name})")
        self.merged_images = {}
        self.large_image_label.setPixmap(QPixmap())
        self.large_image_label.setText(f"Loading Scan {scan_obj.id}...")
        for label in self.small_image_labels:
            label.setPixmap(QPixmap())
            label.setText("")

        scan_dir = os.path.join(PROCESSED_SCANS_DIR, str(scan_obj.id))
        if not os.path.isdir(scan_dir):
            self.large_image_label.setText(f"Error: Data for scan {scan_obj.id} not found.")
            return

        self.load_and_display_merged_image(scan_dir, scan_obj)
        for i, sec_scan_id in enumerate(scan_obj.fine_scan_ids):
            if i < len(self.small_image_labels):
                self.load_and_display_secondary_image(scan_dir, sec_scan_id, scan_obj.group_config, self.small_image_labels[i])

    def show_previous_scan(self):
        self.current_scan_index -= 1
        if self.current_scan_index < 0:
            self.current_scan_index = len(self.available_scans) - 1
        self.display_scan(self.available_scans[self.current_scan_index])

    def show_next_scan(self):
        self.current_scan_index += 1
        if self.current_scan_index >= len(self.available_scans):
            self.current_scan_index = 0
        self.display_scan(self.available_scans[self.current_scan_index])

    def load_and_display_merged_image(self, scan_dir, scan_obj):
        group_config = scan_obj.group_config
        elements = group_config["elements"]
        elements_map = group_config["elements_map"]
        image_paths = {el: os.path.join(scan_dir, f"scan_{scan_obj.id}_{el}.tiff") for el in elements}

        try:
            r_img = np.array(Image.open(image_paths[elements_map["r"]]))
            g_img = np.array(Image.open(image_paths[elements_map["g"]]))
            b_img = np.array(Image.open(image_paths[elements_map["b"]]))

            def normalize(arr):
                arr = arr.astype(np.float32)
                if arr.max() > arr.min():
                    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
                return arr.astype(np.uint8)

            rgb_image_np = np.stack([normalize(r_img), normalize(g_img), normalize(b_img)], axis=-1)
            self.current_base_image = Image.fromarray(rgb_image_np, 'RGB')
            
            json_path = os.path.join(scan_dir, group_config["json_name"])
            self.current_union_data = None
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self.current_union_data = json.load(f)

            self.update_main_image_annotations()

        except FileNotFoundError as e:
            print(f"Error loading image: {e}")
            self.large_image_label.setText(f"Error: {e.filename} not found.")
            self.current_base_image = None
            self.current_union_data = None
        except Exception as e:
            print(f"An error occurred: {e}")
            self.large_image_label.setText("Error loading image.")
            self.current_base_image = None
            self.current_union_data = None

    def update_main_image_annotations(self):
        # This method no longer draws, it just passes data to the label
        if not self.current_base_image:
            return

        # The base image is converted to pixmap and set
        buffer = io.BytesIO()
        self.current_base_image.save(buffer, format='PNG')
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue(), 'PNG')
        self.large_image_label.setPixmap(pixmap)

        # Pass annotation data to the label
        self.large_image_label.setAnnotations(
            self.current_union_data,
            self.show_box_info_checkbox.isChecked()
        )

        # The self.merged_images dictionary is now only used for the small secondary scans.
        # The main image for export is generated on-demand in export_images().

    def generate_export_image(self):
        if not self.current_base_image:
            return None

        # Create a QImage to draw on at the native resolution of the image.
        width = self.current_base_image.width
        height = self.current_base_image.height
        export_image = QImage(width, height, QImage.Format_ARGB32)
        export_image.fill(Qt.transparent)

        # Create a QPainter
        painter = QPainter(export_image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.HighQualityAntialiasing, True)

        # Draw the base image
        buffer = io.BytesIO()
        self.current_base_image.save(buffer, format='PNG')
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue(), 'PNG')
        painter.drawPixmap(0, 0, pixmap)

        # Draw annotations (similar to SquareLabel.paintEvent but without scaling)
        if self.current_union_data:
            pen = QPen(Qt.white)
            pen.setWidth(self.EXPORT_BOX_BORDER_THICKNESS) # Use export-specific variable
            painter.setPen(pen)
            
            font = QFont()
            font.setPointSize(self.ANNOTATION_FONT_SIZE)
            painter.setFont(font)

            for key, item in self.current_union_data.items():
                if 'image_center' in item and 'image_length' in item:
                    x, y = item['image_center']
                    length = item['image_length']
                    if x < 0 or y < 0:
                        x += width / 2
                        y += height / 2
                    
                    half_length = length / 2
                    box_x = x - half_length
                    box_y = y - half_length
                    
                    painter.drawRect(int(box_x), int(box_y), int(length), int(length))


        
        painter.end()
        return export_image

    def load_and_display_secondary_image(self, scan_dir, scan_id, group_config, target_label):
        elements = group_config["elements"]
        elements_map = group_config["elements_map"]
        image_paths = {el: os.path.join(scan_dir, f"detsum_{scan_id}_{el}.tiff") for el in elements}

        try:
            r_img = np.array(Image.open(image_paths[elements_map["r"]]))
            g_img = np.array(Image.open(image_paths[elements_map["g"]]))
            b_img = np.array(Image.open(image_paths[elements_map["b"]]))

            def normalize(arr):
                arr = arr.astype(np.float32)
                if arr.max() > arr.min():
                    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
                return arr.astype(np.uint8)

            rgb_image_np = np.stack([normalize(r_img), normalize(g_img), normalize(b_img)], axis=-1)
            pil_img = Image.fromarray(rgb_image_np, 'RGB')
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue(), 'PNG')
            target_label.setPixmap(pixmap)
            self.merged_images[scan_id] = pil_img
        except FileNotFoundError as e:
            print(f"Error loading secondary image for scan {scan_id}: {e}")
            target_label.setText(f"Not Found")
        except Exception as e:
            print(f"An error occurred while loading secondary image {scan_id}: {e}")
            target_label.setText("Error")

    def copy_scan_data(self, scan_obj):
        group_config = scan_obj.group_config
        scan_id = scan_obj.id
        src_parent_dir = group_config["coarse_path_template"].format(scan_id=scan_id)
        if not os.path.isdir(src_parent_dir):
            print(f"Source directory not found: {src_parent_dir}")
            return None

        scan_dest_dir = os.path.join(PROCESSED_SCANS_DIR, str(scan_id))
        os.makedirs(scan_dest_dir, exist_ok=True)

        for element in group_config["elements"]:
            src_file_name = group_config["coarse_file_template"].format(scan_id=scan_id, element=element)
            src_path = os.path.join(src_parent_dir, src_file_name)
            dest_path = os.path.join(scan_dest_dir, f"scan_{scan_id}_{element}.tiff")
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
            else:
                print(f"Coarse scan source file not found: {src_path}")
                return None # If a coarse scan file is missing, we can't proceed

        json_src_parent_dir = group_config.get("json_path_template", src_parent_dir).format(scan_id=scan_id)
        json_src_path = os.path.join(json_src_parent_dir, group_config["json_name"])
        if os.path.exists(json_src_path):
            shutil.copy(json_src_path, os.path.join(scan_dest_dir, group_config["json_name"]))
        else:
            print(f"JSON file not found: {json_src_path}")
        
        return scan_dest_dir

    def copy_secondary_scans(self, scan_obj):
        group_config = scan_obj.group_config
        dest_dir = os.path.join(PROCESSED_SCANS_DIR, str(scan_obj.id))

        for fine_id in scan_obj.fine_scan_ids:
            src_dir = group_config["fine_path_template"].format(fine_id=fine_id)
            if not os.path.isdir(src_dir):
                print(f"Fine scan source directory not found: {src_dir}")
                continue
            
            for element in group_config["elements"]:
                src_file_name = group_config["fine_file_template"].format(element=element)
                src_path = os.path.join(src_dir, src_file_name)
                dest_file_name = f"detsum_{fine_id}_{element}.tiff"
                dest_path = os.path.join(dest_dir, dest_file_name)
                if os.path.exists(src_path):
                    shutil.copy(src_path, dest_path)
                else:
                    print(f"Fine scan source file not found: {src_path}")

    def gather_data_for_scan(self, scan_obj):
        print(f"--- Gathering data for Scan ID: {scan_obj.id} ({scan_obj.group_name}) ---")
        scan_dir = self.copy_scan_data(scan_obj)
        if not scan_dir:
            print(f"--- Failed to find primary data for Scan ID: {scan_obj.id} ---")
            return False

        if scan_obj.group_config["fine_id_logic"] == "json_boxes":
            json_path = os.path.join(scan_dir, scan_obj.group_config["json_name"])
            num_boxes = 0
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        union_data = json.load(f)
                        num_boxes = len(union_data)
                except (IOError, json.JSONDecodeError) as e:
                    print(f"Error reading or parsing JSON for scan {scan_obj.id}: {e}")
            scan_obj.fine_scan_ids = [scan_obj.id + i for i in range(1, num_boxes + 1)]
        
        if scan_obj.fine_scan_ids:
            self.copy_secondary_scans(scan_obj)
        print(f"--- Finished gathering data for Scan ID: {scan_obj.id} ---")
        return True

    def export_images(self):
        if not self.available_scans:
            print("No images to export.")
            return

        current_scan_obj = self.available_scans[self.current_scan_index]
        base_save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Images")

        if base_save_dir:
            folder_name = f"scan_{current_scan_obj.id}_Coarse_Fine_Scans"
            save_dir = os.path.join(base_save_dir, folder_name)
            os.makedirs(save_dir, exist_ok=True)

            # Export the main image with high quality rendering
            main_export_image = self.generate_export_image()
            if main_export_image:
                file_name = f"merged_scan_{current_scan_obj.id}.png"
                save_path = os.path.join(save_dir, file_name)
                try:
                    main_export_image.save(save_path)
                    print(f"Saved image to {save_path}")
                except Exception as e:
                    print(f"Error saving image {save_path}: {e}")

            # Export the small secondary images (still using PIL for these)
            for scan_id, pil_img in self.merged_images.items():
                if scan_id != current_scan_obj.id:
                    file_name = f"merged_detsum_{scan_id}.png"
                    save_path = os.path.join(save_dir, file_name)
                    try:
                        pil_img.save(save_path)
                        print(f"Saved image to {save_path}")
                    except Exception as e:
                        print(f"Error saving image {save_path}: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ScansGroupedViewer()
    viewer.show()
    sys.exit(app.exec_())