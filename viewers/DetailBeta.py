import sys
import os
import shutil
import json
import io
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QSizePolicy, QPushButton, QFileDialog, QShortcut, QCheckBox
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap, QImage, QPainter, QKeySequence, QFont, QPen

# --- File Path Configuration ---
CONFIG_FILENAME = "ScanDetailViewerFilePath.json"
PATH_CONFIG_FILE = os.path.join(os.path.dirname(__file__), CONFIG_FILENAME)

DEFAULT_PATH_CONFIG = {
    "PROCESSED_SCANS_DIR": "/home/codingcarlos/Documents/github/SULI-2025-Summer/data/scans_grouped",
    "RAW_PRIMARY_SCAN_DIR": "/home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3",
    "RAW_SECONDARY_SCAN_DIR": "/home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3/all_xrf",
    "INFO_PATH": "/home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3/data/user_macros",
}


def load_path_config():
    """Load path overrides from ScanDetailViewerFilePath.json if available."""
    config = DEFAULT_PATH_CONFIG.copy()
    try:
        with open(PATH_CONFIG_FILE, "r") as fp:
            file_config = json.load(fp)
            if isinstance(file_config, dict):
                for key, value in file_config.items():
                    if key in config and isinstance(value, str) and value:
                        config[key] = value
            else:
                print(f"Warning: {CONFIG_FILENAME} must contain a JSON object; using defaults.")
    except FileNotFoundError:
        print(f"Warning: {CONFIG_FILENAME} not found; using default path configuration.")
    except json.JSONDecodeError as exc:
        print(f"Warning: Could not parse {CONFIG_FILENAME}: {exc}; using defaults.")
    return config


PATH_CONFIG = load_path_config()
PROCESSED_SCANS_DIR = PATH_CONFIG["PROCESSED_SCANS_DIR"]
RAW_PRIMARY_SCAN_DIR = PATH_CONFIG["RAW_PRIMARY_SCAN_DIR"]
RAW_SECONDARY_SCAN_DIR = PATH_CONFIG["RAW_SECONDARY_SCAN_DIR"]
INFO_PATH = PATH_CONFIG["INFO_PATH"]

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
    TOGETHER_PAIRWISE_THRESHOLD = 0.90
    TOGETHER_TRIO_THRESHOLD = 0.80
    TOGETHER_CHANNEL_COVERAGE_THRESHOLD = 0.85
    TOGETHER_COMPONENT_DOMINANCE = 0.95
    SEPARATE_PAIRWISE_THRESHOLD = 0.90
    SEPARATE_THIRD_OVERLAP_THRESHOLD = 0.15
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
            "folder_suffix": "CuCaFe",
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
            "folder_suffix": "FeCaSi",
            "json_name": "unions_output_FeCaSi.json",
            "coarse_file_template": "scan_{scan_id}_{element}.tiff",
            "coarse_path_template": os.path.join(RAW_PRIMARY_SCAN_DIR, "automap_{scan_id}"),
            "fine_file_template": "detsum_{element}_K_norm.tiff",
            "fine_path_template": os.path.join(RAW_SECONDARY_SCAN_DIR, "output_tiff_scan2D_{fine_id}"),
            "fine_id_logic": "json_boxes"
        },
        "CrFeMn": {
            "name": "Cr, Fe, Mn Group",
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
            "elements": ["Cr", "Fe", "Mn"],
            "elements_map": {"r": "Cr", "g": "Fe", "b": "Mn"},
            "folder_suffix": "CrFeMn",
            "json_name": "unions_output.json",
            "json_path_template": os.path.join(INFO_PATH, "automap_{scan_id}"),
            "coarse_file_template": "detsum_{element}_K_norm.tiff",
            "coarse_path_template": os.path.join(RAW_SECONDARY_SCAN_DIR, "output_tiff_scan2D_{scan_id}"),
            "fine_file_template": "detsum_{element}_K_norm.tiff",
            "fine_path_template": os.path.join(RAW_SECONDARY_SCAN_DIR, "output_tiff_scan2D_{fine_id}"),
            "fine_id_logic": "range_between_coarse"
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
        self.secondary_images = {}
        self.initial_scan_id = 368304
        self.current_base_image = None
        self.current_plain_base_image = None
        self.current_outlined_base_image = None
        self.current_rgb_image_np = None
        self.outlines_enabled = True
        self.channel_outline_enabled = [True, True, True]  # R, G, B
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

        stats_widget = QWidget()
        stats_layout = QHBoxLayout(stats_widget)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(30)
        main_layout.addWidget(stats_widget)
        stats_layout.addStretch()
        self.scan_overlap_labels = {}
        for stat_name in ("Separate", "Partial", "Together"):
            stat_container = QWidget()
            stat_container_layout = QVBoxLayout(stat_container)
            stat_container_layout.setContentsMargins(10, 5, 10, 5)
            stat_container_layout.setSpacing(2)

            label_widget = QLabel(stat_name)
            label_widget.setAlignment(Qt.AlignCenter)
            label_widget.setStyleSheet("font-size: 12px; color: #bbb;")

            value_label = QLabel("0")
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setStyleSheet("font-size: 18px; font-weight: bold;")

            stat_container_layout.addWidget(label_widget)
            stat_container_layout.addWidget(value_label)
            stats_layout.addWidget(stat_container)
            self.scan_overlap_labels[stat_name] = value_label
        stats_layout.addStretch()
        self.overlap_counts = {key: 0 for key in self.scan_overlap_labels}
        self.scan_overlap_results = {}
        self.refresh_overlap_labels()

        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        main_layout.addWidget(content_widget, 1)
        self.large_image_label = SquareLabel(self, "Large Image Placeholder")
        self.large_image_label.setAlignment(Qt.AlignCenter)
        self.large_image_label.setStyleSheet("background-color: #222; border: 1px solid #444;")
        content_layout.addWidget(self.large_image_label, 1)
        right_widget = QWidget()
        right_layout = QGridLayout(right_widget)
        self.small_image_displays = []
        for i in range(4):
            for j in range(2):
                cell_widget = QWidget()
                cell_layout = QVBoxLayout(cell_widget)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setSpacing(4)

                image_label = SquareLabel(self, f"Small Image {i*2 + j + 1}")
                image_label.setAlignment(Qt.AlignCenter)
                image_label.setStyleSheet("background-color: #222; border: 1px solid #444;")

                type_label = QLabel("Type: N/A")
                type_label.setAlignment(Qt.AlignCenter)
                type_label.setStyleSheet("color: #ccc; font-size: 12px;")

                cell_layout.addWidget(image_label)
                cell_layout.addWidget(type_label)
                right_layout.addWidget(cell_widget, i, j)
                self.small_image_displays.append({
                    "image": image_label,
                    "type": type_label
                })
        content_layout.addWidget(right_widget, 1)

        footer_widget = QWidget()
        footer_layout = QHBoxLayout(footer_widget)
        main_layout.addWidget(footer_widget)
        export_button = QPushButton("Export Merged Images")
        export_button.clicked.connect(self.export_images)

        self.toggle_outlines_button = QPushButton("Hide Outlines")
        self.toggle_outlines_button.setCheckable(True)
        self.toggle_outlines_button.setChecked(True)
        self.toggle_outlines_button.clicked.connect(self.toggle_outlines)

        self.channel_toggle_red = QCheckBox("Red")
        self.channel_toggle_red.setChecked(True)
        self.channel_toggle_red.setStyleSheet("color: red;")
        self.channel_toggle_red.toggled.connect(self.update_channel_outlines)
        self.channel_toggle_green = QCheckBox("Green")
        self.channel_toggle_green.setChecked(True)
        self.channel_toggle_green.setStyleSheet("color: #5bff5b;")
        self.channel_toggle_green.toggled.connect(self.update_channel_outlines)
        self.channel_toggle_blue = QCheckBox("Blue")
        self.channel_toggle_blue.setChecked(True)
        self.channel_toggle_blue.setStyleSheet("color: #66aaff;")
        self.channel_toggle_blue.toggled.connect(self.update_channel_outlines)

        self.show_box_info_checkbox = QCheckBox("Show Union Box Info")
        self.show_box_info_checkbox.setStyleSheet("color: white;")
        self.show_box_info_checkbox.toggled.connect(self.update_main_image_annotations)

        footer_layout.addStretch()
        footer_layout.addWidget(self.toggle_outlines_button)
        footer_layout.addWidget(self.channel_toggle_red)
        footer_layout.addWidget(self.channel_toggle_green)
        footer_layout.addWidget(self.channel_toggle_blue)
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
        all_scans = []
        group_order = ["CuCaFe", "FeCaSi", "CrFeMn"]

        for group_key in group_order:
            group_config = self.SCAN_GROUPS_CONFIG[group_key]
            coarse_ids = group_config["ids"]
            if group_config["fine_id_logic"] == "range_between_coarse":
                for i, coarse_id in enumerate(coarse_ids):
                    scan_obj = XRFScan(coarse_id, group_config)
                    next_coarse_id = coarse_ids[i + 1] if i + 1 < len(coarse_ids) else coarse_id + 8
                    scan_obj.fine_scan_ids = list(range(coarse_id + 1, next_coarse_id))
                    all_scans.append(scan_obj)
            else:
                for coarse_id in coarse_ids:
                    scan_obj = XRFScan(coarse_id, group_config)
                    all_scans.append(scan_obj)
        
        self.available_scans = []
        self.failed_scans = []
        sorted_scans = sorted(all_scans, key=lambda scan: scan.id)
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

        for scan_obj in sorted_scans:
            if self.gather_data_for_scan(scan_obj):
                self.available_scans.append(scan_obj)
            else:
                self.failed_scans.append(f"{scan_obj.id} ({scan_obj.group_name})")

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
            for scan_desc in self.failed_scans:
                print(f"- {scan_desc}")

    def display_scan(self, scan_obj):
        self.scan_id_label.setText(f"Scan ID: {scan_obj.id} ({scan_obj.group_name})")
        self.merged_images = {}
        self.secondary_images = {}
        self.current_plain_base_image = None
        self.current_outlined_base_image = None
        self.current_base_image = None
        self.current_rgb_image_np = None
        self.large_image_label.setPixmap(QPixmap())
        self.large_image_label.setText(f"Loading Scan {scan_obj.id}...")
        for slot in self.small_image_displays:
            slot["image"].setPixmap(QPixmap())
            slot["image"].setText("")
            slot["type"].setText("Type: N/A")
            slot["scan_id"] = None
        self.reset_overlap_counts()

        scan_dir = self.get_processed_scan_dir(scan_obj)
        if not os.path.isdir(scan_dir):
            self.large_image_label.setText(f"Error: Data for scan {scan_obj.id} not found.")
            return

        self.load_and_display_merged_image(scan_dir, scan_obj)
        for i, sec_scan_id in enumerate(scan_obj.fine_scan_ids):
            if i < len(self.small_image_displays):
                self.load_and_display_secondary_image(scan_dir, sec_scan_id, scan_obj.group_config, self.small_image_displays[i])

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

    def get_processed_scan_dir(self, scan_obj, allow_legacy=True):
        """Return the on-disk directory for the processed scan."""
        suffix = scan_obj.group_config.get("folder_suffix") or "".join(scan_obj.group_config["elements"])
        preferred_dir = os.path.join(PROCESSED_SCANS_DIR, f"{scan_obj.id}_{suffix}")
        if not allow_legacy:
            return preferred_dir

        legacy_dir = os.path.join(PROCESSED_SCANS_DIR, str(scan_obj.id))
        if os.path.isdir(preferred_dir) or not os.path.isdir(legacy_dir):
            return preferred_dir
        return legacy_dir

    def reset_overlap_counts(self):
        for key in self.overlap_counts:
            self.overlap_counts[key] = 0
        self.scan_overlap_results.clear()
        self.refresh_overlap_labels()

    def refresh_overlap_labels(self):
        for key, label in self.scan_overlap_labels.items():
            label.setText(str(self.overlap_counts.get(key, 0)))

    def record_scan_classification(self, scan_id, classification):
        previous = self.scan_overlap_results.get(scan_id)
        if previous == classification:
            return
        if previous in self.overlap_counts:
            self.overlap_counts[previous] = max(0, self.overlap_counts[previous] - 1)
        if classification in self.overlap_counts:
            self.overlap_counts[classification] += 1
            self.scan_overlap_results[scan_id] = classification
        else:
            self.scan_overlap_results.pop(scan_id, None)
        self.refresh_overlap_labels()

    def analyze_components(self, mask):
        total = int(mask.sum())
        if total == 0:
            return {"count": 0, "largest_fraction": 0.0, "total": 0}
        mask_uint8 = mask.astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        if num_labels <= 1:
            return {"count": 0, "largest_fraction": 0.0, "total": total}
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(areas.max()) if areas.size else 0
        return {
            "count": len(areas),
            "largest_fraction": (largest / total) if total else 0.0,
            "total": total,
        }

    def outline_top_blobs(self, rgb_image_np, max_blobs=3):
        """Detect and outline the largest blobs in the image."""
        outlined = rgb_image_np.copy()
        gray = cv2.cvtColor(rgb_image_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return outlined

        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_blobs]
        for contour in sorted_contours:
            if cv2.contourArea(contour) <= 0:
                continue
            cv2.drawContours(outlined, [contour], -1, (255, 0, 255), 2, lineType=cv2.LINE_AA)

        return outlined

    def outline_channel_largest_blobs(self, rgb_image_np, channel_enabled=None):
        """Outline the largest blob of each channel in red, green, and blue."""
        outlined = rgb_image_np.copy()
        channel_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # RGB order
        if channel_enabled is None:
            channel_enabled = [True, True, True]

        for channel_index, color in enumerate(channel_colors):
            if channel_index >= len(channel_enabled) or not channel_enabled[channel_index]:
                continue
            channel = rgb_image_np[:, :, channel_index].astype(np.uint8)
            if channel.size == 0:
                continue

            blurred = cv2.GaussianBlur(channel, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) <= 0:
                continue

            cv2.drawContours(outlined, [largest_contour], -1, color, 2, lineType=cv2.LINE_AA)

        return outlined

    def get_active_base_image(self):
        """Return the base image respecting the outlines toggle."""
        if self.outlines_enabled:
            return self.current_outlined_base_image
        return self.current_plain_base_image

    def set_label_pixmap(self, label, pil_img):
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue(), 'PNG')
        label.setPixmap(pixmap)

    def largest_component_mask(self, channel):
        """Return a binary mask for the largest connected component in the channel."""
        if channel.size == 0:
            return None, 0
        channel = channel.astype(np.float32)
        cmax, cmin = channel.max(), channel.min()
        if cmax > cmin:
            channel = (channel - cmin) / (cmax - cmin) * 255.0
        channel_u8 = channel.astype(np.uint8)
        blurred = cv2.GaussianBlur(channel_u8, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        if num_labels <= 1:
            return None, 0
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + np.argmax(areas)
        largest_mask = (labels == largest_idx).astype(np.uint8)
        h, w = largest_mask.shape
        min_dim = min(h, w)
        if min_dim <= 35:
            kernel = np.ones((5, 5), np.uint8)
            iterations = 4
            largest_mask = cv2.dilate(largest_mask, kernel, iterations=iterations)
        elif min_dim <= 45:
            kernel = np.ones((5, 5), np.uint8)
            iterations = 3
            largest_mask = cv2.dilate(largest_mask, kernel, iterations=iterations)
        else:
            # Larger tiles already have more pixels; avoid over-dilation
            pass
        largest_area = int(largest_mask.sum())
        return largest_mask, largest_area

    # Prefer algorithmic classification; overrides kept empty unless a specific case truly requires it.
    CLASSIFICATION_OVERRIDES = {}

    def typeDetector(self, channel_images, fine_id=None, elements=None):
        """Classify overlap based on largest blobs: Separate (no touch), Partial (any pair overlaps), Together (all three intersect and overlap is strong).

        The thresholds below intentionally down-weight tiny overlaps so noisy contacts
        don't trigger "Together" when phases are really separate.
        """
        elements_str = "".join(elements) if isinstance(elements, (list, tuple)) else (elements or "")

        # Baseline thresholds; relaxed for element sets with weaker signals (e.g. Cr/Fe/Mn)
        MIN_PAIR_REL = 0.20       # minimum overlap relative to smaller blob to count as a pair overlap
        MIN_PAIR_PIXELS = 70      # minimum pixels in overlap region to count the pair
        MIN_TRIO_REL = 0.05       # minimum triple-overlap fraction vs smallest blob
        MIN_TRIO_PIXELS = 30      # minimum triple-overlap pixels
        STRONG_COVERAGE = 0.45    # per-channel coverage threshold to call overlap strong

        if "CrFeMn" in elements_str:
            # Cr/Fe/Mn scans tend to be dimmer and fragmented; lower the bar for overlaps
            MIN_PAIR_REL = 0.12
            MIN_PAIR_PIXELS = 40
            MIN_TRIO_REL = 0.03
            MIN_TRIO_PIXELS = 15
            STRONG_COVERAGE = 0.32
        elif "FeCaSi" in elements_str:
            # Fe/Ca/Si often has weaker overlaps than Cu/Ca/Fe
            MIN_PAIR_REL = 0.15
            MIN_PAIR_PIXELS = 50
            MIN_TRIO_REL = 0.04
            MIN_TRIO_PIXELS = 20
            STRONG_COVERAGE = 0.38
        if fine_id in self.CLASSIFICATION_OVERRIDES:
            return self.CLASSIFICATION_OVERRIDES[fine_id]
        if not channel_images or len(channel_images) < 3:
            return None

        largest_masks = []
        areas = []
        for channel in channel_images:
            mask, area = self.largest_component_mask(channel)
            largest_masks.append(mask)
            areas.append(area)

        if any(area == 0 or mask is None for mask, area in zip(largest_masks, areas)):
            return "Separate"

        trio_intersection = np.logical_and.reduce(largest_masks)
        trio_area = int(trio_intersection.sum())
        coverages = [(trio_area / area) if area else 0.0 for area in areas]

        pair_areas = []
        pair_strengths = []
        pair_jaccards = []
        pair_min_areas = []
        for i in range(3):
            for j in range(i + 1, 3):
                pair_area = int(np.logical_and(largest_masks[i], largest_masks[j]).sum())
                pair_areas.append(pair_area)
                min_area = min(areas[i], areas[j]) if areas[i] and areas[j] else 0
                strength = (pair_area / min_area) if min_area else 0.0
                pair_strengths.append(strength)
                pair_min_areas.append(min_area)
                union = areas[i] + areas[j] - pair_area
                pair_jaccards.append((pair_area / union) if union else 0.0)

        def pair_counts_as_overlap(area, strength):
            return area >= MIN_PAIR_PIXELS and strength >= MIN_PAIR_REL

        strong_pair_flags = [pair_counts_as_overlap(a, s) for a, s in zip(pair_areas, pair_strengths)]
        pair_overlap_exists = any(strong_pair_flags)
        all_pairs_overlap = all(strong_pair_flags) if strong_pair_flags else False
        strong_pair_count = sum(1 for flag in strong_pair_flags if flag)

        min_area = min(areas) if areas else 0
        trio_strong_enough = trio_area >= MIN_TRIO_PIXELS and (trio_area / min_area >= MIN_TRIO_REL if min_area else False)
        strong_overlap_pairs = sum(1 for c in coverages if c >= STRONG_COVERAGE)

        max_strength = max(pair_strengths) if pair_strengths else 0.0
        max_jaccard = max(pair_jaccards) if pair_jaccards else 0.0
        max_cover = max(coverages) if coverages else 0.0

        # Together heuristics:
        # 1) Very strong pair overlap with low triple coverage (tight overlap of two phases pulling the third in)
        if strong_pair_count >= 2 and max_strength >= 0.85 and max_jaccard >= 0.35 and max_cover <= 0.32:
            return "Together"
        # 2) Multiple strong pairs with minimal shared area (still indicates co-location)
        if strong_pair_count >= 2 and max_cover <= 0.08 and max_strength >= 0.60:
            return "Together"
        # 3) Triple overlap is sizeable and at least two channels have strong coverage
        if trio_strong_enough and strong_pair_count >= 2 and strong_overlap_pairs >= 2:
            return "Together"
        # 4) Element-set specific relaxation (e.g. CrFeMn): if two overlaps exist and overall signals are modest, still treat as Together
        if "CrFeMn" in elements_str and strong_pair_count >= 2:
            if (max_strength >= 0.35 and max_jaccard >= 0.08) or trio_area >= MIN_TRIO_PIXELS:
                return "Together"

        # Partial if a single strong pair exists with clear overlap but not enough to be Together
        if strong_pair_count == 1 and trio_area <= MIN_TRIO_PIXELS * 2:
            # Use the strongest pair to decide
            best_idx = max(range(len(pair_strengths)), key=lambda k: pair_strengths[k]) if pair_strengths else 0
            best_j = pair_jaccards[best_idx] if pair_jaccards else 0.0
            best_s = pair_strengths[best_idx] if pair_strengths else 0.0
            best_min = pair_min_areas[best_idx] if pair_min_areas else 0
            best_area = pair_areas[best_idx] if pair_areas else 0

            high_jaccard_partial = trio_area == 0 and best_j >= 0.60 and best_s <= 1.05
            very_strong_overlap = best_s >= 0.95 and best_j >= 0.20
            strong_small_min = best_s >= 0.85 and best_min <= 160 and best_area >= 60
            mid_jaccard_large_area = best_j >= 0.45 and best_min <= 450 and best_area >= 250
            tiny_overlap_small_min = best_j >= 0.09 and best_s >= 0.20 and best_min <= 120 and best_area >= 20

            if high_jaccard_partial or very_strong_overlap or strong_small_min or mid_jaccard_large_area or tiny_overlap_small_min:
                return "Partial"

        # Weak single overlaps that still indicate partial contact
        if strong_pair_count == 0 and trio_area <= MIN_TRIO_PIXELS:
            best_idx = max(range(len(pair_strengths)), key=lambda k: pair_strengths[k]) if pair_strengths else 0
            best_j = pair_jaccards[best_idx] if pair_jaccards else 0.0
            best_s = pair_strengths[best_idx] if pair_strengths else 0.0
            best_min = pair_min_areas[best_idx] if pair_min_areas else 0
            best_area = pair_areas[best_idx] if pair_areas else 0
            if best_area >= 20 and best_j >= 0.08 and best_j < 0.70 and best_s >= 0.18 and best_min <= 140 and best_min >= MIN_PAIR_PIXELS:
                return "Partial"

        # Partial if at least two strong overlaps exist, or one strong overlap with notable shared coverage
        if (strong_pair_count >= 2 and max_cover >= 0.18) or (strong_pair_count == 1 and max_cover >= 0.15):
            return "Partial"

        # CrFeMn fallback: if any measurable trio overlap exists with modest pair strength, lean Together
        if "CrFeMn" in elements_str and trio_area > 0 and max_strength >= 0.18:
            return "Together"

        # Element-set fallbacks for weak signals: classify as Partial on modest overlaps
        if "CrFeMn" in elements_str or "FeCaSi" in elements_str:
            low_overlap_pair = any(area >= 15 and strength >= 0.06 for area, strength in zip(pair_areas, pair_strengths))
            if low_overlap_pair or (trio_area > 0 and max_strength >= 0.08):
                return "Partial"

        return "Separate"

    def toggle_outlines(self, checked):
        self.outlines_enabled = checked
        self.toggle_outlines_button.setText("Hide Outlines" if checked else "Show Outlines")
        self.rebuild_outlines_for_all()
        self.current_base_image = self.get_active_base_image()
        self.update_main_image_annotations()
        self.refresh_secondary_outline_display()

    def refresh_secondary_outline_display(self):
        for slot in self.small_image_displays:
            scan_id = slot.get("scan_id")
            if not scan_id:
                continue
            images = self.secondary_images.get(scan_id)
            if not images:
                continue
            active_image = images["outlined"] if self.outlines_enabled else images["plain"]
            self.set_label_pixmap(slot["image"], active_image)
            self.merged_images[scan_id] = active_image

    def update_channel_outlines(self):
        self.channel_outline_enabled = [
            self.channel_toggle_red.isChecked(),
            self.channel_toggle_green.isChecked(),
            self.channel_toggle_blue.isChecked(),
        ]
        self.rebuild_outlines_for_all()
        self.current_base_image = self.get_active_base_image()
        self.update_main_image_annotations()
        self.refresh_secondary_outline_display()

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
            self.current_rgb_image_np = rgb_image_np
            outlined_image_np = self.outline_channel_largest_blobs(rgb_image_np, self.channel_outline_enabled)
            self.current_plain_base_image = Image.fromarray(rgb_image_np, 'RGB')
            self.current_outlined_base_image = Image.fromarray(outlined_image_np, 'RGB')
            self.current_base_image = self.get_active_base_image()
            
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
            self.current_plain_base_image = None
            self.current_outlined_base_image = None
            self.current_union_data = None
        except Exception as e:
            print(f"An error occurred: {e}")
            self.large_image_label.setText("Error loading image.")
            self.current_base_image = None
            self.current_rgb_image_np = None
            self.current_plain_base_image = None
            self.current_outlined_base_image = None
            self.current_union_data = None

    def update_main_image_annotations(self):
        # This method no longer draws, it just passes data to the label
        base_image = self.get_active_base_image()
        if base_image is None:
            return
        self.current_base_image = base_image

        self.set_label_pixmap(self.large_image_label, base_image)

        # Pass annotation data to the label
        self.large_image_label.setAnnotations(
            self.current_union_data,
            self.show_box_info_checkbox.isChecked()
        )

        # The self.merged_images dictionary is now only used for the small secondary scans.
        # The main image for export is generated on-demand in export_images().

    def rebuild_outlines_for_all(self):
        """Recompute outlined images based on current channel toggle selections."""
        if self.current_rgb_image_np is not None:
            outlined_np = self.outline_channel_largest_blobs(self.current_rgb_image_np, self.channel_outline_enabled)
            self.current_outlined_base_image = Image.fromarray(outlined_np, 'RGB')
        for scan_id, images in self.secondary_images.items():
            rgb_np = images.get("rgb_np")
            if rgb_np is None:
                continue
            outlined_np = self.outline_channel_largest_blobs(rgb_np, self.channel_outline_enabled)
            images["outlined"] = Image.fromarray(outlined_np, 'RGB')

    def generate_export_image(self):
        base_image = self.get_active_base_image()
        if not base_image:
            return None

        # Create a QImage to draw on at the native resolution of the image.
        width = base_image.width
        height = base_image.height
        export_image = QImage(width, height, QImage.Format_ARGB32)
        export_image.fill(Qt.transparent)

        # Create a QPainter
        painter = QPainter(export_image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.HighQualityAntialiasing, True)

        # Draw the base image
        buffer = io.BytesIO()
        base_image.save(buffer, format='PNG')
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

    def load_and_display_secondary_image(self, scan_dir, scan_id, group_config, target_slot):
        elements = group_config["elements"]
        elements_map = group_config["elements_map"]
        image_paths = {el: os.path.join(scan_dir, f"detsum_{scan_id}_{el}.tiff") for el in elements}
        image_label = target_slot["image"]
        type_label = target_slot["type"]
        type_label.setText("Type: ...")

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
            outlined_image_np = self.outline_channel_largest_blobs(rgb_image_np, self.channel_outline_enabled)
            pil_plain = Image.fromarray(rgb_image_np, 'RGB')
            pil_outlined = Image.fromarray(outlined_image_np, 'RGB')

            # Store for exports and toggles
            self.secondary_images[scan_id] = {
                "plain": pil_plain,
                "outlined": pil_outlined,
                "rgb_np": rgb_image_np,
            }

            active_image = pil_outlined if self.outlines_enabled else pil_plain
            self.set_label_pixmap(image_label, active_image)
            elements_str = "".join(elements)
            classification = self.typeDetector([r_img, g_img, b_img], fine_id=scan_id, elements=elements_str)
            type_label.setText(f"Type: {classification or 'Unknown'}")
            self.record_scan_classification(scan_id, classification)
            # Keep merged_images aligned with what is shown for export compatibility
            self.merged_images[scan_id] = active_image
            target_slot["scan_id"] = scan_id
        except FileNotFoundError as e:
            print(f"Error loading secondary image for scan {scan_id}: {e}")
            image_label.setPixmap(QPixmap())
            image_label.setText("Not Found")
            type_label.setText("Type: Missing")
            self.record_scan_classification(scan_id, None)
            target_slot["scan_id"] = None
        except Exception as e:
            print(f"An error occurred while loading secondary image {scan_id}: {e}")
            image_label.setPixmap(QPixmap())
            image_label.setText("Error")
            type_label.setText("Type: Error")
            self.record_scan_classification(scan_id, None)
            target_slot["scan_id"] = None

    def copy_scan_data(self, scan_obj):
        group_config = scan_obj.group_config
        scan_id = scan_obj.id
        src_parent_dir = group_config["coarse_path_template"].format(scan_id=scan_id)
        if not os.path.isdir(src_parent_dir):
            print(f"Source directory not found: {src_parent_dir}")
            return None

        scan_dest_dir = self.get_processed_scan_dir(scan_obj, allow_legacy=False)
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
        dest_dir = self.get_processed_scan_dir(scan_obj, allow_legacy=False)

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
