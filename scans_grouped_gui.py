import sys
import os
import shutil
import json
import io
import numpy as np
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QSizePolicy, QPushButton, QFileDialog
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap, QImage, QPainter

# --- File Path Constants ---
PROCESSED_SCANS_DIR = "/home/codingcarlos/Documents/github/SULI-2025-Summer/data/scans_grouped"
RAW_PRIMARY_SCAN_DIR = "/home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3"
RAW_SECONDARY_SCAN_DIR = os.path.join(RAW_PRIMARY_SCAN_DIR, "all_xrf")

class SquareLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event):
        super().paintEvent(event)

        pix = self.pixmap()
        if pix is None or pix.isNull():
            return

        # Get the widget's dimensions
        widget_width = self.width()
        widget_height = self.height()

        # Determine the size of the square
        size = min(widget_width, widget_height)

        # Calculate the top-left corner to center the square
        x = (widget_width - size) // 2
        y = (widget_height - size) // 2

        # Define the target rectangle for drawing
        target_rect = QRect(x, y, size, size)

        # Scale the pixmap smoothly to the target size
        scaled_pix = pix.scaled(target_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # Create a painter and draw the pre-scaled pixmap
        painter = QPainter(self)
        painter.drawPixmap(target_rect, scaled_pix)


class ScansGroupedViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XRF Scan Viewer")
        self.setGeometry(100, 100, 1200, 600)

        # To store PIL images for export
        self.merged_images = {}
        self.initial_scan_id = 368304 # Store the initial scan ID

        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Header
        header_label = QLabel("XRF Scan Viewer")
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")
        main_layout.addWidget(header_label)

        # Navigation
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


        # Content (Image panels)
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        main_layout.addWidget(content_widget, 1) # Set stretch factor to 1

        # Left side: Large image placeholder
        self.large_image_label = SquareLabel("Large Image Placeholder")
        self.large_image_label.setAlignment(Qt.AlignCenter)
        self.large_image_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        content_layout.addWidget(self.large_image_label, 1) # Add directly to the horizontal layout

        # Right side: Four small images placeholder
        right_widget = QWidget()
        right_layout = QGridLayout(right_widget)
        self.small_image_labels = []
        for i in range(4): # 4 rows
            for j in range(2): # 2 columns
                label = SquareLabel(f"Small Image {i*2 + j + 1}")
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("background-color: lightblue; border: 1px solid black;")
                right_layout.addWidget(label, i, j)
                self.small_image_labels.append(label)
        content_layout.addWidget(right_widget, 1)

        # Footer (Legend)
        footer_widget = QWidget()
        footer_layout = QHBoxLayout(footer_widget)
        main_layout.addWidget(footer_widget)

        export_button = QPushButton("Export Merged Images")
        export_button.clicked.connect(self.export_images)
        footer_layout.addStretch()
        footer_layout.addWidget(export_button)
        footer_layout.addStretch()

        self.prev_button.clicked.connect(self.show_previous_scan)
        self.next_button.clicked.connect(self.show_next_scan)

        # --- Data Processing and Display --- #
        nums = [
            #368294, 368296, 368297, 368298, 368299, 368303, 368304, 368311,
            #368313, 368314, 368315, 368333, 
            368343, 368362, 368370, 368383,
            368412, 368442, 368454, 368464, 368472, 368490, 368499, 368513,
            368525, 368530, 368549, 368604, 368612, 368620, 368643, 368653,
            368662, 368671, 368683, 368695, 368701, 368709, 368715, 368722,
            368729, 368743, 368763, 368772, 368782, 368792, 368801, 368826,
            368836, 368851, 368857, 368876, 368890, 368950, 368961, 368984,
            369001, 369009, 369017, 369025, 369042, 369058, 369068, 369089,
            369098, 369116, 369127, 369139, 369155, 369447, 369449, 369453,
            369462, 369467,
            367582
        ]

        # Define all scans to be processed
        all_scan_ids_to_process = nums
        self.available_scans = []
        self.failed_scans = []
        self.current_scan_index = 0

        # Disable buttons during data processing
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

        # 1. Process all scans for data gathering first
        for scan_id in all_scan_ids_to_process:
            if self.gather_data_for_scan(scan_id):
                self.available_scans.append(scan_id)
            else:
                self.failed_scans.append(scan_id)

        # 2. Display the first available scan initially
        if self.available_scans:
            self.display_scan(self.available_scans[self.current_scan_index])
            # Enable navigation buttons only if there's more than one scan to show
            if len(self.available_scans) > 1:
                self.prev_button.setEnabled(True)
                self.next_button.setEnabled(True)
        else:
            self.large_image_label.setText("No valid scan data found to display.")
            self.scan_id_label.setText("Scan ID: None")

        # 3. Report any failures
        if self.failed_scans:
            print("\n--- Data Gathering Summary ---")
            print("Could not find source data for the following scan IDs:")
            for scan_id in self.failed_scans:
                print(f"- {scan_id}")

    def display_scan(self, scan_id):
        self.scan_id_label.setText(f"Scan ID: {scan_id}")
        self.merged_images = {} # Clear images for export

        # Clear all image labels
        self.large_image_label.setPixmap(QPixmap()) # Set a blank pixmap
        self.large_image_label.setText(f"Loading Scan {scan_id}...")
        for label in self.small_image_labels:
            label.setPixmap(QPixmap())
            label.setText("")

        # Get scan directory
        scan_dir = os.path.join(PROCESSED_SCANS_DIR, str(scan_id))
        if not os.path.isdir(scan_dir):
            self.large_image_label.setText(f"Error: Data for scan {scan_id} not found.")
            return

        # Load and display the images for this scan
        num_boxes = self.load_and_display_merged_image(scan_dir, scan_id)
        if num_boxes > 0:
            # Determine the secondary scan IDs that should exist
            secondary_scan_ids = [scan_id + i for i in range(1, num_boxes + 1)]
            for i, sec_scan_id in enumerate(secondary_scan_ids):
                if i < len(self.small_image_labels):
                    self.load_and_display_secondary_image(scan_dir, sec_scan_id, self.small_image_labels[i])

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





    def load_and_display_merged_image(self, scan_dir, scan_id):
        ELEMENTS = ["Ca", "Fe", "Si"]
        image_paths = {
            element: os.path.join(scan_dir, f"scan_{scan_id}_{element}.tiff")
            for element in ELEMENTS
        }

        try:
            # Load images and convert to numpy arrays
            ca_img = np.array(Image.open(image_paths["Ca"]))
            fe_img = np.array(Image.open(image_paths["Fe"]))
            si_img = np.array(Image.open(image_paths["Si"]))

            # Normalize each channel to 0-255
            def normalize(arr):
                arr = arr.astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
                return arr.astype(np.uint8)

            r_channel = normalize(ca_img)
            g_channel = normalize(fe_img)
            b_channel = normalize(si_img)

            # Stack channels to create an RGB image
            rgb_image_np = np.stack([r_channel, g_channel, b_channel], axis=-1)

            # Convert numpy array to PIL Image to draw on it
            pil_img = Image.fromarray(rgb_image_np, 'RGB')

            # Load JSON data and draw boxes
            json_path = os.path.join(scan_dir, "unions_output_FeCaSi.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    union_data = json.load(f)

                draw = ImageDraw.Draw(pil_img)

                for item in union_data.values(): # Iterate over dictionary values
                    if 'image_center' in item and 'image_length' in item:
                        x, y = item['image_center']
                        length = item['image_length']
                        half_length = length / 2
                        # Define the bounding box for the rectangle
                        box = [x - half_length, y - half_length, x + half_length, y + half_length]
                        draw.rectangle(box, outline="white", width=1)

            # Convert PIL image to QPixmap using an in-memory buffer
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue(), 'PNG')

            self.large_image_label.setPixmap(pixmap)
            self.merged_images[scan_id] = pil_img # Store for export
            return len(union_data)

        except FileNotFoundError as e:
            print(f"Error loading image: {e}")
            self.large_image_label.setText(f"Error: {e.filename} not found.")
            return 0
        except Exception as e:
            print(f"An error occurred: {e}")
            self.large_image_label.setText("Error loading image.")
            return 0

    def load_and_display_secondary_image(self, scan_dir, scan_id, target_label):
        ELEMENTS = ["Ca", "Fe", "Si"]
        image_paths = {
            element: os.path.join(scan_dir, f"detsum_{element}_K_norm_{scan_id}.tiff")
            for element in ELEMENTS
        }

        try:
            # Load images and convert to numpy arrays
            ca_img = np.array(Image.open(image_paths["Ca"]))
            fe_img = np.array(Image.open(image_paths["Fe"]))
            si_img = np.array(Image.open(image_paths["Si"]))

            # Normalize each channel to 0-255
            def normalize(arr):
                arr = arr.astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
                return arr.astype(np.uint8)

            r_channel = normalize(ca_img)
            g_channel = normalize(fe_img)
            b_channel = normalize(si_img)

            # Stack channels to create an RGB image
            rgb_image_np = np.stack([r_channel, g_channel, b_channel], axis=-1)
            pil_img = Image.fromarray(rgb_image_np, 'RGB')

            # Convert PIL image to QPixmap using an in-memory buffer
            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            pixmap = QPixmap()
            pixmap.loadFromData(buffer.getvalue(), 'PNG')

            target_label.setPixmap(pixmap)
            self.merged_images[scan_id] = pil_img # Store for export

        except FileNotFoundError as e:
            print(f"Error loading secondary image: {e}")
            target_label.setText(f"Error: File not found.")
        except Exception as e:
            print(f"An error occurred while loading secondary image: {e}")
            target_label.setText("Error loading image.")



    def copy_scan_data(self, scan_id):
        ELEMENTS = ["Ca", "Fe", "Si"]

        src_parent_dir = os.path.join(RAW_PRIMARY_SCAN_DIR, f"automap_{scan_id}")
        if not os.path.isdir(src_parent_dir):
            print(f"Source directory not found: {src_parent_dir}")
            return None

        scan_dest_dir = os.path.join(PROCESSED_SCANS_DIR, str(scan_id))
        os.makedirs(scan_dest_dir, exist_ok=True)
        print(f"Created directory: {scan_dest_dir}")

        for element in ELEMENTS:
            src_file_name = f"scan_{scan_id}_{element}.tiff"
            src_path = os.path.join(src_parent_dir, src_file_name)
            dest_path = os.path.join(scan_dest_dir, src_file_name)

            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")
            else:
                print(f"Source file not found: {src_path}")

        # Copy the JSON file
        json_src_file_name = "unions_output_FeCaSi.json"
        json_src_path = os.path.join(src_parent_dir, json_src_file_name)
        json_dest_path = os.path.join(scan_dest_dir, json_src_file_name)
        if os.path.exists(json_src_path):
            shutil.copy(json_src_path, json_dest_path)
            print(f"Copied {json_src_path} to {json_dest_path}")
        else:
            print(f"Source file not found: {json_src_path}")

        return scan_dest_dir

    def copy_secondary_scans(self, initial_scan_id, num_scans):
        DEST_DIR = os.path.join(PROCESSED_SCANS_DIR, str(initial_scan_id))
        ELEMENTS = ["Fe", "Ca", "Si"] # The elements to look for
        processed_scan_ids = []

        for i in range(1, num_scans + 1):
            scan_id = initial_scan_id + i
            src_dir = os.path.join(RAW_SECONDARY_SCAN_DIR, f"output_tiff_scan2D_{scan_id}")

            if not os.path.isdir(src_dir):
                print(f"Source directory not found: {src_dir}")
                continue

            # Check if all required files exist before adding to list
            all_files_exist = True
            for element in ELEMENTS:
                src_file_name = f"detsum_{element}_K_norm.tiff"
                src_path = os.path.join(src_dir, src_file_name)
                if not os.path.exists(src_path):
                    all_files_exist = False
                    print(f"Source file not found: {src_path}")
                    break
            
            if all_files_exist:
                processed_scan_ids.append(scan_id)
                for element in ELEMENTS:
                    src_file_name = f"detsum_{element}_K_norm.tiff"
                    src_path = os.path.join(src_dir, src_file_name)
                    dest_file_name = f"detsum_{element}_K_norm_{scan_id}.tiff"
                    dest_path = os.path.join(DEST_DIR, dest_file_name)
                    shutil.copy(src_path, dest_path)
                    print(f"Copied {src_path} to {dest_path}")
        
        return processed_scan_ids

    def gather_data_for_scan(self, scan_id):
        print(f"--- Gathering data for Scan ID: {scan_id} ---")
        scan_dir = self.copy_scan_data(scan_id)
        if not scan_dir:
            print(f"--- Failed to find primary data for Scan ID: {scan_id} ---")
            return False

        json_path = os.path.join(scan_dir, "unions_output_FeCaSi.json")
        num_boxes = 0
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    union_data = json.load(f)
                    num_boxes = len(union_data)
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error reading or parsing JSON for scan {scan_id}: {e}")
        
        if num_boxes > 0:
            self.copy_secondary_scans(scan_id, num_boxes)
        print(f"--- Finished gathering data for Scan ID: {scan_id} ---")
        return True


    def export_images(self):
        if not self.merged_images:
            print("No images to export.")
            return

        # Open a dialog to ask the user for a directory
        base_save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Images")

        if base_save_dir:
            # Create the custom subfolder
            folder_name = f"scan_{self.initial_scan_id}_Coarse_Fine_Scans"
            save_dir = os.path.join(base_save_dir, folder_name)
            os.makedirs(save_dir, exist_ok=True)

            for scan_id, pil_img in self.merged_images.items():
                # Differentiate between the primary scan and secondary scans for naming
                if scan_id == self.initial_scan_id:
                    file_name = f"merged_scan_{scan_id}.png"
                else:
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
