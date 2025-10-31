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

class SquareLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event):
        super().paintEvent(event)

        pix = self.pixmap()
        if pix is None:
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

        # Start the data processing workflow
        scan_dir = self.copy_scan_data([self.initial_scan_id])
        if scan_dir:
            num_boxes = self.load_and_display_merged_image(scan_dir, self.initial_scan_id)
            if num_boxes > 0:
                processed_scan_ids = self.copy_secondary_scans(self.initial_scan_id, num_boxes)
                for i, scan_id in enumerate(processed_scan_ids):
                    if i < len(self.small_image_labels):
                        self.load_and_display_secondary_image(scan_dir, scan_id, self.small_image_labels[i])

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



    def copy_scan_data(self, scan_ids):
        BASE_DATA_DIR = "/home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3/automap_368304"
        DEST_PARENT_DIR = "/home/codingcarlos/Documents/github/SULI-2025-Summer/data/scans_grouped"
        ELEMENTS = ["Ca", "Fe", "Si"]

        # For this example, we only use the first scan_id
        scan_id = scan_ids[0]
        scan_dest_dir = os.path.join(DEST_PARENT_DIR, str(scan_id))
        os.makedirs(scan_dest_dir, exist_ok=True)
        print(f"Created directory: {scan_dest_dir}")

        for element in ELEMENTS:
            src_file_name = f"scan_{scan_id}_{element}.tiff"
            src_path = os.path.join(BASE_DATA_DIR, src_file_name)
            dest_path = os.path.join(scan_dest_dir, src_file_name)

            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                print(f"Copied {src_path} to {dest_path}")
            else:
                print(f"Source file not found: {src_path}")

        # Copy the JSON file
        json_src_file_name = "unions_output_FeCaSi.json"
        json_src_path = os.path.join(BASE_DATA_DIR, json_src_file_name)
        json_dest_path = os.path.join(scan_dest_dir, json_src_file_name)
        if os.path.exists(json_src_path):
            shutil.copy(json_src_path, json_dest_path)
            print(f"Copied {json_src_path} to {json_dest_path}")
        else:
            print(f"Source file not found: {json_src_path}")

        return scan_dest_dir

    def copy_secondary_scans(self, initial_scan_id, num_scans):
        BASE_DATA_DIR = "/home/codingcarlos/Desktop/Beamline_Data/Automap_2025Q3/all_xrf"
        DEST_DIR = os.path.join("/home/codingcarlos/Documents/github/SULI-2025-Summer/data/scans_grouped", str(initial_scan_id))
        ELEMENTS = ["Fe", "Ca", "Si"] # The elements to look for
        processed_scan_ids = []

        for i in range(1, num_scans + 1):
            scan_id = initial_scan_id + i
            src_dir = os.path.join(BASE_DATA_DIR, f"output_tiff_scan2D_{scan_id}")

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
