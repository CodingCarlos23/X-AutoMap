import sys
import os
import shutil
import json
import io
import numpy as np
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QSizePolicy
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage

class ScansGroupedViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scans Grouped Viewer")
        self.setGeometry(100, 100, 1200, 600)  # Initial window size

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left side: Large image placeholder
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.large_image_label = QLabel("Large Image Placeholder")
        self.large_image_label.setAlignment(Qt.AlignCenter)
        self.large_image_label.setStyleSheet("background-color: lightgray; border: 1px solid black;")
        self.large_image_label.setMinimumSize(400, 400) # Placeholder size
        self.large_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.large_image_label)
        main_layout.addWidget(left_widget, 1) # Take up 1 part of the horizontal space

        # Right side: Four small images placeholder
        right_widget = QWidget()
        right_layout = QGridLayout(right_widget)
        self.small_image_labels = []
        for i in range(2):
            for j in range(2):
                label = QLabel(f"Small Image {i*2 + j + 1}")
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("background-color: lightblue; border: 1px solid black;")
                label.setMinimumSize(200, 200) # Placeholder size
                label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                right_layout.addWidget(label, i, j)
                self.small_image_labels.append(label)
        main_layout.addWidget(right_widget, 1) # Take up 1 part of the horizontal space

        # Ensure square aspect ratio for image labels
        self.large_image_label.setScaledContents(True)
        for label in self.small_image_labels:
            label.setScaledContents(True)

        # Example usage of the data copying function
        scan_dir = self.copy_scan_data([368304])
        if scan_dir:
            num_boxes = self.load_and_display_merged_image(scan_dir, 368304)
            if num_boxes > 0:
                processed_scan_ids = self.copy_secondary_scans(368304, num_boxes)
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




if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ScansGroupedViewer()
    viewer.show()
    sys.exit(app.exec_())
