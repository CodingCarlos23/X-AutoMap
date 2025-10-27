import sys
import os
import numpy as np
from PIL import Image
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QLabel, QCheckBox, QGraphicsView, QGraphicsScene, 
                             QGraphicsPixmapItem, QFrame, QGraphicsRectItem)
from PyQt5.QtCore import Qt, QLineF
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor

def merge_rgb_images(r_path, g_path, b_path):
    """Merges three grayscale images into a single RGB image."""
    try:
        r_img = Image.open(r_path)
        g_img = Image.open(g_path)
        b_img = Image.open(b_path)

        r_array = np.array(r_img)
        g_array = np.array(g_img)
        b_array = np.array(b_img)
        
        if r_array.max() > 0:
            r_array = (r_array / r_array.max() * 255).astype(np.uint8)
        else:
            r_array = r_array.astype(np.uint8)
            
        if g_array.max() > 0:
            g_array = (g_array / g_array.max() * 255).astype(np.uint8)
        else:
            g_array = g_array.astype(np.uint8)

        if b_array.max() > 0:
            b_array = (b_array / b_array.max() * 255).astype(np.uint8)
        else:
            b_array = b_array.astype(np.uint8)

        rgb_image_array = np.stack([r_array, g_array, b_array], axis=-1)
        return Image.fromarray(rgb_image_array)
    except FileNotFoundError as e:
        print(f"Warning: Could not find image file: {e}")
        return Image.new('RGB', (256, 256), (0, 0, 0)) # Return a black placeholder

def create_stitched_grid(base_root, scan_ids, elements, scan_folders_template, info_path):
    """Creates an 8x8 grid of merged RGB images with column overlap and loads union box data."""
    merged_images = []
    all_box_data = [] # To store box data for all scans

    grid_size = 8 # Define grid_size here for use in loop
    overlap_pixels = 5 # Define overlap_pixels here for use in loop

    for index, scan_id in enumerate(scan_ids):
        scan_folder = scan_folders_template.format(sid=scan_id)
        
        r_path = os.path.join(base_root, scan_folder, elements["R"])
        g_path = os.path.join(base_root, scan_folder, elements["G"])
        b_path = os.path.join(base_root, scan_folder, elements["B"])
        
        merged_images.append(merge_rgb_images(r_path, g_path, b_path))

        # Load union_output.json
        current_scan_box_data = []
        try:
            json_dir = os.path.join(info_path, f"automap_{scan_id}")
            json_file_path = os.path.join(json_dir, "unions_output.json")
            
            with open(json_file_path, 'r') as f:
                boxes = json.load(f)
                # Iterate through key-value pairs if boxes is a dictionary, or directly if it's a list
                if isinstance(boxes, dict):
                    for key, box_value in boxes.items():
                        current_scan_box_data.append({
                            'scan_id': scan_id,
                            'original_box': box_value, # Store the dictionary value
                            'grid_row': index // grid_size,
                            'grid_col': index % grid_size
                        })
                elif isinstance(boxes, list):
                    for box_value in boxes:
                        current_scan_box_data.append({
                            'scan_id': scan_id,
                            'original_box': box_value, 
                            'grid_row': index // grid_size,
                            'grid_col': index % grid_size
                        })
                else:
                    print(f"Warning: Unsupported JSON structure for scan {scan_id}. Expected dict or list.")

        except FileNotFoundError:
            print(f"Warning: unions_output.json not found for scan {scan_id} at {json_file_path}")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON for scan {scan_id} at {json_file_path}")
        
        all_box_data.extend(current_scan_box_data)

    if not merged_images:
        # Return a default image and info if no images were loaded
        default_img_width, default_img_height = 256, 256
        total_width = grid_size * default_img_width - (grid_size - 1) * overlap_pixels
        total_height = grid_size * default_img_height
        grid_image = Image.new('RGB', (total_width, total_height), (0,0,0))
        img_info = {
            'img_width': default_img_width,
            'img_height': default_img_height,
            'grid_size': grid_size,
            'overlap_pixels': overlap_pixels,
            'all_box_data': []
        }
        return grid_image, img_info
        
    img_width, img_height = merged_images[0].size

    # Calculate the new total width of the grid image
    total_width = grid_size * img_width - (grid_size - 1) * overlap_pixels
    total_height = grid_size * img_height # No vertical overlap requested

    grid_image = Image.new('RGB', (total_width, total_height))

    for index, img in enumerate(merged_images):
        row = index // grid_size
        col = index % grid_size
        
        # Adjust x-coordinate for overlap
        x_offset = col * (img_width - overlap_pixels)
        y_offset = row * img_height
        
        grid_image.paste(img, (x_offset, y_offset))
        
    img_info = {
        'img_width': img_width,
        'img_height': img_height,
        'grid_size': grid_size,
        'overlap_pixels': overlap_pixels,
        'all_box_data': all_box_data
    }
    return grid_image, img_info

class ZoomableView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self._pixmap_item)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

        self.borders = []
        self.union_boxes = []
        self.img_info = None # To store img_width, img_height, grid_size, overlap_pixels

    def set_pixmap(self, pixmap, img_info=None):
        self._pixmap_item.setPixmap(pixmap)
        self.img_info = img_info
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.create_borders()
        self.create_union_boxes()

    def create_borders(self):
        # Clear existing borders
        for border in self.borders:
            self.scene.removeItem(border)
        self.borders.clear()

        if not self.img_info:
            return

        img_width = self.img_info['img_width']
        img_height = self.img_info['img_height']
        grid_size = self.img_info['grid_size']
        overlap_pixels = self.img_info['overlap_pixels']

        pen = QPen(QColor(Qt.white))
        pen.setStyle(Qt.DotLine)
        pen.setWidth(1) 

        # Draw internal borders (top and left for each cell)
        for row in range(grid_size):
            for col in range(grid_size):
                x_offset = col * (img_width - overlap_pixels)
                y_offset = row * img_height

                # Top border
                self.borders.append(self.scene.addLine(QLineF(x_offset, y_offset, x_offset + img_width, y_offset), pen))
                # Left border
                self.borders.append(self.scene.addLine(QLineF(x_offset, y_offset, x_offset, y_offset + img_height), pen))

        # Draw rightmost vertical borders
        # The x-coordinate for the rightmost edge of the last image in the last column
        # This is (x_offset of last column) + img_width
        final_x_right_edge = (grid_size - 1) * (img_width - overlap_pixels) + img_width
        for row in range(grid_size):
            y_offset = row * img_height
            self.borders.append(self.scene.addLine(QLineF(final_x_right_edge, y_offset, final_x_right_edge, y_offset + img_height), pen))

        # Draw bottommost horizontal borders
        # The y-coordinate for the bottommost edge of the grid
        final_y = grid_size * img_height
        for col in range(grid_size):
            x_offset = col * (img_width - overlap_pixels)
            self.borders.append(self.scene.addLine(QLineF(x_offset, final_y, x_offset + img_width, final_y), pen))
        
        self.set_borders_visible(False) # Initially hidden

    def set_borders_visible(self, visible):
        for border in self.borders:
            border.setVisible(visible)

    def create_union_boxes(self):
        for box_item in self.union_boxes:
            self.scene.removeItem(box_item)
        self.union_boxes.clear()

        if not self.img_info or not self.img_info.get('all_box_data'):
            return

        img_width = self.img_info['img_width']
        img_height = self.img_info['img_height']
        grid_size = self.img_info['grid_size']
        overlap_pixels = self.img_info['overlap_pixels']

        pen = QPen(QColor(Qt.white))
        pen.setStyle(Qt.DotLine)
        pen.setWidth(1)

        for box_data in self.img_info['all_box_data']:
            scan_id = box_data['scan_id']
            original_box_data = box_data['original_box'] 
            grid_row = box_data['grid_row']
            grid_col = box_data['grid_col']

            # Calculate the top-left corner of the image within the stitched grid
            image_x_offset = grid_col * (img_width - overlap_pixels)
            image_y_offset = grid_row * img_height

            # Extract data from the new JSON structure
            # Ensure these keys exist in original_box_data
            center_x_json = original_box_data.get('image_center', [0, 0])[0]
            center_y_json = original_box_data.get('image_center', [0, 0])[1]
            length_json = original_box_data.get('image_length', 0)

            # Calculate top-left (x, y) and width/height for QGraphicsRectItem
            box_width = length_json
            box_height = length_json
            box_x = image_x_offset + (center_x_json - length_json / 2)
            box_y = image_y_offset + (center_y_json - length_json / 2)

            rect_item = QGraphicsRectItem(box_x, box_y, box_width, box_height)
            rect_item.setPen(pen)
            
            # Set tooltip using the provided image_center
            rect_item.setToolTip(f"Scan ID: {scan_id}\nImage Center: ({center_x_json:.2f}, {center_y_json:.2f})")
            
            self.scene.addItem(rect_item)
            self.union_boxes.append(rect_item)
        
        self.set_union_boxes_visible(False) # Initially hidden

    def set_union_boxes_visible(self, visible):
        for box_item in self.union_boxes:
            box_item.setVisible(visible)

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

class DataStitcherGUI(QMainWindow):
    def __init__(self, stitched_image=None, img_info=None):
        super().__init__()
        self.setWindowTitle("Data Stitcher")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        # Image area
        self.image_view = ZoomableView()
        self.layout.addWidget(self.image_view)

        if stitched_image:
            q_image = QImage(stitched_image.tobytes(), stitched_image.width, stitched_image.height, stitched_image.width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_view.set_pixmap(pixmap, img_info)

        # Legend area
        self.legend_area = QVBoxLayout()
        self.layout.addLayout(self.legend_area)

        self.legend_label = QLabel("Legend")
        self.legend_area.addWidget(self.legend_label)

        self.feature1_checkbox = QCheckBox("Feature 1")
        self.legend_area.addWidget(self.feature1_checkbox)
        self.feature2_checkbox = QCheckBox("Feature 2")
        self.legend_area.addWidget(self.feature2_checkbox)
        self.feature3_checkbox = QCheckBox("Feature 3")
        self.legend_area.addWidget(self.feature3_checkbox)

        self.show_borders_checkbox = QCheckBox("Show Borders")
        self.legend_area.addWidget(self.show_borders_checkbox)
        self.show_borders_checkbox.toggled.connect(self.image_view.set_borders_visible)

        self.show_union_boxes_checkbox = QCheckBox("Show Union Boxes") # New checkbox
        self.legend_area.addWidget(self.show_union_boxes_checkbox)
        self.show_union_boxes_checkbox.toggled.connect(self.image_view.set_union_boxes_visible) # Connect signal

        self.legend_area.addStretch()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # === CONFIGURATION ===
    # This section uses the configuration you provided.
    # If the files are not found at these paths, black placeholder images will be used.
    base_root = "/home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3/all_xrf"
    scan_ids = [
        367582, 367589, 367592, 367596, 367600, 367609, 367614, 367622,
        367630, 367634, 367638, 367641, 367646, 367653, 367658, 367663,
        367667, 367671, 367675, 367680, 367686, 367692, 367698, 367703,
        367710, 367715, 367720, 367726, 367733, 367741, 367744, 367748,
        367754, 367760, 367767, 367772, 367780, 367786, 367789, 367795,
        367798, 367803, 367807, 367813, 367816, 367819, 367825, 367830,
        367837, 367846, 367851, 367857, 367862, 367870, 367873, 367880,
        367885, 367890, 367897, 367899, 367903, 367910, 367915, 367921
    ]
    elements = {
        "G": "detsum_Ca_K_norm.tiff",
        "B": "detsum_Fe_K_norm.tiff",
        "R": "detsum_Cu_K_norm.tiff"
    }
    scan_folders_template = "output_tiff_scan2D_{sid}"
    
    info_path = "/home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3/data/user_macros" # Get info_path from user's code

    stitched_image, img_info = create_stitched_grid(base_root, scan_ids, elements, scan_folders_template, info_path) # Pass info_path

    main_win = DataStitcherGUI(stitched_image, img_info)
    main_win.show()
    sys.exit(app.exec_())

    this is good just add the hover data information then add the clickable box that itll open diplay of its matching fine scan 