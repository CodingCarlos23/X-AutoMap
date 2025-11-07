import sys
import os
import numpy as np
from PIL import Image
import json
from PyQt5.QtWidgets import (
                             QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QLabel, QCheckBox, QGraphicsView, QGraphicsScene, 
                             QGraphicsPixmapItem, QFrame, QGraphicsRectItem, QGraphicsTextItem, QPushButton, QFileDialog)
from PyQt5.QtCore import Qt, QLineF, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QFont, QPainter
import subprocess
import shutil

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

def create_stitched_grid(base_root, scan_id_pairs, elements, scan_folders_template, info_path, fine_path):
    """Creates a 3x3 grid of merged RGB images with column overlap and loads union box data."""
    merged_images = []
    all_box_data = [] # To store box data for all scans

    grid_size = 3 # Define grid_size here for use in loop
    overlap_pixels = 5 # Define overlap_pixels here for use in loop

    coarse_scan_ids = [pair[0] for pair in scan_id_pairs]

    for index, scan_id in enumerate(coarse_scan_ids):
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
                if isinstance(boxes, dict):
                    for key, box_value in boxes.items():
                        current_scan_box_data.append({
                            'scan_id': scan_id,
                            'original_box': box_value,
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
        default_img_width, default_img_height = 256, 256
        total_width = grid_size * default_img_width - (grid_size - 1) * overlap_pixels
        total_height = grid_size * default_img_height
        grid_image = Image.new('RGB', (total_width, total_height), (0,0,0))
        img_info = {
            'img_width': default_img_width,
            'img_height': default_img_height,
            'grid_size': grid_size,
            'overlap_pixels': overlap_pixels,
            'all_box_data': [],
            'scan_ids': [],
            'fine_path': ""
        }
        return grid_image, img_info
        
    img_width, img_height = merged_images[0].size

    total_width = grid_size * img_width - (grid_size - 1) * overlap_pixels
    total_height = grid_size * img_height

    grid_image = Image.new('RGB', (total_width, total_height))

    for index, img in enumerate(merged_images):
        row = index // grid_size
        col = index % grid_size
        
        x_offset = col * (img_width - overlap_pixels)
        y_offset = row * img_height
        
        grid_image.paste(img, (x_offset, y_offset))
        
    img_info = {
        'img_width': img_width,
        'img_height': img_height,
        'grid_size': grid_size,
        'overlap_pixels': overlap_pixels,
        'all_box_data': all_box_data,
        'scan_ids': coarse_scan_ids,
        'scan_id_pairs': scan_id_pairs,
        'fine_path': fine_path,
        'info_path': info_path
    }
    return grid_image, img_info

class HoverableGraphicsRectItem(QGraphicsRectItem):
    def __init__(self, x, y, width, height, scan_id, real_center_x, real_center_y, hover_text_item):
        super().__init__(x, y, width, height) # Call QGraphicsRectItem's constructor
        self.setAcceptHoverEvents(True)
        self.scan_id = scan_id
        self.real_center_x = real_center_x
        self.real_center_y = real_center_y
        self.hover_text_item = hover_text_item
        self.original_pen = QPen(QColor(Qt.white)) # Default pen
        self.original_pen.setStyle(Qt.DotLine)
        self.original_pen.setWidth(1)
        self.setPen(self.original_pen)

    def hoverEnterEvent(self, event):
        self.hover_text_item.setPlainText(f"Scan ID: {self.scan_id}\nReal Center (μm): ({self.real_center_x:.2f}, {self.real_center_y:.2f})")
        self.hover_text_item.setPos(self.mapToScene(event.pos()) + QPointF(10, -30))
        self.hover_text_item.setVisible(True)
        
        highlight_pen = QPen(QColor(Qt.yellow))
        highlight_pen.setStyle(Qt.DotLine)
        highlight_pen.setWidth(2)
        self.setPen(highlight_pen)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.hover_text_item.setVisible(False)
        self.setPen(self.original_pen)
        super().hoverLeaveEvent(event)

class ZoomableView(QGraphicsView):
    mouseMoved = pyqtSignal(QPointF) # Signal to emit mouse coordinates

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.scene.setBackgroundBrush(QColor(Qt.black))
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self.scene.addItem(self._pixmap_item)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setMouseTracking(True)

        self.hover_text_item = QGraphicsTextItem()
        self.hover_text_item.setDefaultTextColor(Qt.white)
        font = QFont()
        font.setPointSize(6) # Set font size for hover text
        self.hover_text_item.setFont(font)
        self.hover_text_item.setZValue(100) # Ensure it's on top
        self.hover_text_item.setVisible(False)
        self.scene.addItem(self.hover_text_item)

        self.borders = []
        self.union_boxes = []
        self.drawn_boxes = [] # For user-drawn boxes
        self.img_info = None # To store img_width, img_height, grid_size, overlap_pixels
        self.fine_path_base = "" # Store the base path for fine scans
        self.info_path = "" # Store the base path for coarse scan info

        self.drawing_mode = False
        self.start_point = None
        self.current_rect = None

    def set_drawing_mode(self, enabled):
        self.drawing_mode = enabled
        if self.drawing_mode:
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def set_pixmap(self, pixmap, img_info=None):
        self._pixmap_item.setPixmap(pixmap)
        self.img_info = img_info
        self.fine_path_base = img_info.get('fine_path', "") # Get fine_path from img_info
        self.info_path = img_info.get('info_path', "") # Get info_path from img_info
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self._create_borders_internal()
        self.create_union_boxes()

    def _create_borders_internal(self):
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
        pen.setStyle(Qt.SolidLine)
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

            # Get accurate real center (μm) for hover text
            real_cx = original_box_data['real_center_um'][0]
            real_cy = original_box_data['real_center_um'][1]

            # --- Apply negative shift formula ---
            x_shift_tile = img_width / 2
            y_shift_tile = img_height / 2
            if center_x_json < 0 or center_y_json < 0:
                center_x_json += x_shift_tile
                center_y_json += y_shift_tile

            length_json = original_box_data.get('image_length', 0)

            # Calculate top-left (x, y) and width/height for QGraphicsRectItem
            box_width = length_json
            box_height = length_json
            box_x = image_x_offset + (center_x_json - length_json / 2)
            box_y = image_y_offset + (center_y_json - length_json / 2)

            rect_item = HoverableGraphicsRectItem(
                box_x, box_y, box_width, box_height, # Pass individual coordinates
                scan_id,
                real_cx,
                real_cy,
                self.hover_text_item # Pass the shared hover_text_item
            )
            rect_item.setPen(pen)
            # rect_item.clicked.connect(self.open_fine_scan_folder) # Connect the signal
            
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

    def mouseMoveEvent(self, event):
        if self.drawing_mode and self.start_point and (event.buttons() & Qt.LeftButton):
            end_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, end_point).normalized()
            if self.current_rect:
                self.current_rect.setRect(rect)
        else:
            # Always call super() first to ensure panning works
            super().mouseMoveEvent(event)

            scene_pos = self.mapToScene(event.pos())
        self.mouseMoved.emit(scene_pos) # Emit the signal with scene coordinates

        # Only handle tooltips if no buttons are pressed (i.e., hovering)
        if event.buttons() == Qt.NoButton:
            item = self.itemAt(event.pos())

            # If not hovering over a HoverableGraphicsRectItem, check for grid cell
            if not isinstance(item, HoverableGraphicsRectItem):
                if self.img_info and 'scan_ids' in self.img_info and self.img_info['scan_ids']:
                    img_width = self.img_info['img_width']
                    img_height = self.img_info['img_height']
                    grid_size = self.img_info['grid_size']
                    overlap_pixels = self.img_info['overlap_pixels']

                    scene_rect = self.scene.sceneRect()
                    if not scene_rect.contains(scene_pos):
                        self.hover_text_item.setVisible(False)
                    else:
                        effective_width = img_width - overlap_pixels
                        if effective_width <= 0:
                            self.hover_text_item.setVisible(False)
                        else:
                            col = int(scene_pos.x() / effective_width)
                            row = int(scene_pos.y() / img_height)

                            if 0 <= row < grid_size and 0 <= col < grid_size:
                                index = row * grid_size + col
                                if 0 <= index < len(self.img_info['scan_ids']):
                                    scan_id = self.img_info['scan_ids'][index]
                                    self.hover_text_item.setPlainText(f"Scan ID: {scan_id}\nMouse Coords: (X: {scene_pos.x():.2f}, Y: {scene_pos.y():.2f})")
                                    self.hover_text_item.setPos(self.mapToScene(event.pos()) + QPointF(10, -30))
                                    self.hover_text_item.setVisible(True)
                                else:
                                    self.hover_text_item.setVisible(False)
                            else:
                                self.hover_text_item.setVisible(False)
                else:
                    self.hover_text_item.setVisible(False)
            # If hovering over a HoverableGraphicsRectItem, its own hoverEnterEvent will handle visibility
            # So, we don't need to do anything here.
        else:
            # If a mouse button is pressed, hide the tooltip
            self.hover_text_item.setVisible(False)

    def mousePressEvent(self, event):
        if self.drawing_mode and event.button() == Qt.LeftButton:
            self.start_point = self.mapToScene(event.pos())
            pen = QPen(QColor(Qt.yellow))
            pen.setStyle(Qt.DotLine)
            pen.setWidth(2)
            self.current_rect = QGraphicsRectItem(QRectF(self.start_point, self.start_point))
            self.current_rect.setPen(pen)
            self.scene.addItem(self.current_rect)
            self.drawn_boxes.append(self.current_rect)
        else:
            if event.button() == Qt.LeftButton:
                item = self.itemAt(event.pos())
                if isinstance(item, HoverableGraphicsRectItem):
                    self.open_fine_scan_folder(item.scan_id)
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing_mode and self.start_point and event.button() == Qt.LeftButton:
            self.start_point = None
            self.current_rect = None
        else:
            super().mouseReleaseEvent(event)

    def open_fine_scan_folder(self, scan_id):
        if not self.fine_path_base:
            print("Fine path base not configured.")
            return
        if not self.info_path:
            print("Info path not configured.")
            return

        # Find the folder that starts with scan_id
        target_folder_prefix = str(scan_id) + "-"
        found_folder = None
        try:
            for folder_name in os.listdir(self.fine_path_base):
                if folder_name.startswith(target_folder_prefix) and os.path.isdir(os.path.join(self.fine_path_base, folder_name)):
                    found_folder = folder_name
                    break
        except FileNotFoundError:
            print(f"Error: Fine path base directory not found: {self.fine_path_base}")
            return

        if found_folder:
            # Copy the corresponding unions_output.json to the fine scan folder
            source_json_dir = os.path.join(self.info_path, f"automap_{scan_id}")
            source_json_path = os.path.join(source_json_dir, "unions_output.json")
            
            dest_folder_path = os.path.join(self.fine_path_base, found_folder)
            dest_json_path = os.path.join(dest_folder_path, "unions_output.json")

            if os.path.exists(source_json_path):
                try:
                    shutil.copy(source_json_path, dest_json_path)
                    print(f"Copied {source_json_path} to {dest_json_path}")
                except Exception as e:
                    print(f"Error copying JSON file: {e}")
            else:
                print(f"Warning: Source JSON not found, not copied: {source_json_path}")

            # Open the folder
            full_path = os.path.join(self.fine_path_base, found_folder)
            print(f"Opening folder: {full_path}")
            if sys.platform == "win32":
                os.startfile(full_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", full_path])
            else: # Linux
                subprocess.run(["xdg-open", full_path])
        else:
            print(f"No fine scan folder found for scan ID: {scan_id} in {self.fine_path_base}")

class DataStitcherGUI(QMainWindow):
    def __init__(self, stitched_image=None, img_info=None):
        super().__init__()
        self.setWindowTitle("Data Stitcher")
        self.setStyleSheet(
            """
            QMainWindow { background-color: black; }
            QLabel { color: white; }
            QCheckBox { color: white; }
            QPushButton { background-color: #444; color: white; border: 1px solid #666; padding: 5px; border-radius: 3px; }
            QPushButton:hover { background-color: #555; }
            QPushButton:disabled { background-color: #222; color: #888; border: 1px solid #444; }
            """)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget) # Changed to QVBoxLayout

        # Image area
        self.image_view = ZoomableView()
        self.layout.addWidget(self.image_view, 1) # Added stretch factor to image_view

        if stitched_image:
            q_image = QImage(stitched_image.tobytes(), stitched_image.width, stitched_image.height, stitched_image.width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_view.set_pixmap(pixmap, img_info)

        # Legend area
        self.legend_area = QHBoxLayout() # Changed to QHBoxLayout
        self.layout.addLayout(self.legend_area) # Legend is added to the main QVBoxLayout

        self.legend_label = QLabel("Legend")
        self.legend_area.addWidget(self.legend_label)

        self.mouse_coords_label = QLabel("Mouse Coords: (X: -, Y: -) | Real: (X: -, Y: -)")
        self.legend_area.addWidget(self.mouse_coords_label)

        # Real-world coordinate conversion parameters
        self.origin_x = 10.0
        self.origin_y = 20.0
        self.micron_per_pixel_x = 0.25
        self.micron_per_pixel_y = 0.25

        self.image_view.mouseMoved.connect(self.update_mouse_coords_label)



        self.show_borders_checkbox = QCheckBox("Show Borders")
        self.legend_area.addWidget(self.show_borders_checkbox)
        self.show_borders_checkbox.toggled.connect(self.image_view.set_borders_visible)

        self.show_union_boxes_checkbox = QCheckBox("Show Union Boxes") # New checkbox
        self.legend_area.addWidget(self.show_union_boxes_checkbox)
        self.show_union_boxes_checkbox.toggled.connect(self.image_view.set_union_boxes_visible) # Connect signal

        self.legend_area.addStretch()

        self.export_button = QPushButton("Export 3by3 as PNG")
        self.legend_area.addWidget(self.export_button)
        self.export_button.clicked.connect(self.export_image)

        self.draw_box_button = QPushButton("Draw Box")
        self.draw_box_button.setCheckable(True)
        self.legend_area.addWidget(self.draw_box_button)
        self.draw_box_button.toggled.connect(self.image_view.set_drawing_mode)

    def export_image(self):
        scene = self.image_view.scene
        if scene.itemsBoundingRect().isEmpty():
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "stitched_3by3.png", "PNG Images (*.png);;All Files (*)")

        if file_path:
            if not file_path.lower().endswith('.png'):
                file_path += '.png'
            
            # Create a QImage to render the scene to
            image = QImage(scene.sceneRect().size().toSize(), QImage.Format_ARGB32)
            image.fill(Qt.transparent)

            painter = QPainter(image)
            scene.render(painter)
            painter.end()

            image.save(file_path)

    def update_mouse_coords_label(self, pos):
        # Calculate real-world coordinates
        real_x = self.origin_x + (pos.x() * self.micron_per_pixel_x)
        real_y = self.origin_y + (pos.y() * self.micron_per_pixel_y)
        self.mouse_coords_label.setText(f"Mouse Coords: (X: {pos.x():.2f}, Y: {pos.y():.2f}) | Real: (X: {real_x:.2f}, Y: {real_y:.2f})")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # === CONFIGURATION for 3x3 grid ===
    base_root = "/home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3/all_xrf"
    scan_id_pairs = [
        (367798, 367803), (367803, 367807), (367807, 367813),
        (367837, 367846), (367846, 367851), (367851, 367857),
        (367885, 367890), (367890, 367897), (367897, 367899)
    ] # Placeholder IDs
    elements = {
        "G": "detsum_Ca_K_norm.tiff",
        "B": "detsum_Fe_K_norm.tiff",
        "R": "detsum_Cu_K_norm.tiff"
    }
    scan_folders_template = "output_tiff_scan2D_{sid}"
    
    info_path = "/home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3/data/user_macros"
    fine_path = "/home/codingcarlos/Desktop/Data/FineImages"
    
    stitched_image, img_info = create_stitched_grid(base_root, scan_id_pairs, elements, scan_folders_template, info_path, fine_path)

    main_win = DataStitcherGUI(stitched_image, img_info)
    main_win.show()
    sys.exit(app.exec_())
