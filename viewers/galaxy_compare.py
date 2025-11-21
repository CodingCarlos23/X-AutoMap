import os
import sys
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QWidget, QFileDialog
)
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt


class YOLOViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image_folder = "/home/codingcarlos/Desktop/DeepSpaceYoloDataset/images"
        self.label_folder = "/home/codingcarlos/Desktop/DeepSpaceYoloDataset/labels"
        self.file_list = []
        self.index = 0
        self.annotated_pixmap = None
        self.raw_pixmap = None
        self.raw_pixmap_with_detections = None
        self.cv_params = self.load_cv_params()

        self.setWindowTitle("YOLO Annotation Viewer")
        self.resize(900, 700)
        self.setFocusPolicy(Qt.StrongFocus)

        # --- UI ELEMENTS ---
        self.annotated_label = QLabel(alignment=Qt.AlignCenter)
        self.annotated_label.setScaledContents(False)
        self.annotated_label.setMinimumSize(400, 300)

        self.raw_label = QLabel(alignment=Qt.AlignCenter)
        self.raw_label.setScaledContents(False)
        self.raw_label.setMinimumSize(400, 300)

        self.btn_prev = QPushButton("⟵ Previous")
        self.btn_next = QPushButton("Next ⟶")
        self.btn_load_images = QPushButton("Load Images Folder")
        self.btn_load_labels = QPushButton("Load Labels Folder")

        # Connect buttons
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_load_images.clicked.connect(self.load_image_folder)
        self.btn_load_labels.clicked.connect(self.load_label_folder)
        self.btn_prev.setShortcut(Qt.Key_Left)
        self.btn_next.setShortcut(Qt.Key_Right)

        # Path display labels so the user always sees which folders/files are used
        self.image_path_label = QLabel()
        self.label_path_label = QLabel()
        self.current_file_label = QLabel("No image loaded")
        for lbl in (self.image_path_label, self.label_path_label, self.current_file_label):
            lbl.setWordWrap(True)

        # Layout: info on top; annotated view left, raw view right
        h_nav = QHBoxLayout()
        h_nav.addWidget(self.btn_prev)
        h_nav.addWidget(self.btn_next)

        h_load = QHBoxLayout()
        h_load.addWidget(self.btn_load_images)
        h_load.addWidget(self.btn_load_labels)

        info_layout = QVBoxLayout()
        info_layout.addLayout(h_load)
        info_layout.addWidget(self.image_path_label)
        info_layout.addWidget(self.label_path_label)
        info_layout.addWidget(self.current_file_label)

        views_layout = QHBoxLayout()
        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("Their annotation"))
        left_col.addWidget(self.annotated_label, stretch=1)

        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Our annotation"))
        right_col.addWidget(self.raw_label, stretch=1)

        views_layout.addLayout(left_col, stretch=1)
        views_layout.addLayout(right_col, stretch=1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(info_layout)
        main_layout.addLayout(views_layout, stretch=1)
        main_layout.addLayout(h_nav)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.update_path_labels()
        self.try_build_file_list()

    def load_cv_params(self):
        defaults = {
            "min_threshold_intensity": 100,
            "min_threshold_area": 200,
            "max_threshold_area": 1600,
            "dilation_size": 5,
            "dilation_iterations": 3,
        }
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            json_path = os.path.join(base_dir, "other", "initial_scan.json")
            with open(json_path, "r") as f:
                params = json.load(f)
            defaults.update({
                "min_threshold_intensity": params.get("min_threshold_intensity", defaults["min_threshold_intensity"]),
                "min_threshold_area": params.get("min_threshold_area", defaults["min_threshold_area"]),
                "max_threshold_area": params.get("max_threshold_area", defaults["max_threshold_area"]),
                "dilation_size": int(params.get("dialaiton_size", defaults["dilation_size"])),
                "dilation_iterations": int(params.get("dialation_iteration", defaults["dilation_iterations"])),
            })
        except Exception:
            pass
        return defaults

    # -----------------------------
    # LOAD FOLDERS
    # -----------------------------
    def load_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.image_folder = folder
            self.try_build_file_list()
            self.update_path_labels()

    def load_label_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Label Folder")
        if folder:
            self.label_folder = folder
            self.try_build_file_list()
            self.update_path_labels()

    # Build synchronized file list once both folders chosen
    def try_build_file_list(self):
        if not self.image_folder or not self.label_folder:
            return

        images = [f for f in os.listdir(self.image_folder)
                  if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]

        # Keep only images with matching label file
        self.file_list = [
            img for img in images
            if os.path.isfile(os.path.join(
                self.label_folder,
                os.path.splitext(img)[0] + ".txt"))
        ]

        self.file_list.sort()
        self.index = 0

        if self.file_list:
            self.load_image()
        else:
            self.update_path_labels()

    # -----------------------------
    # NAVIGATION
    # -----------------------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key_Left:
            self.prev_image()

    def prev_image(self):
        if self.file_list:
            self.index = (self.index - 1) % len(self.file_list)
            self.load_image()

    def next_image(self):
        if self.file_list:
            self.index = (self.index + 1) % len(self.file_list)
            self.load_image()

    # -----------------------------
    # LOADING + DRAWING BOXES
    # -----------------------------
    def load_image(self):
        if not self.file_list:
            self.update_path_labels()
            return

        filename = self.file_list[self.index]
        img_path = os.path.join(self.image_folder, filename)
        label_path = os.path.join(self.label_folder,
                                  os.path.splitext(filename)[0] + ".txt")

        if not os.path.isfile(img_path):
            self.current_file_label.setText(f"Image file not found: {img_path}")
            self.annotated_label.clear()
            self.raw_label.clear()
            self.annotated_pixmap = None
            self.raw_pixmap = None
            self.raw_pixmap_with_detections = None
            self.update_path_labels(img_path, label_path)
            return

        if not os.path.isfile(label_path):
            self.current_file_label.setText(f"Label file not found: {label_path}")
            self.annotated_label.clear()
            self.raw_label.clear()
            self.annotated_pixmap = None
            self.raw_pixmap = None
            self.raw_pixmap_with_detections = None
            self.update_path_labels(img_path, label_path)
            return

        pixmap_raw = QPixmap(img_path)
        if pixmap_raw.isNull():
            self.current_file_label.setText(f"Could not load image: {img_path}")
            self.annotated_label.clear()
            self.raw_label.clear()
            self.annotated_pixmap = None
            self.raw_pixmap = None
            self.raw_pixmap_with_detections = None
            self.update_path_labels(img_path, label_path)
            return

        pixmap_annotated = pixmap_raw.copy()

        painter = QPainter(pixmap_annotated)
        pen = QPen(Qt.red, 3)
        painter.setPen(pen)

        img_w = pixmap_annotated.width()
        img_h = pixmap_annotated.height()

        # Read YOLO boxes
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls, x_c, y_c, w, h = map(float, parts)

                # Denormalize box
                box_w = w * img_w
                box_h = h * img_h
                box_x = (x_c * img_w) - box_w / 2
                box_y = (y_c * img_h) - box_h / 2

                painter.drawRect(int(box_x), int(box_y),
                                 int(box_w), int(box_h))

        painter.end()

        self.annotated_pixmap = pixmap_annotated
        self.raw_pixmap = pixmap_raw
        self.run_cv_detection(img_path)
        self.refresh_all_pixmaps()
        self.update_path_labels(img_path, label_path)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_all_pixmaps()

    def refresh_all_pixmaps(self):
        self.refresh_display_pixmap(self.annotated_label, self.annotated_pixmap)
        raw_source = self.raw_pixmap_with_detections or self.raw_pixmap
        self.refresh_display_pixmap(self.raw_label, raw_source)

    def refresh_display_pixmap(self, label, pixmap):
        if pixmap is None:
            label.clear()
            return
        target_size = label.size()
        scaled = pixmap.scaled(
            target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(scaled)

    def update_path_labels(self, image_path=None, label_path=None):
        self.image_path_label.setText(f"Images folder: {self.image_folder}")
        self.label_path_label.setText(f"Labels folder: {self.label_folder}")
        if image_path and label_path:
            self.current_file_label.setText(
                f"Image file: {image_path}\nLabel file: {label_path}"
            )
        else:
            self.current_file_label.setText("No image loaded")

    # -----------------------------
    # CV DETECTION (right image)
    # -----------------------------
    def normalize_and_dilate(self, img):
        norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        kernel = np.ones((self.cv_params["dilation_size"], self.cv_params["dilation_size"]), np.uint8)
        dilated = cv2.dilate(norm, kernel, iterations=self.cv_params["dilation_iterations"])
        return norm, dilated

    def detect_blobs_cv(self, img_norm):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = self.cv_params["min_threshold_intensity"]
        params.maxThreshold = 255
        params.filterByArea = True
        params.minArea = self.cv_params["min_threshold_area"]
        params.maxArea = self.cv_params["max_threshold_area"]
        params.thresholdStep = 2
        params.filterByColor = False
        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByConvexity = False
        params.minRepeatability = 1

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img_norm)
        boxes = []
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            radius = int(kp.size / 2)
            size = 2 * radius
            boxes.append((x - radius, y - radius, size, size))
        return boxes

    def run_cv_detection(self, img_path):
        self.raw_pixmap_with_detections = None
        if self.raw_pixmap is None:
            return
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            return
        img_float = img_gray.astype(np.float32)
        _, img_proc = self.normalize_and_dilate(img_float)
        boxes = self.detect_blobs_cv(img_proc)
        if not boxes:
            return
        overlay = self.raw_pixmap.copy()
        painter = QPainter(overlay)
        painter.setPen(QPen(Qt.blue, 2))
        for x, y, w, h in boxes:
            painter.drawRect(int(x), int(y), int(w), int(h))
        painter.end()
        self.raw_pixmap_with_detections = overlay


# -----------------------------
# RUN APPLICATION
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = YOLOViewer()
    viewer.show()
    sys.exit(app.exec_())
