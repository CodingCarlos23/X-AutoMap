import os
import re
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def normalize_channel(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    cmax, cmin = arr.max(), arr.min()
    if cmax > cmin:
        arr = (arr - cmin) / (cmax - cmin) * 255.0
    return arr.astype(np.uint8)


def qpixmap_from_array(arr: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(arr)
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


DEFAULT_SCAN_DIR = "/home/codingcarlos/Documents/github/SULI-2025-Summer/data/scans_grouped/368604_FeCaSi"


class SingleScanView(QMainWindow):
    def __init__(self, initial_folder: str = None):
        super().__init__()
        self.setWindowTitle("Single Scan View (Fe/Ca/Si)")
        self.setGeometry(100, 100, 1000, 900)

        self.scan_dir = None
        self.available_ids: List[int] = []
        self.element_names: List[str] = []

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        controls = QHBoxLayout()
        self.load_btn = QPushButton("Select Folder")
        self.load_btn.clicked.connect(self.select_folder)
        controls.addWidget(self.load_btn)
        controls.addWidget(QLabel("Fine ID:"))
        self.id_selector = QComboBox()
        self.id_selector.currentIndexChanged.connect(self.change_id)
        controls.addWidget(self.id_selector)
        controls.addStretch()
        root.addLayout(controls)

        grid_box = QGroupBox("Channels")
        grid_layout = QGridLayout(grid_box)
        grid_layout.setSpacing(8)
        self.labels: Dict[str, QLabel] = {}
        self.label_keys = ["channel_r", "channel_g", "channel_b", "merged"]
        for idx, key in enumerate(self.label_keys):
            box = QVBoxLayout()
            title = QLabel("")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("color:white;background-color:#333;padding:4px;border:1px solid #555;")
            lbl = QLabel("No Image")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background-color:#222;border:1px solid #444;color:white;")
            lbl.setScaledContents(True)
            lbl.setMinimumSize(QSize(200, 200))
            box.addWidget(title)
            box.addWidget(lbl)
            r, c = divmod(idx, 2)
            container = QWidget()
            container.setLayout(box)
            grid_layout.addWidget(container, r, c)
            self.labels[self.label_keys[idx]] = lbl
            self.labels[f"{self.label_keys[idx]}_title"] = title
        root.addWidget(grid_box, 1)

        self.status_lbl = QLabel("Select a folder containing Fe/Ca/Si detsum TIFFs.")
        root.addWidget(self.status_lbl)

        folder_to_load = initial_folder
        if not folder_to_load and os.path.isdir(DEFAULT_SCAN_DIR):
            folder_to_load = DEFAULT_SCAN_DIR
        if folder_to_load:
            self.load_folder(folder_to_load)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Scan Folder")
        if not folder:
            return
        self.load_folder(folder)

    def detect_elements(self, folder: str) -> List[str]:
        pattern = re.compile(r"detsum_(\d+)_([^.]+)\.tiff$")
        element_counts: Dict[int, set] = {}
        for name in os.listdir(folder):
            m = pattern.match(name)
            if not m:
                continue
            fid = int(m.group(1))
            elem = m.group(2)
            element_counts.setdefault(fid, set()).add(elem)
        # Pick the id with the most elements as the reference
        best_id = None
        best_set = set()
        for fid, elems in element_counts.items():
            if len(elems) > len(best_set):
                best_set = elems
                best_id = fid
        return sorted(list(best_set))[:3]

    def update_label_titles(self):
        names = self.element_names
        colors = ["Red", "Green", "Blue"]
        for idx, key in enumerate(self.label_keys):
            title_lbl = self.labels.get(f"{key}_title")
            if not title_lbl:
                continue
            if key == "merged":
                title_lbl.setText("Merged RGB")
            else:
                if idx < len(names):
                    title_lbl.setText(f"detsum_<id>_{names[idx]} ({colors[idx]})")
                else:
                    title_lbl.setText(f"Channel {colors[idx]}")

    def load_folder(self, folder: str):
        self.scan_dir = folder
        self.status_lbl.setText(f"Folder: {folder}")
        self.element_names = self.detect_elements(folder)
        if len(self.element_names) < 3:
            self.clear_images("Could not find three channel elements in this folder.")
            self.available_ids = []
            self.id_selector.clear()
            return
        self.update_label_titles()
        self.available_ids = self.find_fine_ids(folder)
        self.id_selector.blockSignals(True)
        self.id_selector.clear()
        for fid in self.available_ids:
            self.id_selector.addItem(str(fid), fid)
        self.id_selector.blockSignals(False)
        if self.available_ids:
            # Prefer fine ID 368606 if present, otherwise first available
            default_idx = 0
            if 368606 in self.available_ids:
                default_idx = self.available_ids.index(368606)
            self.id_selector.setCurrentIndex(default_idx)
            self.load_images(self.available_ids[default_idx])
        else:
            self.clear_images("No matching detsum_<id>_<element>.tiff files found.")

    def change_id(self, idx: int):
        if idx < 0 or idx >= len(self.available_ids):
            return
        fid = self.id_selector.currentData()
        if fid:
            self.load_images(int(fid))

    def find_fine_ids(self, folder: str) -> List[int]:
        ids = set()
        for element in self.element_names:
            for name in os.listdir(folder):
                m = re.match(r"detsum_(\d+)_%s\.tiff$" % re.escape(element), name)
                if m:
                    ids.add(int(m.group(1)))
        # Keep only IDs that have all three elements
        valid = []
        for fid in ids:
            missing = False
            for element in self.element_names:
                path = os.path.join(folder, f"detsum_{fid}_{element}.tiff")
                if not os.path.exists(path):
                    missing = True
                    break
            if not missing:
                valid.append(fid)
        return sorted(valid)

    def load_images(self, fine_id: int):
        try:
            paths = [
                os.path.join(self.scan_dir, f"detsum_{fine_id}_{self.element_names[0]}.tiff"),
                os.path.join(self.scan_dir, f"detsum_{fine_id}_{self.element_names[1]}.tiff"),
                os.path.join(self.scan_dir, f"detsum_{fine_id}_{self.element_names[2]}.tiff"),
            ]
            r_img = np.array(Image.open(paths[0]))
            g_img = np.array(Image.open(paths[1]))
            b_img = np.array(Image.open(paths[2]))
        except FileNotFoundError:
            self.clear_images(f"Missing channel files for fine ID {fine_id}")
            return

        r_norm = normalize_channel(r_img)
        g_norm = normalize_channel(g_img)
        b_norm = normalize_channel(b_img)

        merged = np.stack([r_norm, g_norm, b_norm], axis=-1)

        ca_rgb = np.zeros_like(merged)
        ca_rgb[:, :, 0] = r_norm
        fe_rgb = np.zeros_like(merged)
        fe_rgb[:, :, 1] = g_norm
        si_rgb = np.zeros_like(merged)
        si_rgb[:, :, 2] = b_norm

        self.labels["channel_r"].setPixmap(qpixmap_from_array(ca_rgb))
        self.labels["channel_g"].setPixmap(qpixmap_from_array(fe_rgb))
        self.labels["channel_b"].setPixmap(qpixmap_from_array(si_rgb))
        self.labels["merged"].setPixmap(qpixmap_from_array(merged))
        # Update title labels with actual file names and colors
        color_names = ["Red", "Green", "Blue"]
        for idx, key in enumerate(["channel_r", "channel_g", "channel_b"]):
            title = self.labels.get(f"{key}_title")
            if title:
                title.setText(f"detsum_{fine_id}_{self.element_names[idx]}.tiff ({color_names[idx]})")
        merged_title = self.labels.get("merged_title")
        if merged_title:
            merged_title.setText("Merged RGB")
        self.status_lbl.setText(f"Loaded fine ID {fine_id}")

    def clear_images(self, message: str):
        for lbl in self.labels.values():
            lbl.setPixmap(QPixmap())
            lbl.setText("No Image")
        self.status_lbl.setText(message)


if __name__ == "__main__":
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    app = QApplication(sys.argv)
    viewer = SingleScanView(initial)
    viewer.show()
    sys.exit(app.exec_())
