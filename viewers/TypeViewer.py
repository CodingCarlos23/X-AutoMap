import os
import json
import sys
import io
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QShortcut,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QGroupBox,
)
from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QToolTip

# Reuse configuration and classifier logic from DetailBeta
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from viewers.DetailBeta import ScansGroupedViewer, PROCESSED_SCANS_DIR


def normalize_channel(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    cmax, cmin = arr.max(), arr.min()
    if cmax > cmin:
        arr = (arr - cmin) / (cmax - cmin) * 255.0
    return arr.astype(np.uint8)


def load_rgb_image(scan_dir: str, fine_id: int, elements_map: Dict[str, str]) -> np.ndarray:
    r_img = np.array(Image.open(os.path.join(scan_dir, f"detsum_{fine_id}_{elements_map['r']}.tiff")))
    g_img = np.array(Image.open(os.path.join(scan_dir, f"detsum_{fine_id}_{elements_map['g']}.tiff")))
    b_img = np.array(Image.open(os.path.join(scan_dir, f"detsum_{fine_id}_{elements_map['b']}.tiff")))
    return np.stack([normalize_channel(r_img), normalize_channel(g_img), normalize_channel(b_img)], axis=-1)


def processed_scan_dir(coarse_id: int, group_config: Dict) -> str:
    suffix = group_config.get("folder_suffix") or "".join(group_config["elements"])
    preferred_dir = os.path.join(PROCESSED_SCANS_DIR, f"{coarse_id}_{suffix}")
    legacy_dir = os.path.join(PROCESSED_SCANS_DIR, str(coarse_id))
    return preferred_dir if os.path.isdir(preferred_dir) or not os.path.isdir(legacy_dir) else legacy_dir


def fine_ids_for_scan(coarse_id: int, group_config: Dict, next_coarse_id: int = None) -> List[int]:
    logic = group_config.get("fine_id_logic")
    if logic == "range_between_coarse":
        end_id = next_coarse_id if next_coarse_id is not None else coarse_id + 8
        return list(range(coarse_id + 1, end_id))
    if logic == "json_boxes":
        scan_dir = processed_scan_dir(coarse_id, group_config)
        json_path = os.path.join(scan_dir, group_config["json_name"])
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                return [coarse_id + i for i in range(1, len(data) + 1)]
            except (IOError, json.JSONDecodeError):
                return []
    return []


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    pix = QPixmap()
    pix.loadFromData(buf.getvalue(), "PNG")
    return pix


@dataclass
class ClassifiedFine:
    coarse_id: int
    fine_id: int
    classification: str
    pixmap_plain: QPixmap
    pixmap_outlined: QPixmap
    elements: str

    def get_pixmap(self, outlines_enabled: bool) -> QPixmap:
        return self.pixmap_outlined if outlines_enabled else self.pixmap_plain


class CategoryPanel(QWidget):
    def __init__(self, name: str, items: List[ClassifiedFine], outlines_enabled: bool, on_nav=None, parent=None):
        super().__init__(parent)
        self.name = name
        self.items = items
        self.outlines_enabled = outlines_enabled
        self.page = 0
        self.per_page = 9
        self.on_nav = on_nav

        outer_layout = QVBoxLayout(self)
        self.setAttribute(Qt.WA_AlwaysShowToolTips, True)
        header = QHBoxLayout()
        self.label = QLabel(f"{name}: {len(items)}")
        header.addWidget(self.label)
        header.addStretch()
        self.prev_btn = QPushButton("Prev")
        self.next_btn = QPushButton("Next")
        self.prev_btn.clicked.connect(self.prev_page)
        self.next_btn.clicked.connect(self.next_page)
        header.addWidget(self.prev_btn)
        header.addWidget(self.next_btn)
        outer_layout.addLayout(header)

        grid_widget = QWidget()
        self.grid = QGridLayout(grid_widget)
        self.grid.setSpacing(8)
        outer_layout.addWidget(grid_widget)

        self.slots: List[QLabel] = []
        for idx in range(self.per_page):
            lbl = QLabel("No Image")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background-color:#222;border:1px solid #444;color:white;")
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setScaledContents(True)
            lbl.setMinimumSize(120, 120)
            lbl.setMouseTracking(True)
            lbl.setAttribute(Qt.WA_AlwaysShowToolTips, True)
            lbl.setAttribute(Qt.WA_Hover, True)
            lbl.installEventFilter(self)
            self.grid.addWidget(lbl, idx // 3, idx % 3)
            self.slots.append(lbl)

        self.update_page()

    def update_page(self):
        start = self.page * self.per_page
        subset = self.items[start : start + self.per_page]
        for lbl, item in zip(self.slots, subset):
            lbl.setPixmap(item.get_pixmap(self.outlines_enabled))
            lbl.setToolTip(
                f"Coarse {item.coarse_id} | Fine {item.fine_id}\nType: {item.classification}\nElements: {item.elements}"
            )
            lbl.setText("")
        for lbl in self.slots[len(subset) :]:
            lbl.setPixmap(QPixmap())
            lbl.setText("No Image")
            lbl.setToolTip("No Image")
        self.prev_btn.setEnabled(self.page > 0)
        self.next_btn.setEnabled(start + self.per_page < len(self.items))
        self.label.setText(f"{self.name}: {len(self.items)} (Page {self.page + 1}/{max(1, (len(self.items)-1)//self.per_page + 1)})")

    def next_page(self):
        if (self.page + 1) * self.per_page < len(self.items):
            if self.on_nav:
                self.on_nav(self.name)
            self.page += 1
            self.update_page()

    def prev_page(self):
        if self.page > 0:
            if self.on_nav:
                self.on_nav(self.name)
            self.page -= 1
            self.update_page()

    def set_outlines_enabled(self, enabled: bool):
        self.outlines_enabled = enabled
        self.update_page()

    def eventFilter(self, obj, event):
        if obj in self.slots and event.type() in (QEvent.HoverEnter, QEvent.HoverMove, QEvent.MouseMove):
            tooltip = obj.toolTip()
            if tooltip:
                global_pos = obj.mapToGlobal(event.pos())
                QToolTip.showText(global_pos, tooltip, obj)
        return super().eventFilter(obj, event)


class TypeViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fine Scan Type Viewer")
        self.setGeometry(50, 50, 1400, 800)
        self.setFocusPolicy(Qt.StrongFocus)

        self.classifier = ScansGroupedViewer.__new__(ScansGroupedViewer)
        # Ensure required attributes for outlining
        self.classifier.channel_outline_enabled = [True, True, True]
        self.outlines_enabled = True
        self.last_active_panel = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        top_bar = QHBoxLayout()
        self.toggle_outlines_btn = QPushButton("Hide Outlines")
        self.toggle_outlines_btn.setCheckable(True)
        self.toggle_outlines_btn.setChecked(True)
        self.toggle_outlines_btn.clicked.connect(self.toggle_outlines)
        top_bar.addWidget(self.toggle_outlines_btn)

        self.toggle_red = QCheckBox("Red")
        self.toggle_red.setChecked(True)
        self.toggle_red.setStyleSheet("color:red;")
        self.toggle_red.stateChanged.connect(self.update_channel_toggles)

        self.toggle_green = QCheckBox("Green")
        self.toggle_green.setChecked(True)
        self.toggle_green.setStyleSheet("color:#5bff5b;")
        self.toggle_green.stateChanged.connect(self.update_channel_toggles)

        self.toggle_blue = QCheckBox("Blue")
        self.toggle_blue.setChecked(True)
        self.toggle_blue.setStyleSheet("color:#66aaff;")
        self.toggle_blue.stateChanged.connect(self.update_channel_toggles)

        top_bar.addWidget(self.toggle_red)
        top_bar.addWidget(self.toggle_green)
        top_bar.addWidget(self.toggle_blue)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        row_layout = QHBoxLayout()
        layout.addLayout(row_layout)

        self.shortcut_left = QShortcut(Qt.Key_Left, self)
        self.shortcut_left.activated.connect(self.nav_prev)
        self.shortcut_right = QShortcut(Qt.Key_Right, self)
        self.shortcut_right.activated.connect(self.nav_next)

        self.panels = {}
        self.categories = {"Separate": [], "Partial": [], "Together": []}

        self.process_all_scans()
        for cat_name in ["Separate", "Partial", "Together"]:
            box = QGroupBox(cat_name)
            vbox = QVBoxLayout(box)
            panel = CategoryPanel(cat_name, self.categories[cat_name], self.outlines_enabled, on_nav=self.set_active_panel)
            vbox.addWidget(panel)
            row_layout.addWidget(box, 1)
            self.panels[cat_name] = panel

    def toggle_outlines(self, checked: bool):
        self.outlines_enabled = checked
        self.toggle_outlines_btn.setText("Hide Outlines" if checked else "Show Outlines")
        for panel in self.panels.values():
            panel.set_outlines_enabled(self.outlines_enabled)
        self.rebuild_pixmaps()

    def set_active_panel(self, name: str):
        self.last_active_panel = name

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            panel = self.panels.get(self.last_active_panel) or self.panels.get("Separate") or next(iter(self.panels.values()))
            if event.key() == Qt.Key_Left:
                panel.prev_page()
            else:
                panel.next_page()
            event.accept()
            return
        super().keyPressEvent(event)

    def nav_prev(self):
        panel = self.panels.get(self.last_active_panel) or self.panels.get("Separate") or next(iter(self.panels.values()))
        panel.prev_page()

    def nav_next(self):
        panel = self.panels.get(self.last_active_panel) or self.panels.get("Separate") or next(iter(self.panels.values()))
        panel.next_page()

    def update_channel_toggles(self):
        self.classifier.channel_outline_enabled = [
            self.toggle_red.isChecked(),
            self.toggle_green.isChecked(),
            self.toggle_blue.isChecked(),
        ]
        self.rebuild_pixmaps()

    def rebuild_pixmaps(self):
        for cat, items in self.categories.items():
            for item in items:
                qimg = item.pixmap_plain.toImage()
                ptr = qimg.bits()
                ptr.setsize(qimg.byteCount())
                arr = np.frombuffer(ptr, np.uint8).reshape((qimg.height(), qimg.width(), 4))
                rgb_np = arr[:, :, :3][:, :, ::-1] if qimg.format() == qimg.Format_RGBA8888 else arr[:, :, :3]
                outlined_np = self.classifier.outline_channel_largest_blobs(
                    rgb_np, channel_enabled=self.classifier.channel_outline_enabled
                )
                item.pixmap_outlined = pil_to_qpixmap(Image.fromarray(outlined_np, "RGB"))
            if cat in self.panels:
                self.panels[cat].update_page()

    def process_all_scans(self):
        config = ScansGroupedViewer.SCAN_GROUPS_CONFIG
        group_order = ["CuCaFe", "FeCaSi", "CrFeMn"]

        for group_key in group_order:
            group_config = config[group_key]
            coarse_ids = sorted(group_config["ids"])
            for idx, coarse_id in enumerate(coarse_ids):
                next_id = coarse_ids[idx + 1] if idx + 1 < len(coarse_ids) else coarse_id + 8
                fine_ids = fine_ids_for_scan(coarse_id, group_config, next_id)
                scan_dir = processed_scan_dir(coarse_id, group_config)
                if not os.path.isdir(scan_dir):
                    continue
                for fine_id in fine_ids:
                    try:
                        rgb_np = load_rgb_image(scan_dir, fine_id, group_config["elements_map"])
                    except FileNotFoundError:
                        continue
                    classification = self.classifier.typeDetector(
                        [rgb_np[:, :, 0], rgb_np[:, :, 1], rgb_np[:, :, 2]], fine_id=fine_id
                    )
                    pil_img = Image.fromarray(rgb_np, "RGB")
                    outlined_np = self.classifier.outline_channel_largest_blobs(
                        rgb_np, channel_enabled=self.classifier.channel_outline_enabled
                    )
                    pixmap_plain = pil_to_qpixmap(pil_img)
                    pixmap_outlined = pil_to_qpixmap(Image.fromarray(outlined_np, "RGB"))
                    elements_str = ", ".join(group_config["elements"])
                    self.categories[classification or "Separate"].append(
                        ClassifiedFine(
                            coarse_id=coarse_id,
                            fine_id=fine_id,
                            classification=classification or "Separate",
                            pixmap_plain=pixmap_plain,
                            pixmap_outlined=pixmap_outlined,
                            elements=elements_str,
                        )
                    )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = TypeViewer()
    viewer.show()
    sys.exit(app.exec_())
