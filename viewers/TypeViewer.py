import os
import json
import sys
import io
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QRectF, QSizeF, QPointF, QEvent, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont
from PyQt5.QtSvg import QSvgGenerator
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
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
    QTabWidget,
)
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
    group: str

    def get_pixmap(self, outlines_enabled: bool) -> QPixmap:
        return self.pixmap_outlined if outlines_enabled else self.pixmap_plain


class ClickableLabel(QLabel):
    clicked = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._item = None

    def set_item(self, item):
        self._item = item

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._item is not None:
            self.clicked.emit(self._item)
        super().mousePressEvent(event)


class CategoryPanel(QWidget):
    def __init__(
        self,
        name: str,
        items: List[ClassifiedFine],
        outlines_enabled: bool,
        on_nav=None,
        on_click=None,
        parent=None,
    ):
        super().__init__(parent)
        self.name = name
        self.items = items
        self.outlines_enabled = outlines_enabled
        self.page = 0
        self.per_page = 9
        self.on_nav = on_nav
        self.on_click = on_click

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
            lbl = ClickableLabel()
            lbl.setText("No Image")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background-color:#222;border:1px solid #444;color:white;")
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setScaledContents(True)
            lbl.setMinimumSize(120, 120)
            lbl.setMouseTracking(True)
            lbl.setAttribute(Qt.WA_AlwaysShowToolTips, True)
            lbl.setAttribute(Qt.WA_Hover, True)
            lbl.installEventFilter(self)
            lbl.clicked.connect(self.handle_click)
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
            lbl.set_item(item)
        for lbl in self.slots[len(subset) :]:
            lbl.setPixmap(QPixmap())
            lbl.setText("No Image")
            lbl.setToolTip("No Image")
            lbl.set_item(None)
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

    def handle_click(self, item: ClassifiedFine):
        if self.on_click and item:
            self.on_click(item)


class BarGraphWidget(QWidget):
    def __init__(
        self,
        counts: Dict[str, int],
        stacked_data: Dict[str, Dict[str, int]] = None,
        group_order=None,
        group_colors=None,
        category_order=None,
        parent=None,
    ):
        super().__init__(parent)
        self.counts = counts
        self.stacked_data = stacked_data or {}
        self.group_order = list(group_order) if group_order else list(self.stacked_data.keys())
        self.group_colors = group_colors or {}
        self.mode = "total"  # "total" or "stacked"
        self.category_order = list(category_order) if category_order else list(self.counts.keys())
        self.setMinimumHeight(320)

    def set_counts(self, counts: Dict[str, int]):
        self.counts = counts
        if not self.category_order:
            self.category_order = list(counts.keys())
        self.update()

    def set_stacked_data(self, stacked_data: Dict[str, Dict[str, int]], group_order=None):
        self.stacked_data = stacked_data or {}
        if group_order is not None:
            self.group_order = list(group_order)
        self.update()

    def set_mode(self, mode: str):
        self.mode = mode
        self.update()

    def _draw_graph(self, painter: QPainter, bounds: QRectF):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(bounds, QColor("#101010"))

        base_font = painter.font()
        data_empty = not self.counts if self.mode == "total" else not self.stacked_data
        if data_empty:
            painter.setPen(Qt.white)
            painter.drawText(bounds, Qt.AlignCenter, "No data to display")
            return

        def ordered_categories(available: List[str]) -> List[str]:
            if self.category_order:
                return [c for c in self.category_order if c in available]
            return available

        if self.mode == "stacked" and self.stacked_data:
            all_cats = {cat for group_vals in self.stacked_data.values() for cat in group_vals.keys()}
            categories = ordered_categories(list(all_cats))
            values = []
            for cat in categories:
                total = sum(self.stacked_data.get(group, {}).get(cat, 0) for group in self.group_order or self.stacked_data.keys())
                values.append(total)
        else:
            categories = ordered_categories(list(self.counts.keys()))
            values = [self.counts.get(c, 0) for c in categories]

        max_val = max(values) if values else 0
        if max_val == 0:
            painter.setPen(Qt.white)
            painter.drawText(bounds, Qt.AlignCenter, "All counts are zero")
            return

        margin = 50
        axis_rect = QRectF(bounds.left() + margin, bounds.top() + margin, bounds.width() - 2 * margin, bounds.height() - 2 * margin)
        bar_slot = axis_rect.width() / max(len(categories), 1)
        bar_width = bar_slot * 0.6
        colors = {"Separate": QColor("#66aaff"), "Partial": QColor("#ffb347"), "Together": QColor("#6ee7b7")}

        painter.setPen(QColor("#888888"))
        painter.drawLine(axis_rect.bottomLeft(), axis_rect.bottomRight())
        painter.drawLine(axis_rect.bottomLeft(), axis_rect.topLeft())

        for idx, (cat, val) in enumerate(zip(categories, values)):
            x_center = axis_rect.left() + (idx + 0.5) * bar_slot
            bar_height = (val / max_val) * axis_rect.height()
            bar_rect = QRectF(
                x_center - bar_width / 2,
                axis_rect.bottom() - bar_height,
                bar_width,
                bar_height,
            )
            if self.mode == "stacked" and self.stacked_data:
                y_cursor = bar_rect.bottom()
                for idx_group, group in enumerate(self.group_order):
                    val_seg = self.stacked_data.get(group, {}).get(cat, 0)
                    if val_seg == 0:
                        continue
                    seg_height = (val_seg / max_val) * axis_rect.height()
                    seg_rect = QRectF(bar_rect.left(), y_cursor - seg_height, bar_width, seg_height)
                    painter.setBrush(self.group_colors.get(group, QColor("#cccccc")))
                    painter.setPen(Qt.NoPen)
                    painter.drawRect(seg_rect)
                    y_cursor -= seg_height
            else:
                painter.setBrush(colors.get(cat, QColor("#cccccc")))
                painter.setPen(Qt.NoPen)
                painter.drawRect(bar_rect)

            # Category label along the x-axis
            painter.setPen(Qt.white)
            cat_font = QFont(base_font)
            if cat_font.pointSize() > 0:
                cat_font.setPointSize(cat_font.pointSize() + 4)
            else:
                cat_font.setPointSize(14)
            painter.setFont(cat_font)
            painter.drawText(
                QRectF(bar_rect.left(), axis_rect.bottom(), bar_rect.width(), margin / 2),
                Qt.AlignHCenter | Qt.AlignTop,
                cat,
            )

            # Value label above the bar, with a larger font for readability
            value_font = QFont(base_font)
            if value_font.pointSize() > 0:
                value_font.setPointSize(value_font.pointSize() + 6)
            else:
                value_font.setPointSize(16)
            painter.setFont(value_font)
            fm = painter.fontMetrics()
            label_height = fm.height()
            # Keep label above the bar while ensuring it stays within the view bounds
            top_y = max(bounds.top() + 4, bar_rect.top() - label_height - 6)
            painter.drawText(
                QRectF(bar_rect.left(), top_y, bar_rect.width(), label_height + 4),
                Qt.AlignHCenter | Qt.AlignBottom,
                str(val),
            )
            painter.setFont(base_font)

        # Legend for stacked mode
        if self.mode == "stacked" and self.stacked_data and self.group_order:
            legend_margin = 12
            swatch_size = QSizeF(18, 14)
            spacing = 10
            x = bounds.right() - legend_margin - swatch_size.width()
            y = bounds.top() + legend_margin
            legend_font = QFont(base_font)
            legend_font.setPointSize(max(legend_font.pointSize() + 2, 12))
            painter.setFont(legend_font)
            fm = painter.fontMetrics()
            row_height = max(swatch_size.height(), fm.height())
            max_label_width = max((fm.width(g) for g in self.group_order), default=0)
            legend_width = swatch_size.width() + spacing + max_label_width
            x = bounds.right() - legend_width - legend_margin
            for group in self.group_order:
                painter.setBrush(self.group_colors.get(group, QColor("#cccccc")))
                painter.setPen(Qt.NoPen)
                rect_y = y + (row_height - swatch_size.height()) / 2
                painter.drawRect(QRectF(x, rect_y, swatch_size.width(), swatch_size.height()))
                painter.setPen(Qt.white)
                painter.drawText(
                    QRectF(x + swatch_size.width() + spacing, y, max_label_width, row_height),
                    Qt.AlignLeft | Qt.AlignVCenter,
                    group,
                )
                y += row_height + spacing

    def paintEvent(self, event):
        painter = QPainter(self)
        self._draw_graph(painter, QRectF(QPointF(0, 0), QSizeF(self.width(), self.height())))
        painter.end()

    def export_svg(self, path: str):
        generator = QSvgGenerator()
        generator.setFileName(path)
        generator.setSize(self.size())
        generator.setViewBox(QRectF(0, 0, self.width(), self.height()))
        generator.setTitle("Type Counts")
        painter = QPainter(generator)
        self._draw_graph(painter, QRectF(QPointF(0, 0), QSizeF(self.width(), self.height())))
        painter.end()


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
        self.clicked_log_path = Path(__file__).with_name("clicked.txt")
        self.group_order = ["CuCaFe", "FeCaSi", "CrFeMn"]
        self.group_colors = {
            "CuCaFe": QColor("#8ecae6"),
            "FeCaSi": QColor("#ff9f1c"),
            "CrFeMn": QColor("#9b8df2"),
        }

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        tabs = QTabWidget()
        root_layout.addWidget(tabs)

        gallery_tab = QWidget()
        gallery_layout = QVBoxLayout(gallery_tab)

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
        gallery_layout.addLayout(top_bar)

        row_layout = QHBoxLayout()
        gallery_layout.addLayout(row_layout)

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
            panel = CategoryPanel(
                cat_name,
                self.categories[cat_name],
                self.outlines_enabled,
                on_nav=self.set_active_panel,
                on_click=self.handle_item_click,
            )
            vbox.addWidget(panel)
            row_layout.addWidget(box, 1)
            self.panels[cat_name] = panel

        tabs.addTab(gallery_tab, "Gallery")

        bar_tab = QWidget()
        bar_layout = QVBoxLayout(bar_tab)
        self.category_order = ["Separate", "Partial", "Together"]
        self.bar_graph = BarGraphWidget(
            self.type_counts(),
            self.stacked_counts,
            group_order=self.group_order,
            group_colors=self.group_colors,
            category_order=self.category_order,
        )
        bar_layout.addWidget(self.bar_graph, 1)
        controls_row = QHBoxLayout()
        self.stacked_toggle = QCheckBox("Stacked by group")
        self.stacked_toggle.stateChanged.connect(self.toggle_stacked_mode)
        controls_row.addWidget(self.stacked_toggle)
        controls_row.addStretch()
        self.export_svg_btn = QPushButton("Export SVG")
        self.export_svg_btn.clicked.connect(self.export_bar_graph)
        controls_row.addWidget(self.export_svg_btn)
        bar_layout.addLayout(controls_row)
        tabs.addTab(bar_tab, "Type Counts")

    def toggle_outlines(self, checked: bool):
        self.outlines_enabled = checked
        self.toggle_outlines_btn.setText("Hide Outlines" if checked else "Show Outlines")
        for panel in self.panels.values():
            panel.set_outlines_enabled(self.outlines_enabled)
        self.rebuild_pixmaps()

    def set_active_panel(self, name: str):
        self.last_active_panel = name

    def handle_item_click(self, item: ClassifiedFine):
        try:
            with open(self.clicked_log_path, "a", encoding="utf-8") as f:
                f.write(f"coarse:{item.coarse_id}, fine:{item.fine_id}, type:{item.classification}\n")
        except OSError:
            pass

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

    def type_counts(self) -> Dict[str, int]:
        return {cat: len(items) for cat, items in self.categories.items()}

    def toggle_stacked_mode(self, state: int):
        mode = "stacked" if state == Qt.Checked else "total"
        self.bar_graph.set_mode(mode)

    def export_bar_graph(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Type Counts", "type_counts.svg", "SVG Files (*.svg)")
        if path:
            if not path.lower().endswith(".svg"):
                path += ".svg"
            self.bar_graph.export_svg(path)

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
        self.stacked_counts = {grp: {"Separate": 0, "Partial": 0, "Together": 0} for grp in self.group_order}

        for group_key in self.group_order:
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
                            group=group_key,
                        )
                    )
                    safe_cat = classification or "Separate"
                    if group_key not in self.stacked_counts:
                        self.stacked_counts[group_key] = {"Separate": 0, "Partial": 0, "Together": 0}
                    if safe_cat not in self.stacked_counts[group_key]:
                        self.stacked_counts[group_key][safe_cat] = 0
                    self.stacked_counts[group_key][safe_cat] += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = TypeViewer()
    viewer.show()
    sys.exit(app.exec_())
