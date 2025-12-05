import os
import json
import sys
import io
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt, QRectF, QSizeF, QPointF, QEvent, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QImage, QPdfWriter
from PyQt5.QtGui import QPagedPaintDevice
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
    QComboBox,
    QWidget,
    QSizePolicy,
    QGroupBox,
    QTabWidget,
    QComboBox,
)
from PyQt5.QtWidgets import QToolTip

# Reuse configuration and classifier logic from DetailBeta
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from viewers.DetailBeta import ScansGroupedViewer, PROCESSED_SCANS_DIR
from utils import compute_px_per_um


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
    channel_map: Dict[str, str]
    group: str
    width_px: int
    height_px: int
    px_per_um: float
    real_size_um: tuple

    def get_pixmap(self, outlines_enabled: bool) -> QPixmap:
        return self.pixmap_outlined if outlines_enabled else self.pixmap_plain

    def channel_label(self) -> str:
        return "R: {r}  G: {g}  B: {b}".format(
            r=self.channel_map.get("r", "?"),
            g=self.channel_map.get("g", "?"),
            b=self.channel_map.get("b", "?"),
        )


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
            size_text = f"Size: {item.width_px}x{item.height_px} px"
            if item.real_size_um and any(v > 0 for v in item.real_size_um):
                size_text += " | {:.2f}x{:.2f} um".format(item.real_size_um[0], item.real_size_um[1])
            lbl.setToolTip(
                f"Coarse {item.coarse_id} | Fine {item.fine_id}\nType: {item.classification}\nElements: {item.elements}\nChannels: {item.channel_label()}\n{size_text}"
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

    def render_to_image(self, size: QSize = QSize(2000, 1200)) -> QImage:
        img = QImage(size, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        painter = QPainter(img)
        self._draw_graph(painter, QRectF(QPointF(0, 0), QSizeF(size)))
        painter.end()
        return img

    def export_png(self, path: str, size: QSize = QSize(2000, 1200)):
        image = self.render_to_image(size)
        image.save(path)

    def export_pdf(self, path: str, size: QSize = QSize(2000, 1200), dpi: int = 300):
        image = self.render_to_image(size)
        writer = QPdfWriter(path)
        writer.setResolution(dpi)
        width_mm = (image.width() / dpi) * 25.4
        height_mm = (image.height() / dpi) * 25.4
        writer.setPageSizeMM(QSizeF(width_mm, height_mm))
        painter = QPainter(writer)
        painter.drawImage(0, 0, image)
        painter.end()

    def _draw_graph(self, painter: QPainter, bounds: QRectF):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(bounds, QColor("#101010"))

        base_font = painter.font()
        base_font.setPointSize(16)
        painter.setFont(base_font)
        data_empty = not self.counts if self.mode == "total" else not self.stacked_data
        if data_empty:
            painter.setPen(Qt.white)
            painter.drawText(bounds, Qt.AlignCenter, "No data to display")
            return

        # Title
        painter.setPen(Qt.white)
        title_font = QFont(base_font)
        title_font.setPointSize(max(title_font.pointSize() + 4, 18))
        painter.setFont(title_font)
        title_text = "Type Counts - Total" if self.mode != "stacked" else "Type Counts - Stacked by Group"
        if hasattr(self, "_title_suffix") and self._title_suffix:
            title_text = f"{title_text} ({self._title_suffix})"
        painter.drawText(QRectF(bounds.left(), bounds.top() + 8, bounds.width(), 32), Qt.AlignHCenter | Qt.AlignTop, title_text)
        painter.setFont(base_font)

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

        margin_lr = 90
        margin_top = 80
        margin_bottom = 110
        axis_rect = QRectF(
            bounds.left() + margin_lr,
            bounds.top() + margin_top,
            bounds.width() - 2 * margin_lr,
            bounds.height() - (margin_top + margin_bottom),
        )
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
            cat_label_height = margin_bottom * 0.6
            painter.drawText(
                QRectF(x_center - bar_slot / 2, axis_rect.bottom(), bar_slot, cat_label_height),
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
        self.group_configs = ScansGroupedViewer.SCAN_GROUPS_CONFIG
        self.px_per_um_cache: Dict[str, float] = {}
        self.box_cache: Dict[str, List[dict]] = {}
        self.size_bins: List[str] = []
        self.size_overall_counts: Dict[str, int] = {}
        self.size_category_counts: Dict[str, Dict[str, int]] = {}
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
        self.category_order = ["Separate", "Partial", "Together"]

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

        group_tab = QWidget()
        group_layout = QVBoxLayout(group_tab)
        selector_row = QHBoxLayout()
        selector_label = QLabel("Elemental group:")
        self.group_selector = QComboBox()
        for group_key in self.group_order:
            label = self.group_configs.get(group_key, {}).get("name", group_key)
            self.group_selector.addItem(label, group_key)
        self.group_selector.currentIndexChanged.connect(self.update_group_graph)
        selector_row.addWidget(selector_label)
        selector_row.addWidget(self.group_selector, 1)
        selector_row.addStretch()
        group_layout.addLayout(selector_row)
        self.group_graph = BarGraphWidget(
            self.group_type_counts(self.group_order[0]),
            {},
            category_order=self.category_order,
        )
        self.group_graph.setMinimumHeight(320)
        group_layout.addWidget(self.group_graph, 1)
        tabs.addTab(group_tab, "Group Counts")

        size_tab = QWidget()
        size_layout = QVBoxLayout(size_tab)
        controls_size = QHBoxLayout()
        controls_size.addWidget(QLabel("Category:"))
        self.size_category_selector = QComboBox()
        self.size_category_selector.addItem("Total (all types)", "total")
        for cat in self.category_order:
            self.size_category_selector.addItem(cat, cat)
        controls_size.addSpacing(12)
        controls_size.addWidget(QLabel("Group:"))
        self.size_group_selector = QComboBox()
        self.size_group_selector.addItem("All groups", "all")
        for g in self.group_order:
            self.size_group_selector.addItem(g, g)
        self.size_category_selector.currentIndexChanged.connect(self.update_size_graph)
        self.size_group_selector.currentIndexChanged.connect(self.update_size_graph)
        controls_size.addWidget(self.size_category_selector)
        controls_size.addWidget(self.size_group_selector)
        controls_size.addStretch()
        size_layout.addLayout(controls_size)
        self.size_overall_graph = BarGraphWidget({})
        self.size_overall_graph.setMinimumHeight(360)
        size_layout.addWidget(self.size_overall_graph)
        tabs.addTab(size_tab, "Size Distribution")

        export_tab = QWidget()
        export_layout = QVBoxLayout(export_tab)
        export_layout.addStretch()
        self.export_all_btn = QPushButton("Export All Assets")
        self.export_all_btn.setMinimumHeight(40)
        self.export_all_btn.clicked.connect(self.export_all_assets)
        export_layout.addWidget(self.export_all_btn)
        export_layout.addStretch()
        tabs.addTab(export_tab, "Export")

        # Now that graphs exist, populate them with computed distributions
        self.compute_size_distributions()

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
                f.write(
                    "coarse:{coarse}, fine:{fine}, type:{cls}, elements:{elements}, group:{group}\n".format(
                        coarse=item.coarse_id,
                        fine=item.fine_id,
                        cls=item.classification,
                        elements=item.elements,
                        group=item.group,
                    )
                )
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

    def group_type_counts(self, group: str) -> Dict[str, int]:
        counts = {"Separate": 0, "Partial": 0, "Together": 0}
        counts.update(self.stacked_counts.get(group, {}))
        return counts

    def update_group_graph(self):
        group_key = self.group_selector.currentData()
        if not group_key:
            return
        self.group_graph.set_counts(self.group_type_counts(group_key))

    def extract_px_per_um_from_box(self, box: Dict) -> float:
        if not isinstance(box, dict):
            return None
        if "px_per_um" in box and box["px_per_um"]:
            try:
                return float(box["px_per_um"])
            except (TypeError, ValueError):
                return None
        return compute_px_per_um(box)

    def load_boxes_for_scan(self, scan_dir: str, group_config: Dict) -> List[dict]:
        json_name = group_config.get("json_name")
        if not json_name:
            return []
        cache_key = os.path.join(scan_dir, json_name)
        if cache_key in self.box_cache:
            return self.box_cache[cache_key]
        json_path = os.path.join(scan_dir, json_name)
        if not os.path.exists(json_path):
            self.box_cache[cache_key] = []
            return []
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            self.box_cache[cache_key] = []
            return []

        boxes: List[dict] = []
        if isinstance(data, list):
            boxes = [b for b in data if isinstance(b, dict)]
        elif isinstance(data, dict):
            # Sort dict keys numerically if possible to preserve box order
            try:
                def key_num(k):
                    import re
                    m = re.search(r"(\\d+)", k)
                    return int(m.group(1)) if m else 0
                for k in sorted(data.keys(), key=key_num):
                    if isinstance(data[k], dict):
                        boxes.append(data[k])
            except Exception:
                boxes = [v for v in data.values() if isinstance(v, dict)]
        # Sort by real_center_um to align with fine-scan ordering on disk
        def center_key(box):
            if isinstance(box, dict) and "real_center_um" in box and isinstance(box["real_center_um"], (list, tuple)):
                cx, cy = box["real_center_um"][0], box["real_center_um"][1] if len(box["real_center_um"]) > 1 else 0
                return (cx, cy)
            return (0, 0)
        boxes = sorted(boxes, key=center_key)
        self.box_cache[cache_key] = boxes
        return boxes

    def px_per_um_for_scan(self, scan_dir: str, group_config: Dict) -> float:
        cache_key = os.path.join(scan_dir, group_config.get("json_name", ""))
        if cache_key in self.px_per_um_cache:
            return self.px_per_um_cache[cache_key]
        boxes = self.load_boxes_for_scan(scan_dir, group_config)
        px_per_um = None
        for box in boxes:
            px_per_um = self.extract_px_per_um_from_box(box)
            if px_per_um:
                break

        self.px_per_um_cache[cache_key] = px_per_um
        return px_per_um

    def size_value_um(self, item: ClassifiedFine) -> float:
        if item.real_size_um and any(v > 0 for v in item.real_size_um):
            return max(item.real_size_um)
        if item.px_per_um > 0:
            return max(item.width_px, item.height_px) / item.px_per_um
        return None

    def build_size_bins(self) -> List[str]:
        # Fixed bins from 0–10 microns, plus an "Extreme" bin for >10
        step = 1.0
        labels = []
        start = 0.0
        end = 10.0
        cur = start
        while cur < end:
            nxt = cur + step
            labels.append(f"{int(cur)}-{int(nxt)}")
            cur = nxt
        labels.append("Extreme")  # >10
        return labels

    def histogram_counts(self, values: List[float], labels: List[str]) -> Dict[str, int]:
        counts = {label: 0 for label in labels}
        if not labels:
            return counts
        step = 1.0
        for v in values:
            if v is None:
                continue
            if v > 10.0:
                counts["Extreme"] += 1
                continue
            if v < 0:
                v = 0
            idx = int(v // step)
            idx = min(idx, 9)  # 0-9 cover 0-10
            counts[labels[idx]] += 1
        return counts

    def compute_size_distributions(self):
        values = []
        by_cat = {cat: [] for cat in self.category_order}
        by_group_cat = {grp: {cat: [] for cat in self.category_order} for grp in self.group_order}
        by_group_total = {grp: [] for grp in self.group_order}
        for cat, items in self.categories.items():
            for item in items:
                v = self.size_value_um(item)
                if v:
                    values.append(v)
                    by_cat[cat].append(v)
                    if item.group in by_group_cat:
                        by_group_cat[item.group][cat].append(v)
                        by_group_total[item.group].append(v)

        labels = self.build_size_bins()
        self.size_bins = labels
        self.size_overall_counts = self.histogram_counts(values, labels)
        self.size_category_counts = {}
        for cat in self.category_order:
            self.size_category_counts[cat] = self.histogram_counts(by_cat.get(cat, []), labels)
        self.size_group_total_counts = {}
        self.size_group_category_counts = {}
        for grp in self.group_order:
            self.size_group_total_counts[grp] = self.histogram_counts(by_group_total.get(grp, []), labels)
            self.size_group_category_counts[grp] = {}
            for cat in self.category_order:
                self.size_group_category_counts[grp][cat] = self.histogram_counts(by_group_cat.get(grp, {}).get(cat, []), labels)

        # Update graphs if initialized
        if hasattr(self, "size_overall_graph") and self.size_overall_graph:
            self.update_size_graph()

    def update_size_graph(self):
        cat_key = self.size_category_selector.currentData() if hasattr(self, "size_category_selector") else "total"
        group_key = self.size_group_selector.currentData() if hasattr(self, "size_group_selector") else "all"
        labels = self.size_bins or self.build_size_bins()
        counts = {}
        if group_key == "all":
            if cat_key == "total":
                counts = self.size_overall_counts
            else:
                counts = self.size_category_counts.get(cat_key, {})
        else:
            if cat_key == "total":
                counts = self.size_group_total_counts.get(group_key, {})
            else:
                counts = self.size_group_category_counts.get(group_key, {}).get(cat_key, {})
        # Ensure we always have counts for all labels
        counts = {label: counts.get(label, 0) for label in labels}
        self.size_overall_graph.set_counts(counts)
        self.size_overall_graph.category_order = labels
        self.size_overall_graph.update()

    def export_scan_stats(self, dest: Path):
        stats_path = dest / "CourseFineScanStats.csv"
        header = [
            "coarse_id",
            "fine_id",
            "group",
            "classification",
            "width_px",
            "height_px",
            "px_per_um",
            "size_um_x",
            "size_um_y",
        ]
        rows = []
        for cat, items in self.categories.items():
            for item in items:
                sx, sy = (item.real_size_um if item.real_size_um else (0.0, 0.0))
                rows.append(
                    [
                        item.coarse_id,
                        item.fine_id,
                        item.group,
                        item.classification,
                        item.width_px,
                        item.height_px,
                        round(item.px_per_um, 4),
                        round(sx, 4),
                        round(sy, 4),
                    ]
                )
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                f.write(",".join(header) + "\n")
                for row in rows:
                    f.write(",".join(str(v) for v in row) + "\n")
        except OSError:
            pass

    def toggle_stacked_mode(self, state: int):
        mode = "stacked" if state == Qt.Checked else "total"
        self.bar_graph.set_mode(mode)

    def export_bar_graph(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Type Counts", "type_counts.svg", "SVG Files (*.svg)")
        if path:
            if not path.lower().endswith(".svg"):
                path += ".svg"
            self.bar_graph.export_svg(path)

    def export_bar_graph_variants(self, dest: Path):
        original_mode = self.bar_graph.mode
        original_checked = self.stacked_toggle.isChecked()
        prev_block = self.stacked_toggle.blockSignals(True)
        try:
            self.bar_graph._title_suffix = "All Groups"
            self.bar_graph.set_mode("total")
            self.stacked_toggle.setChecked(False)
            self.bar_graph.export_png(str(dest / "type_counts_total.png"))

            self.bar_graph.set_mode("stacked")
            self.stacked_toggle.setChecked(True)
            self.bar_graph.export_png(str(dest / "type_counts_stacked.png"))
        finally:
            self.bar_graph._title_suffix = ""
            self.bar_graph.set_mode(original_mode)
            self.stacked_toggle.setChecked(original_checked)
            self.stacked_toggle.blockSignals(prev_block)

    def export_group_bar_graphs(self, dest: Path):
        original_group = self.group_selector.currentData()
        for group_key in self.group_order:
            self.group_graph.set_counts(self.group_type_counts(group_key))
            self.group_graph._title_suffix = group_key
            safe_name = group_key.lower()
            self.group_graph.export_png(str(dest / f"type_counts_{safe_name}.png"))
        if original_group is not None:
            self.group_graph.set_counts(self.group_type_counts(original_group))
        self.group_graph._title_suffix = ""

    def export_size_graphs(self, dest: Path):
        if not self.size_bins:
            return
        overall_path_png = dest / "size_distribution_overall.png"
        self.size_overall_graph.export_png(str(overall_path_png))

    def export_all_assets(self):
        dest_dir = QFileDialog.getExistingDirectory(self, "Select Export Destination")
        if not dest_dir:
            return
        dest = Path(dest_dir)

        # Export bar graph variants (total and separated) as high-res PNG
        self.export_bar_graph_variants(dest)

        # Export per-group bar graphs as PNG
        self.export_group_bar_graphs(dest)

        # Export size distribution graphs as PNG
        self.export_size_graphs(dest)

        # Export 5x5 grids per category (paged) as PNG and PDF with channel labels
        self.export_category_grids(dest)
        self.export_scan_stats(dest)

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

    def save_pdf_from_image(self, image: QImage, path: Path, dpi: int = 300):
        writer = QPdfWriter(str(path))
        writer.setResolution(dpi)
        width_mm = (image.width() / dpi) * 25.4
        height_mm = (image.height() / dpi) * 25.4
        writer.setPageSizeMM(QSizeF(width_mm, height_mm))
        painter = QPainter(writer)
        painter.drawImage(0, 0, image)
        painter.end()

    def export_category_grids(self, dest: Path):
        cols = 5
        rows = 5
        cell = 300
        label_height = 44
        margin = 40
        spacing = 20
        title_height = 80
        width = margin * 2 + cols * cell + spacing * (cols - 1)
        height = margin * 2 + title_height + rows * cell + spacing * (rows - 1)
        size = QSize(int(width), int(height))
        combined_pages: List[QImage] = []
        label_font = QFont()
        label_font.setPointSize(10)

        def chunks(seq, n):
            for i in range(0, len(seq), n):
                yield seq[i : i + n]

        for category, items in self.categories.items():
            if not items:
                continue
            pages = list(chunks(items, cols * rows))
            for idx, page in enumerate(pages):
                image = QImage(size, QImage.Format_ARGB32)
                image.fill(Qt.white)
                painter = QPainter(image)
                painter.fillRect(QRectF(QPointF(0, 0), QSizeF(size)), Qt.white)
                title = f"{category} (page {idx + 1})"
                painter.setPen(Qt.black)
                title_font = QFont()
                title_font.setPointSize(18)
                title_font.setBold(True)
                painter.setFont(title_font)
                painter.drawText(QRectF(0, margin / 2, width, title_height), Qt.AlignCenter, title)

                for pos, item in enumerate(page):
                    row = pos // cols
                    col = pos % cols
                    x = margin + col * (cell + spacing)
                    y = margin + title_height + row * (cell + spacing)
                    pix = item.get_pixmap(self.outlines_enabled)
                    image_area_height = cell - label_height
                    scaled = pix.scaled(cell, image_area_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    target_x = x + (cell - scaled.width()) / 2
                    target_y = y + (image_area_height - scaled.height()) / 2
                    painter.drawPixmap(int(target_x), int(target_y), scaled)
                    painter.setFont(label_font)
                    painter.setPen(Qt.black)
                    channel_rect = QRectF(x, y + image_area_height, cell, label_height / 3)
                    size_rect = QRectF(x, y + image_area_height + label_height / 3, cell, label_height / 3)
                    id_rect = QRectF(x, y + image_area_height + 2 * label_height / 3, cell, label_height / 3)
                    painter.drawText(
                        channel_rect,
                        Qt.AlignCenter | Qt.AlignVCenter,
                        item.channel_label(),
                    )
                    size_text = "Size: {:.2f}x{:.2f} um".format(
                        item.real_size_um[0], item.real_size_um[1]
                    ) if item.real_size_um and any(v > 0 for v in item.real_size_um) else ""
                    painter.drawText(
                        size_rect,
                        Qt.AlignCenter | Qt.AlignVCenter,
                        size_text,
                    )
                    painter.drawText(
                        id_rect,
                        Qt.AlignCenter | Qt.AlignVCenter,
                        f"C:{item.coarse_id}  F:{item.fine_id}",
                    )

                painter.end()
                base = f"{category.lower()}_{idx + 1}"
                png_path = dest / f"{base}.png"
                image.save(str(png_path))
                combined_pages.append(image)

        if combined_pages:
            combined_pdf_path = dest / "TypesImages.pdf"
            writer = QPdfWriter(str(combined_pdf_path))
            writer.setResolution(300)
            writer.setPageSizeMM(QSizeF((width / 300) * 25.4, (height / 300) * 25.4))
            painter = QPainter(writer)
            for idx, img in enumerate(combined_pages):
                if idx > 0:
                    writer.newPage()
                painter.drawImage(0, 0, img)
            painter.end()

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
                boxes = self.load_boxes_for_scan(scan_dir, group_config)
                px_per_um_scan = self.px_per_um_for_scan(scan_dir, group_config)
                for fine_id in fine_ids:
                    box_for_fine = None
                    # Map fine_id to box index for json_boxes logic: fine_id = coarse_id + idx + 1
                    if boxes:
                        idx = fine_id - coarse_id - 1
                        if 0 <= idx < len(boxes):
                            box_for_fine = boxes[idx]
                    px_per_um = self.extract_px_per_um_from_box(box_for_fine) if box_for_fine else None
                    if not px_per_um:
                        px_per_um = px_per_um_scan
                    try:
                        rgb_np = load_rgb_image(scan_dir, fine_id, group_config["elements_map"])
                    except FileNotFoundError:
                        continue
                    h, w, _ = rgb_np.shape
                    elements_str = "".join(group_config.get("elements", []))
                    channel_map = dict(group_config.get("elements_map", {}))
                    real_size_um = None
                    if box_for_fine and "real_size_um" in box_for_fine:
                        real_size_um = tuple(box_for_fine["real_size_um"])
                    elif px_per_um:
                        real_size_um = (w / px_per_um, h / px_per_um)
                    classification = self.classifier.typeDetector(
                        [rgb_np[:, :, 0], rgb_np[:, :, 1], rgb_np[:, :, 2]],
                        fine_id=fine_id,
                        elements=elements_str,
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
                            channel_map=channel_map,
                            group=group_key,
                            width_px=w,
                            height_px=h,
                            px_per_um=px_per_um or 0.0,
                            real_size_um=real_size_um or (0.0, 0.0),
                        )
                    )
                    safe_cat = classification or "Separate"
                    if group_key not in self.stacked_counts:
                        self.stacked_counts[group_key] = {"Separate": 0, "Partial": 0, "Together": 0}
                    if safe_cat not in self.stacked_counts[group_key]:
                        self.stacked_counts[group_key][safe_cat] = 0
                    self.stacked_counts[group_key][safe_cat] += 1

        self.compute_size_distributions()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = TypeViewer()
    viewer.show()
    sys.exit(app.exec_())
