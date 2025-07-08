import globals as G

def on_dir_selected():
    global selected_directory
    selected_directory = QFileDialog.getExistingDirectory(window, "Select Directory") 
    if not selected_directory:
        return

    all_files = sorted(os.listdir(selected_directory))  # Alphabetical (case-sensitive)
    file_list_widget.clear()
    file_paths.clear()

    for fname in all_files:
        full_path = os.path.join(selected_directory, fname)
        ext = os.path.splitext(fname)[1].lower()  # Get file extension, lowercase
        if os.path.isfile(full_path) and ext in ['.tif', '.tiff']:
            item = QListWidgetItem(f"{fname} ({ext[1:].upper()})")
            item.setCheckState(Qt.Unchecked)
            file_list_widget.addItem(item)
            file_paths.append(full_path) 

def update_selection():
    global selected_files_order

    # Check which items are checked currently
    checked_indices = [i for i in range(file_list_widget.count()) if file_list_widget.item(i).checkState() == Qt.Checked]

    # Find newly checked or unchecked items by comparing to stored order
    # Remove unchecked items from order
    selected_files_order = [i for i in selected_files_order if i in checked_indices]

    # Add newly checked items at the end
    for i in checked_indices:
        if i not in selected_files_order:
            if len(selected_files_order) < 3:
                selected_files_order.append(i)
            else:
                # Too many selected, uncheck this item immediately
                file_list_widget.item(i).setCheckState(Qt.Unchecked)

    # If less than 3 selected, clear img_paths and file_names partially
    for idx in range(3):
        if idx < len(selected_files_order):
            path = file_paths[selected_files_order[idx]]
            img_paths[idx] = path
            file_names[idx] = os.path.basename(path)
        else:
            img_paths[idx] = None
            file_names[idx] = None

def on_confirm_clicked():
    global microns_per_pixel_x
    microns_per_pixel_x = float_input_micron_x.value()
    # print(microns_per_pixel_x)
    global microns_per_pixel_y
    microns_per_pixel_y = float_input_micron_y.value()
    # print(microns_per_pixel_y)
    global true_origin_x
    true_origin_x = origin_x_input.value()
    global true_origin_y
    true_origin_y = origin_y_input.value()
    # print(true_origin_x)
    # print(true_origin_y)

    if len(selected_files_order) != 3:
        QMessageBox.warning(window, "Invalid Selection", "Please select exactly 3 items.")
        return
    QApplication.processEvents()
    init_gui()


def update_progress(val):
    progress_bar.setValue(val)
    QApplication.processEvents()


def create_manual_scan_tab():
    data_fields = {
        "min_threshold_intensity": None,
        "min_threshold_area": None,
        "microns_per_pixel_x": None,
        "microns_per_pixel_y": None,
        "true_origin_x": None,
        "true_origin_y": None,
        "number_of_scans": None,
        "debt_names": None,
        "x_motor": None,
        "x_start": None,
        "x_end": None,
        "y_motor": None,
        "y_start": None,
        "y_end": None,
        "dwell_time": None,
    }

    # Dictionary to store references to input fields
    input_fields = {}

    manual_scan_widget = QWidget()
    manual_scan_layout = QVBoxLayout()

    # Create input fields
    for key in data_fields:
        row = QHBoxLayout()
        label = QLabel(key)
        line_edit = QLineEdit()
        input_fields[key] = line_edit
        row.addWidget(label)
        row.addWidget(line_edit)
        manual_scan_layout.addLayout(row)

    # Create the send button
    send_first_scan_btn = QPushButton("Send First Scan")

    # Button click handler
    def handle_send_scan():
        scan_data = {}
        for key, line_edit in input_fields.items():
            text = line_edit.text().strip()
            # Attempt to convert to number if possible
            if text == "":
                scan_data[key] = None
            else:
                try:
                    # Float first, then int fallback
                    value = float(text)
                    if value.is_integer():
                        value = int(value)
                    scan_data[key] = value
                except ValueError:
                    scan_data[key] = text  # Leave as string if not a number

        # Save as JSON file
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save Scan Parameters", "", "JSON Files (*.json)"
        )
        if file_path:
            with open(file_path, "w") as f:
                json.dump(scan_data, f, indent=4)
            print(f"Scan parameters saved to: {file_path}")

    send_first_scan_btn.clicked.connect(handle_send_scan)
    manual_scan_layout.addWidget(send_first_scan_btn)

    manual_scan_widget.setLayout(manual_scan_layout)
    return manual_scan_widget


class ZoomableGraphicsView(G.QGraphicsView):
    def __init__(self, scene, hover_label, x_label, y_label, x_micron_label, y_micron_label):
        super().__init__(scene)
        self.union_objects = []
        self.union_dict = {}
        self.current_qimage = None
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self._drag_active = False
        self.hover_label = hover_label
        self.blobs = []
        self.visible_colors = set(element_colors)
        self.x_label = x_label
        self.y_label = y_label
        self.x_micron_label = x_micron_label
        self.y_micron_label = y_micron_label
        self.highlighted_union_indices = []
        self.highlight_items = []
        self.union_blobs = [...]  # Indexable list or dict


    def wheelEvent(self, event):
        cursor_pos = event.pos()
        scene_pos = self.mapToScene(cursor_pos)
        zoom_factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(zoom_factor, zoom_factor)
        mouse_centered = self.mapFromScene(scene_pos)
        delta = cursor_pos - mouse_centered
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._drag_active = True
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)
            self._drag_active = False
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_active:
            super().mouseMoveEvent(event)
            return
        pos = self.mapToScene(event.pos())
        x, y = int(pos.x()), int(pos.y())
        self.x_label.setText(f"X: {x}")
        self.y_label.setText(f"Y: {y}")
        self.x_micron_label.setText(f"X Real location(µm): {(x * microns_per_pixel_x) + true_origin_x:.2f}")
        self.y_micron_label.setText(f"Y Real location(µm): {(y * microns_per_pixel_y) + true_origin_y:.2f}")
        for blob in self.blobs:
            if blob['color'] not in self.visible_colors:
                continue
            cx, cy = blob['center']
            r = blob['radius']
            
            real_w = blob['box_size'] * microns_per_pixel_x
            real_h = blob['box_size'] * microns_per_pixel_y
            real_cx = (cx * microns_per_pixel_x) + true_origin_x
            real_cy = (cy * microns_per_pixel_y) + true_origin_y

            if abs(x - cx) <= r and abs(y - cy) <= r:
                html = (
                    f"{blob['Box']}<br>"
                    f"Center: ({cx}, {cy})<br>"
                    f"Length: {blob['box_size']} px<br>"
                    f"Box area: {blob['box_size'] * blob['box_size']} px²<br>"
                    f"<br>"
                    f"Real Center location(µm): ({(cx * microns_per_pixel_x) + true_origin_x:.2f} µm, {(cy * microns_per_pixel_y) + true_origin_y:.2f} µm)<br>"
                    f"Real box size(µm): ({blob['box_size'] * microns_per_pixel_x:.2f} µm × {blob['box_size'] * microns_per_pixel_y:.2f} µm)<br>"
                    f"Real box area(µm): ({blob['box_size']**2 * microns_per_pixel_x * microns_per_pixel_y:.2f} µm²)<br>"
                    f"Real Top-Left: ({real_cx - real_w / 2:.2f}, {real_cy - real_h / 2:.2f}) µm<br>"
                    f"Real Bottom-Right: ({real_cx + real_w / 2:.2f}, {real_cy + real_h / 2:.2f}) µm<br>"
                    f"<br>"
                    f"Max intensity: {blob['max_intensity']:.3f}<br>"
                    f"Mean intensity: {blob['mean_intensity']:.3f}<br>"
                    f"Mean dilation intensity: {blob['mean_dilation']:.1f}"
                )
                self.hover_label.setText(html)
                self.hover_label.adjustSize()
                self.hover_label.move(event.x() + 15, event.y() - 30)
                self.hover_label.setStyleSheet(
                    f"background-color: {blob['color']}; color: white; border: 1px solid black; padding: 4px;"
                )
                self.hover_label.show()
                return 
                
        if checkboxes['union'].isChecked():  # actual union box visibility check
            for ub in global_union_boxes:
                if ub['rect'].contains(pos.toPoint()):
                    self.hover_label.setText(ub['text'])
                    self.hover_label.adjustSize()
                    self.hover_label.move(event.x() + 15, event.y() - 30)
                    self.hover_label.setStyleSheet(
                        "background-color: white; color: black; border: 1px solid black; padding: 4px;"
                    )
                    self.hover_label.show()
                    return


        
        self.hover_label.hide()

    def update_blobs(self, blobs, visible_colors):
        self.blobs = blobs
        self.visible_colors = visible_colors

    def highlight_selected_union_boxes(self, selected_items):
        # Remove previous highlight circles
        for item in self.highlight_items:
            self.scene().removeItem(item)
        self.highlight_items.clear()
    
        for item in selected_items:
            text = item.toolTip()  # or item.toolTip() if needed
    
            # Extract pixel center from: "Center: (428, 447)"
            center_match = re.search(r"Center: \((\d+), (\d+)\)", text)
    
            # Extract length from: "Length: 18 px"
            length_match = re.search(r"Length: (\d+)\s*px", text)
    
            if center_match and length_match:
                x = int(center_match.group(1))
                y = int(center_match.group(2))
                length = int(length_match.group(1))
                radius = length / 2 + 5  # slightly larger than the box
    
                circle = QGraphicsEllipseItem(x - radius, y - radius, radius * 2, radius * 2)
                circle.setPen(QPen(QColor("yellow"), 2, Qt.DashLine))
                circle.setZValue(100)
                self.scene().addItem(circle)
                self.highlight_items.append(circle)

    
def redraw_boxes(blobs, selected_colors, onto_img=None):
    updated_img = onto_img or graphics_view.current_qimage.copy()
    painter = QPainter(updated_img)
    painter.setRenderHint(QPainter.Antialiasing, False)
    for blob in blobs:
        if blob['color'] in selected_colors:
            cx, cy, r = *blob['center'], blob['radius']
            painter.setPen(QPen(QColor(blob['color']), 1))
            painter.drawRect(cx - r, cy - r, 2 * r, 2 * r)
    painter.end()

    if onto_img is None:
        pixmap_item.setPixmap(QPixmap.fromImage(updated_img))  # if standalone call


def union_box_drawer(union_dict, base_img=None, clear_only=False):
    global global_union_boxes
    global_union_boxes = []  # Clear old ones

    valid_img = base_img or graphics_view.current_qimage

    if clear_only:
        pixmap_item.setPixmap(QPixmap.fromImage(valid_img))
        return              
        
    updated_img = valid_img.copy()
    painter = QPainter(updated_img)
    painter.setRenderHint(QPainter.Antialiasing, False)
    painter.setPen(QPen(QColor('white'), 1))

    for idx, ub in union_dict.items():
        cx, cy = ub['center']
        length = ub['length']
        x = int(cx - length / 2)
        y = int(cy - length / 2)
        w = h = int(length)

        area = ub['area']
        real_cx, real_cy = ub['real_center']
        real_w, real_h = ub['real_size']
        real_area = ub['real_area']
        real_tl = ub['real_top_left']
        real_br = ub['real_bottom_right']

        # Draw box
        painter.drawRect(x, y, w, h)

        # Exact hover text format
        hover_text = (
            f"<b>Union Box #{idx}</b><br>"
            f"Center: ({cx}, {cy})<br>"
            f"Length: {length} px<br>"
            f"Area: {area} px²<br><br>"
            f"Real Center: ({real_cx:.2f} µm, {real_cy:.2f} µm)<br>"
            f"Real Size: {real_w:.2f} × {real_h:.2f} µm<br>"
            f"Real Area: {real_area:.2f} µm²<br><br>"
            f"Real Top-Left: ({real_tl[0]:.2f}, {real_tl[1]:.2f}) µm<br>"
            f"Real Bottom-Right: ({real_br[0]:.2f}, {real_br[1]:.2f}) µm"
        )
        
        # Store for hover lookup
        global_union_boxes.append({
            'rect': QRect(x, y, w, h),
            'text': hover_text
        })

    painter.end()
    pixmap_item.setPixmap(QPixmap.fromImage(updated_img))

def update_boxes():
    # nonlocal selected_colors
    selected_colors = {c for c, cb in checkboxes.items() if cb.isChecked() and c != 'union'}

    # Start from a shared copy of the current base image
    base_img = graphics_view.current_qimage.copy()

    # Get blobs for selected colors
    blobs = [b for b in get_current_blobs() if b['color'] in selected_colors]
    graphics_view.update_blobs(blobs, selected_colors)

    # Draw element boxes on base_img
    redraw_boxes(blobs, selected_colors, onto_img=base_img)

    # Conditionally draw union boxes on the same image
    if union_checkbox.isChecked():
        union_box_drawer(graphics_view.union_dict, base_img=base_img)
    else:
        pixmap_item.setPixmap(QPixmap.fromImage(base_img))  # just show element boxes

    hover_label.hide()

def on_slider_change(value, color):
        snapped = round(value / 10) * 10
        snapped = max(0, min(250, snapped))
        if thresholds[color] != snapped:
            thresholds[color] = snapped
            sliders[color].blockSignals(True)
            sliders[color].setValue(snapped)
            sliders[color].blockSignals(False)
            slider_labels[color].setText(f"{checkboxes[color].text()}_threshold: {snapped}")
            update_boxes()


def on_area_slider_change(value, color):
    valid_areas = list(range(10, 401, 10)) 
    snapped = min(valid_areas, key=lambda a: abs(a - value))
    if area_thresholds[color] != snapped:
        area_thresholds[color] = snapped
        area_sliders[color].blockSignals(True)
        area_sliders[color].setValue(snapped)
        area_sliders[color].blockSignals(False)
        area_slider_labels[color].setText(f"{checkboxes[color].text()}_min_area: {snapped}")
        update_boxes()


    
def add_box():
    # Notify user
    QMessageBox.information(window, "Add Union Box", "Click and drag to define a new union box.")

    temp_state = {'start': None}

    def on_press(event):
        temp_state['start'] = graphics_view.mapToScene(event.pos()).toPoint()

    def on_release(event):
        if temp_state['start'] is None:
            return
    
        end = graphics_view.mapToScene(event.pos()).toPoint()
        start = temp_state['start']
        temp_state['start'] = None
    
        x1, y1 = start.x(), start.y()
        x2, y2 = end.x(), end.y()
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = max(abs(x2 - x1), abs(y2 - y1))
        area = length * length
    
        real_cx = (cx * microns_per_pixel_x) + true_origin_x
        real_cy = (cy * microns_per_pixel_y) + true_origin_y
        real_length_x = length * microns_per_pixel_x
        real_length_y = length * microns_per_pixel_y
        real_area = real_length_x * real_length_y
    
        real_top_left = (
            ((cx - length // 2) * microns_per_pixel_x) + true_origin_x,
            ((cy - length // 2) * microns_per_pixel_y) + true_origin_y
        )
        real_bottom_right = (
            ((cx + length // 2) * microns_per_pixel_x) + true_origin_x,
            ((cy + length // 2) * microns_per_pixel_y) + true_origin_y
        )
    
        new_union = {
            'center': (cx, cy),
            'length': length,
            'area': area,
            'real_center': (real_cx, real_cy),
            'real_size': (real_length_x, real_length_y),
            'real_area': real_area,
            'real_top_left': real_top_left,
            'real_bottom_right': real_bottom_right
        }
    
        # Assign next available index in the union dict
        current_union_dict = graphics_view.union_dict
        next_index = max(current_union_dict.keys(), default=0) + 1
        current_union_dict[next_index] = new_union
    
        # Update both object list and dictionary
        graphics_view.union_objects = list(current_union_dict.values())
        graphics_view.union_dict = current_union_dict

        # Add new item to QListWidget with consistent style/tooltip
        ub = new_union  # The most recently added one
        idx = next_index
        cx, cy = ub['center']
        length = ub['length']
        area = ub['area']
        real_cx, real_cy = ub['real_center']
        real_w, real_h = ub['real_size']
        real_area = ub['real_area']
        real_tl = ub['real_top_left']
        real_br = ub['real_bottom_right']

        global custom_box_number

        item_text = f"Custom Box #{custom_box_number}"

        
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, custom_box_number)
        
        tooltip_text = (
            f"<b>Custom Box #{custom_box_number}</b><br>"
            f"Center: ({cx}, {cy})<br>"
            f"Length: {length} px<br>"
            f"Area: {area} px²<br><br>"
            f"Real Center: ({real_cx:.2f} µm, {real_cy:.2f} µm)<br>"
            f"Real Size: {real_w:.2f} × {real_h:.2f} µm<br>"
            f"Real Area: {real_area:.2f} µm²<br><br>"
            f"Real Top-Left: ({real_tl[0]:.2f}, {real_tl[1]:.2f}) µm<br>"
            f"Real Bottom-Right: ({real_br[0]:.2f}, {real_br[1]:.2f}) µm"
        )
        item.setToolTip(tooltip_text)

        custom_box_number = custom_box_number + 1
        
        union_list_widget.addItem(item)

        output_path = os.path.join(selected_directory, "union_blobs.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(graphics_view.union_dict, f)
            
        # Restore original mouse events and update display
        graphics_view.mousePressEvent = original_mouse_press
        graphics_view.mouseReleaseEvent = original_mouse_release
        update_boxes()

    # Temporarily override mouse events
    original_mouse_press = graphics_view.mousePressEvent
    original_mouse_release = graphics_view.mouseReleaseEvent
    graphics_view.mousePressEvent = on_press
    graphics_view.mouseReleaseEvent = on_release

def on_union_item_selected():
    selected_items = union_list_widget.selectedItems()
    graphics_view.highlight_selected_union_boxes(selected_items)


def send_to_list():
    # Collect all existing item texts in queue_server_list
    existing_texts = {queue_server_list.item(i).text() for i in range(queue_server_list.count())}

    for item in union_list_widget.selectedItems():
        if item.text() not in existing_texts:
            new_item = QListWidgetItem(item.text())
            new_item.setToolTip(item.toolTip())
            new_item.setData(Qt.UserRole, item.data(Qt.UserRole))  # preserve index if needed
            queue_server_list.addItem(new_item)

def send_to_queue_server():
    data = {}
    for i in range(queue_server_list.count()):
        item = queue_server_list.item(i)
        text = item.text()
        tooltip = item.toolTip()
        index = item.data(Qt.UserRole)

        # Extract "Union Box #X" part as the key
        key = text.split("|")[0].strip()  # e.g., "Union Box #1"

        data[key] = {
            "text": text,
            "tooltip": tooltip,
            "index": index  # optional
        }

    json_safe_data = make_json_serializable(data)

    output_path = os.path.join(selected_directory, "selected_blobs_to_queue_server.json")
    with open(output_path, "w") as f:
        json.dump(json_safe_data, f, indent=2)

    structure_blob_tooltips(output_path)
    send_json_boxes_to_queue_with_center_move(output_path)

    queue_server_list.clear()
    queue_server_list.addItem("✅ Data sent and saved")

def get_elements_list():    
    # Collect all visible blobs across elements
    union_list_widget.clear()
    all_blobs = get_current_blobs()
    for blob in all_blobs:
        cx, cy = blob['center']
        box_size = blob['box_size']

        real_cx = (cx * microns_per_pixel_x) + true_origin_x
        real_cy = (cy * microns_per_pixel_y) + true_origin_y
        real_length_x = box_size * microns_per_pixel_x
        real_length_y = box_size * microns_per_pixel_y
        real_area = real_length_x * real_length_y

        # HTML tooltip
        html = (
            f"{blob['Box']}<br>"
            f"Center: ({cx}, {cy})<br>"
            f"Length: {box_size} px<br>"
            f"Box area: {box_size * box_size} px²<br>"
            f"<br>"
            f"Real Center location (µm): ({real_cx:.2f} µm, {real_cy:.2f} µm)<br>"
            f"Real box size (µm): ({real_length_x:.2f} µm × {real_length_y:.2f} µm)<br>"
            f"Real box area (µm²): ({real_area:.2f} µm²)<br>"
            f"Real Top-Left: ({real_cx - real_length_x / 2:.2f}, {real_cy - real_length_y / 2:.2f}) µm<br>"
            f"Real Bottom-Right: ({real_cx + real_length_x / 2:.2f}, {real_cy + real_length_y / 2:.2f}) µm<br>"
            f"<br>"
            f"Max intensity: {blob['max_intensity']:.3f}<br>"
            f"Mean intensity: {blob['mean_intensity']:.3f}<br>"
            f"Mean dilation intensity: {blob['mean_dilation']:.1f}"
        )

        # List item
        title = blob['Box']
        item = QListWidgetItem(title)
        item.setData(Qt.UserRole, blob)  # Store full blob data if needed later
        item.setData(Qt.ToolTipRole, html)  # So we can show it in a custom hover

        union_list_widget.addItem(item)

def clear_queue_server_list():    
    queue_server_list.clear() 
