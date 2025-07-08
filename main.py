from globals import *
import gui
import utils

img_paths = [None, None, None]  # [R, G, B]
file_names = [None, None, None]
element_colors = ["red", "green", "blue"]

graphics_view = None
controls_widget = None

microns_per_pixel_x = 0.0
microns_per_pixel_y = 0.0

true_origin_x = 0.0
true_origin_y = 0.0

selected_directory = None

selected_files_order = []


app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("Multi-Element Image Analyzer")
window.resize(600, 500)


main_layout = QVBoxLayout()

# Directory selector
dir_button = QPushButton("Choose Directory")
dir_button.clicked.connect(on_dir_selected)
main_layout.addWidget(dir_button)

# List of files in the selected directory
file_list_widget = QListWidget()
file_list_widget.setSelectionMode(QListWidget.NoSelection)
file_list_widget.itemChanged.connect(update_selection)
# Horizontal layout: file list on the left, confirm button on the right
file_confirm_layout = QHBoxLayout()
file_confirm_layout.addWidget(file_list_widget)

# Right side (confirm button)
right_panel = QVBoxLayout()
confirm_button = QPushButton("Confirm Selection")
confirm_button.clicked.connect(on_confirm_clicked)
right_panel.addWidget(confirm_button)
right_panel.addStretch()

float_input_micron_x  = QDoubleSpinBox()
float_input_micron_x.setPrefix("Enter X(µm) per pixel:")
float_input_micron_x.setRange(0.0, 1000.0)         # Adjust range as needed
float_input_micron_x.setSingleStep(0.1)            # Step size
float_input_micron_x.setDecimals(3)                # Number of decimal places
float_input_micron_x.setValue(1.0)                 # Default value
right_panel.addWidget(float_input_micron_x)

float_input_micron_y  = QDoubleSpinBox()
float_input_micron_y.setPrefix("Enter Y(µm) per pixel:")
float_input_micron_y.setRange(0.0, 1000.0)         # Adjust range as needed
float_input_micron_y.setSingleStep(0.1)            # Step size
float_input_micron_y.setDecimals(3)                # Number of decimal places
float_input_micron_y.setValue(1.0)                 # Default value
right_panel.addWidget(float_input_micron_y)

origin_x_input = QDoubleSpinBox()
origin_x_input.setPrefix("Origin X(µm): ")
origin_x_input.setRange(-1e6, 1e6)
origin_x_input.setDecimals(2)
origin_x_input.setValue(0.0)                 # Default value

origin_y_input = QDoubleSpinBox()
origin_y_input.setPrefix("Origin Y(µm): ")
origin_y_input.setRange(-1e6, 1e6)
origin_y_input.setDecimals(2)
origin_x_input.setValue(0.0)                 # Default value

right_panel.addWidget(origin_x_input)
right_panel.addWidget(origin_y_input)

file_confirm_layout.addLayout(right_panel)

main_layout.addLayout(file_confirm_layout)

# Track full file paths
file_paths = []

# Global list to store hoverable union boxes
global_union_boxes = []

# Global Variable for current blob detection 
current_iteration = 0

#Custom Box Counter
custom_box_number = 1











def init_gui():
    global current_iteration
    current_iteration = 0
    global precomputed_blobs
    precomputed_blobs = {}
    
    # Remove old progress bars if they exist
    for i in reversed(range(main_layout.count())):
        widget = main_layout.itemAt(i).widget()
        if isinstance(widget, QProgressBar):
            main_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()

    # for btn in buttons:
    #     btn.setEnabled(False)

    # Load and convert
    global graphics_view, controls_widget

    if graphics_view is not None:
        main_layout.removeWidget(graphics_view)
        graphics_view.setParent(None)
        graphics_view.deleteLater()
        graphics_view = None

    if controls_widget is not None:
        main_layout.removeWidget(controls_widget)
        controls_widget.setParent(None)
        controls_widget.deleteLater()
        controls_widget = None
        
    img_r, img_g, img_b = [tiff.imread(p).astype(np.float32) for p in img_paths]
    
    # Resize to majority shape
    shapes = [img_r.shape, img_g.shape, img_b.shape]
    shape_counts = Counter(shapes)
    target_shape = shape_counts.most_common(1)[0][0]
    # print(f"Target (majority) shape: {target_shape}")
    
    img_r = resize_if_needed(img_r, file_names[0])
    img_g = resize_if_needed(img_g, file_names[1])
    img_b = resize_if_needed(img_b, file_names[2])

    norm_dilated = [normalize_and_dilate(im) for im in [img_r, img_g, img_b]]
    normalized = [nd[0] for nd in norm_dilated]
    dilated = [nd[1] for nd in norm_dilated]

    thresholds_range = list(range(0, 256, 10))
    area_range = list(range(10, 501, 10))

    QApplication.processEvents()
    progress_bar = QProgressBar()
    progress_bar.setRange(0, len(element_colors) * len(thresholds_range) * len(area_range))
    progress_bar.setValue(0)
    progress_bar.setTextVisible(True)
    progress_bar.setFormat("Computing blobs... %p%")
    main_layout.addWidget(progress_bar)
    QApplication.processEvents() 
    progress_bar.show()
    QApplication.processEvents()
    total_iterations = len(element_colors) * len(thresholds_range) * len(area_range)
    precomputed_blobs = {}
    
    QTimer.singleShot(0, lambda: start_blob_computation(
        element_colors,
        thresholds_range,
        area_range,
        precomputed_blobs,
        dilated,
        img_r,
        img_g,
        img_b,
        file_names,
        progress_bar
    )) 
 
    os.makedirs("data", exist_ok=True)

    QApplication.processEvents()

    # Make a deep copy so original data stays untouched
    blobs_to_save = copy.deepcopy(precomputed_blobs)    
    # Add real-world info and tooltip HTML to copied blobs
    for color in blobs_to_save:
        for key in blobs_to_save[color]:
            for blob in blobs_to_save[color][key]:
                cx, cy = blob['center']
    
                # Calculate real-world values
                real_center_x = (cx * microns_per_pixel_x) + true_origin_x
                real_center_y = (cy * microns_per_pixel_y) + true_origin_y
                real_box_size_x = blob['box_size'] * microns_per_pixel_x
                real_box_size_y = blob['box_size'] * microns_per_pixel_y
                real_box_area = (blob['box_size'] ** 2) * microns_per_pixel_x * microns_per_pixel_y
    
                # Store real-world values
                blob['real_center'] = (real_center_x, real_center_y)
                blob['real_box_size'] = (real_box_size_x, real_box_size_y)
                blob['real_box_area'] = real_box_area

                real_w = blob['box_size'] * microns_per_pixel_x
                real_h = blob['box_size'] * microns_per_pixel_y
                real_cx = (cx * microns_per_pixel_x) + true_origin_x
                real_cy = (cy * microns_per_pixel_y) + true_origin_y
                real_tl_x = real_cx - real_w / 2
                real_tl_y = real_cy - real_h / 2
                real_br_x = real_cx + real_w / 2
                real_br_y = real_cy + real_h / 2

                real_w = blob['box_size'] * microns_per_pixel_x
                real_h = blob['box_size'] * microns_per_pixel_y
                real_cx = (cx * microns_per_pixel_x) + true_origin_x
                real_cy = (cy * microns_per_pixel_y) + true_origin_y
                
                # Compose tooltip HTML
                blob['tooltip_html'] = (
                    f"{blob['Box']}<br>"
                    f"Center: ({cx}, {cy})<br>"
                    f"Length: {blob['box_size']} px<br>"
                    f"Box area: {blob['box_size'] * blob['box_size']} px²<br>"
                    f"<br>"
                    f"Real Center location(µm): ({real_center_x:.2f} µm, {real_center_y:.2f} µm)<br>"
                    f"Real box size(µm): ({real_box_size_x:.2f} µm × {real_box_size_y:.2f} µm)<br>"
                    f"Real box area(µm²): {real_box_area:.2f} µm²<br>"
                    f"Real Top-Left: ({real_cx - real_w / 2:.2f}, {real_cy - real_h / 2:.2f}) µm<br>"
                    f"Real Bottom-Right: ({real_cx + real_w / 2:.2f}, {real_cy + real_h / 2:.2f}) µm<br>"
                    f"<br>"
                    f"Max intensity: {blob['max_intensity']:.3f}<br>"
                    f"Mean intensity: {blob['mean_intensity']:.3f}<br>"
                    f"Mean dilation intensity: {blob['mean_dilation']:.1f}"
                )
                
    # Save the augmented copy to disk
    output_path = os.path.join(selected_directory, "precomputed_blobs_with_real_info.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(blobs_to_save, f) 
        
    progress_bar.hide()
    QApplication.processEvents()

    # print(time.strftime('%H:%M:%S'))
    thresholds = {color: 100 for color in element_colors}
    area_thresholds = {color: 200 for color in element_colors}
    
    merged_rgb = cv2.merge([
        cv2.normalize(img_r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.normalize(img_g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.normalize(img_b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ])

    hover_label = QLabel()
    hover_label.setWindowFlags(Qt.ToolTip)
    hover_label.hide()

    x_label = QLabel("X: 0")
    y_label = QLabel("Y: 0")

    x_micron_label = QLabel("X Real: 0")
    y_micron_label = QLabel("Y Real: 0")
    
    scene = QGraphicsScene()
    q_img = QImage(merged_rgb.data, merged_rgb.shape[1], merged_rgb.shape[0], merged_rgb.shape[1] * 3, QImage.Format_RGB888)

    graphics_view = ZoomableGraphicsView(scene, hover_label, x_label, y_label, x_micron_label, y_micron_label)
    graphics_view.current_qimage = q_img
    pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(graphics_view.current_qimage))
    scene.addItem(pixmap_item)
    main_layout.addWidget(graphics_view)

    checkboxes = {}
    selected_colors = set(element_colors)
 
    legend_layout = QVBoxLayout()
    legend_label = QLabel("Legend")
    legend_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
    legend_layout.addWidget(legend_label)
    for i, color in enumerate(element_colors):
        cb = QCheckBox(file_names[i])
        cb.setChecked(True)
        cb.setStyleSheet(f"color: {color}")
        cb.stateChanged.connect(update_boxes)
        checkboxes[color] = cb
        legend_layout.addWidget(cb)

    union_checkbox = QCheckBox("Union Boxes")
    union_checkbox.setChecked(True)
    union_checkbox.setStyleSheet("color: black")
    union_checkbox.stateChanged.connect(update_boxes)
    legend_layout.addWidget(union_checkbox)
    checkboxes['union'] = union_checkbox


    legend_layout.addStretch()

    sliders = {}
    slider_labels = {}
    slider_layout = QHBoxLayout()
        
    for color in element_colors:
        i = element_colors.index(color)
        vbox = QVBoxLayout()
        label = QLabel(f"{file_names[i]}_threshold: {thresholds[color]}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(255)
        slider.setTickInterval(10)
        slider.setValue(thresholds[color])
        slider.setTickPosition(QSlider.TicksBelow)
        slider.valueChanged.connect(lambda val, c=color: on_slider_change(val, c))
        sliders[color] = slider
        slider_labels[color] = label
        vbox.addWidget(label)
        vbox.addWidget(slider)
        slider_layout.addLayout(vbox)

    area_sliders = {}
    area_slider_labels = {}

    area_slider_layout = QHBoxLayout()
    
    for color in element_colors:
        i = element_colors.index(color)
        vbox = QVBoxLayout()
        label = QLabel(f"{file_names[i]}_min_area: {area_thresholds[color]}")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(10)
        slider.setMaximum(400)
        slider.setTickInterval(10)
        slider.setValue(area_thresholds[color])
        slider.setTickPosition(QSlider.TicksBelow)
        slider.valueChanged.connect(lambda val, c=color: on_area_slider_change(val, c))
        area_sliders[color] = slider
        area_slider_labels[color] = label
        vbox.addWidget(label)
        vbox.addWidget(slider)
        area_slider_layout.addLayout(vbox)

    exit_btn = QPushButton("Exit")
    exit_btn.clicked.connect(lambda: (window.close(), app.quit()))
    reset_btn = QPushButton("Reset View")
    reset_btn.clicked.connect(lambda: graphics_view.resetTransform())
        
    add_btn = QPushButton("Add custom box")
    add_btn.clicked.connect(add_box)
    
    union_btn = QPushButton("Get current unions")
    union_btn.clicked.connect(union_function)

    union_list_widget = QListWidget()
    union_list_widget.setSelectionMode(QListWidget.MultiSelection)  # Allow multiple selection
    union_list_widget.setMinimumHeight(200)
    union_list_widget.itemSelectionChanged.connect(on_union_item_selected)
    union_list_widget.setLineWidth(1)
    union_list_widget.setStyleSheet("""
        QListWidget {
            border: 1px solid #aaa;
            border-radius: 4px;
            background-color: #fdfdfd;
        }
        QListWidget::item {
            padding: 8px;
            border-bottom: 1px solid #ccc;  /* ← This creates the line */
        }
        QListWidget::item:selected {
            background-color: #007acc;
            color: white;
            font-weight: bold;
            border: 1px solid #005b9f;
        }
        QListWidget::item:hover {
            background-color: #e0f0ff;
        }
    """)
    union_list_widget.setMouseTracking(True)  # Required for hover without clicking
    union_list_widget.setEnabled(True)

    queue_server_list = QListWidget()
    queue_server_list.setMinimumHeight(200)
    queue_server_list.setLineWidth(1)

    send_to_list_btn = QPushButton("Add to Queue server List")
    send_to_list_btn.clicked.connect(send_to_list)

    send_elements_to_list_btn = QPushButton("Add all individual element scans to list")
    send_elements_to_list_btn.clicked.connect(get_elements_list)

    send_to_queue_server_btn = QPushButton("Send to Queue server List")    
    send_to_queue_server_btn.clicked.connect(send_to_queue_server)

    clear_queue_server_btn = QPushButton("Clear")    
    clear_queue_server_btn.clicked.connect(clear_queue_server_list)

    union_list_layout = QVBoxLayout()
    union_list_layout.addWidget(union_list_widget)
    union_list_layout.addWidget(send_to_list_btn)
    union_list_layout.addWidget(send_elements_to_list_btn)
    union_list_layout.addWidget(union_btn)
    union_list_layout.addWidget(add_btn)

    queue_server_list_layout = QVBoxLayout()
    queue_server_list_layout.addWidget(queue_server_list)
    queue_server_list_layout.addWidget(send_to_queue_server_btn)
    queue_server_list_layout.addWidget(clear_queue_server_btn)

    dual_list_layout = QHBoxLayout()
    dual_list_layout.addLayout(union_list_layout)
    dual_list_layout.addLayout(queue_server_list_layout)

    controls = QVBoxLayout()
    controls.addWidget(exit_btn)
    controls.addWidget(reset_btn)
    
    controls.addLayout(dual_list_layout)
    # controls.addWidget(union_list_widget)
    controls.addWidget(send_to_list_btn)
    controls.addWidget(send_to_queue_server_btn)
    controls.addLayout(legend_layout)
    controls.addLayout(slider_layout)
    controls.addLayout(area_slider_layout)
    controls.addWidget(x_label)
    controls.addWidget(y_label)
    controls.addWidget(x_micron_label)
    controls.addWidget(y_micron_label)

    layout = QHBoxLayout()
    layout.addWidget(graphics_view)
    side_panel = QWidget()
    side_panel.setLayout(controls)
    controls_widget = side_panel
    layout.addWidget(side_panel)

    main_layout.addLayout(layout)
    hover_label.setParent(window)

    blobs = get_current_blobs()
    graphics_view.update_blobs(blobs, selected_colors)
    redraw_boxes(blobs, selected_colors)
    window.resize(1900, 1000)

layout_wrapper = QWidget()
layout_wrapper.setLayout(main_layout)
tabs = QTabWidget()
tabs.addTab(layout_wrapper, "Home")

tabs.addTab(create_manual_scan_tab(), "Manual Scan")

screen = QVBoxLayout()
screen.addWidget(tabs)

window.setLayout(screen)
window.show()
sys.exit(app.exec_())