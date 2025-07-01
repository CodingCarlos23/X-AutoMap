
 
import json
 
def send_json_boxes_to_queue_with_center_move(json_file_path, dets="dets1", x_motor="zpssx", y_motor="zpssy", exp_t=0.01, px_per_um=1.25):
    """
    For each region in the JSON file:
    - Move stage to real_center_um
    - Perform fly2d scan centered on that position
    """
    with open(json_file_path, "r") as f:
        boxes = json.load(f)
 
    for label, info in boxes.items():
        cx, cy = info["real_center_um"]         # center in um
        sx, sy = info["real_size_um"]           # size in um
        num_x = int(sx * px_per_um)
        num_y = int(sy * px_per_um)
 
        # Define relative scan range around center
        x_start = -sx / 2
        x_end = sx / 2
        y_start = -sy / 2
        y_end = sy / 2
 
        # Detector names
        det_names = [d.name for d in eval(dets)]
 
        # Create ROI dictionary to move motors first
        roi = {x_motor: cx, y_motor: cy}
 
        RM.item_add(BPlan(
            "recover_pos_and_scan",
            label,
            roi,
            det_names,
            x_motor,
            x_start,
            x_end,
            num_x,
            y_motor,
            y_start,
            y_end,
            num_y,
            exp_t
        ))
        print(f"Queued: {label} | center ({cx:.1f}, {cy:.1f}) µm | size ({sx:.1f}, {sy:.1f}) µm")
