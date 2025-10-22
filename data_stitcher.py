#!/usr/bin/env python3
"""
merge_and_stitch.py
1. Combine Ca_K, Fe_K, and Cu_K TIFF images into RGB composites
   for multiple scan folders.
2. Stitch all merged images into an 8x8 mosaic in numerical order.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import json

# === CONFIG ===
base_root = "/home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3/xrf_data_mosaic_auto_PM_200um"
base_root = "/home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3/all_xrf"
info_path = "/home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3/data/user_macros"
save_dir = "/home/codingcarlos/Desktop/Data/MergeImages"
os.makedirs(save_dir, exist_ok=True)

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

scan_folders = [f"output_tiff_scan2D_{sid}" for sid in scan_ids]

# === FILE ELEMENTS ===
elements = {
    "G": "detsum_Ca_K_norm.tiff",
    "B": "detsum_Fe_K_norm.tiff",
    "R": "detsum_Cu_K_norm.tiff"
}

# === NORMALIZATION ===
def normalize(arr):
    arr = arr - np.min(arr)
    if np.max(arr) > 0:
        arr = arr / np.max(arr)
    return (arr * 255).astype(np.uint8)

# === MERGE EACH FOLDER ===
def merge_folder(scan_folder):
    base_dir = os.path.join(base_root, scan_folder)
    print(f"\n🔹 Processing: {scan_folder}")
    channels = {}

    for channel, filename in elements.items():
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            print(f"⚠️ Missing file: {path}")
            return False
        img = Image.open(path).convert("F")
        channels[channel] = np.array(img)

    R = normalize(channels["R"])
    G = normalize(channels["G"])
    B = normalize(channels["B"])

    rgb = np.stack([R, G, B], axis=-1)
    rgb_img = Image.fromarray(rgb, mode="RGB")

    output_path = os.path.join(save_dir, f"merged_{scan_folder}.png")
    rgb_img.save(output_path)
    print(f"✅ Saved merged image: {output_path}")
    return True

# === CREATE MOSAIC ===
def stitch_images(scan_ids, grid_size=(8, 8), draw_tile_borders=False, column_shift_amount=0, draw_union_boxes=False):
    print("\n🧩 Stitching 8x8 mosaic...")
    
    merged_files = [f"merged_output_tiff_scan2D_{sid}.png" for sid in scan_ids]
    tiles = []
    tile_size = None

    for filename in merged_files:
        path = os.path.join(save_dir, filename)
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
            if tile_size is None:
                tile_size = img.size
            else:
                img = img.resize(tile_size)
            tiles.append(img)
        else:
            if tile_size is None:
                tile_size = (512, 512)
            black = Image.fromarray(np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8))
            tiles.append(black)

    if not tiles:
        print("No images to stitch.")
        return

    rows, cols = grid_size
    tile_w, tile_h = tile_size

    # Calculate mosaic width considering column shifts
    mosaic_w = cols * tile_w - (cols - 1) * column_shift_amount
    mosaic_h = rows * tile_h

    mosaic = Image.new("RGB", (mosaic_w, mosaic_h))

    draw = ImageDraw.Draw(mosaic)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(tiles):
                # Calculate x position with cumulative shift
                x_tile_on_mosaic = j * tile_w - j * column_shift_amount
                y_tile_on_mosaic = i * tile_h
                mosaic.paste(tiles[idx], (x_tile_on_mosaic, y_tile_on_mosaic))

                if draw_tile_borders:
                    draw.rectangle([x_tile_on_mosaic, y_tile_on_mosaic, x_tile_on_mosaic + tile_w - 1, y_tile_on_mosaic + tile_h - 1], outline="white", width=1)

                if draw_union_boxes:
                    current_scan_id = scan_ids[idx]
                    union_json_path = os.path.join(info_path, f"automap_{current_scan_id}", "unions_output.json")
                    
                    if os.path.exists(union_json_path):
                        try:
                            with open(union_json_path, "r") as f:
                                union_data = json.load(f)
                            
                            for box_key, box_info in union_data.items():
                                cx_img = box_info["image_center"][0]
                                cy_img = box_info["image_center"][1]
                                L_img = box_info["image_length"]

                                x_start_on_tile = cx_img - L_img / 2
                                y_start_on_tile = cy_img - L_img / 2
                                x_end_on_tile = cx_img + L_img / 2
                                y_end_on_tile = cy_img + L_img / 2

                                x_start_mosaic = x_tile_on_mosaic + x_start_on_tile
                                y_start_mosaic = y_tile_on_mosaic + y_start_on_tile
                                x_end_mosaic = x_tile_on_mosaic + x_end_on_tile
                                y_end_mosaic = y_tile_on_mosaic + y_end_on_tile
                                
                                draw.rectangle([x_start_mosaic, y_start_mosaic, x_end_mosaic - 1, y_end_mosaic - 1], outline="white", width=1)
                        except Exception as e:
                            print(f"⚠️ Error loading or processing {union_json_path}: {e}")
                    else:
                        print(f"⚠️ unions_output.json not found for scan ID {current_scan_id} at {union_json_path}")

    mosaic_path = os.path.join(save_dir, "stitched_8x8_mosaic.png")
    mosaic.save(mosaic_path)
    print(f"✅ 8x8 mosaic saved to: {mosaic_path}")

def stitch_union_tiffs(scan_ids, grid_size=(8, 8), base_path=None):
    print("\n🧩 Stitching 8x8 union TIFF mosaic...")
    from PIL import Image

    if base_path is None:
        print("Error: base_path for union TIFFs is not provided.")
        return

    tiles = []
    tile_size = None

    for sid in scan_ids:
        # Assuming the structure is base_path/automap_sid/xrt/union_elements.tiff
        tiff_path = os.path.join(base_path, f"automap_{sid}", "Union of elements.tiff")
        
        if os.path.exists(tiff_path):
            img = Image.open(tiff_path).convert("RGB")
            if tile_size is None:
                tile_size = img.size
            else:
                img = img.resize(tile_size)
            tiles.append(img)
        else:
            print(f"⚠️ Missing union TIFF: {tiff_path}")
            if tile_size is None:
                tile_size = (512, 512) # Default size if first tile is missing
            black = Image.fromarray(np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8))
            tiles.append(black)

    if not tiles:
        print("No union TIFFs to stitch.")
        return

    rows, cols = grid_size
    tile_w, tile_h = tile_size
    mosaic = Image.new("RGB", (cols * tile_w, rows * tile_h))

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(tiles):
                x = j * tile_w
                y = i * tile_h
                mosaic.paste(tiles[idx], (x, y))

    mosaic_path = os.path.join(save_dir, "stitched_8x8_Union_tiff(old).tiff")
    mosaic.save(mosaic_path)
    print(f"✅ 8x8 union TIFF mosaic saved to: {mosaic_path}")

# === MAIN ===
if __name__ == "__main__":
    for folder in scan_folders:
        merge_folder(folder)
    draw_tile_borders_param = False
    column_shift_amount_param = 5
    draw_union_boxes_param = True
    stitch_images(scan_ids, draw_tile_borders=draw_tile_borders_param, column_shift_amount=column_shift_amount_param, draw_union_boxes=draw_union_boxes_param)
    print("\n🎉 All merging and stitching complete!")

    # === SUB === # New section
    print("\n--- Starting sub-section: Union TIFF Stitching ---")
    stitch_union_tiffs(scan_ids, base_path=info_path)
    print("--- Sub-section complete ---")