#!/usr/bin/env python3
"""
merge_and_stitch_highres.py
Combines Ca_K, Fe_K, and Cu_K TIFFs into RGBs and stitches them into a
high-resolution 8x8 mosaic with clearly readable overlay labels.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

# === CONFIG ===
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

# === HIGH-RES STITCH FUNCTION ===
def stitch_images(scan_ids, grid_size=(8, 8), draw_tile_borders=False,
                  column_shift_amount=0, draw_union_boxes=False,
                  show_box_labels=False, upscale_factor=2):
    """
    Stitch all merged images into a mosaic with optional overlays.
    upscale_factor increases resolution for clearer labels.
    """
    print("\n🧩 Stitching high-resolution 8x8 mosaic...")

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

    mosaic_w = int((cols * tile_w - (cols - 1) * column_shift_amount) * upscale_factor)
    mosaic_h = int(rows * tile_h * upscale_factor)
    mosaic = Image.new("RGB", (mosaic_w, mosaic_h))
    draw = ImageDraw.Draw(mosaic)

    # scale factor for coordinates
    scale = upscale_factor

    # larger, clearer font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", int(3 * upscale_factor))
    except:
        font = ImageFont.load_default()

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(tiles):
                x_tile = int((j * tile_w - j * column_shift_amount) * scale)
                y_tile = int(i * tile_h * scale)

                tile_img = tiles[idx].resize((int(tile_w * scale), int(tile_h * scale)), Image.LANCZOS)
                mosaic.paste(tile_img, (x_tile, y_tile))

                if draw_tile_borders:
                    draw.rectangle([x_tile, y_tile,
                                    x_tile + tile_w * scale - 1,
                                    y_tile + tile_h * scale - 1],
                                   outline="white", width=int(1 * scale))

                if draw_union_boxes:
                    current_scan_id = scan_ids[idx]
                    union_json_path = os.path.join(info_path, f"automap_{current_scan_id}", "unions_output.json")

                    if os.path.exists(union_json_path):
                        try:
                            with open(union_json_path, "r") as f:
                                union_data = json.load(f)

                            for box_info in union_data.values():
                                cx_img = box_info["image_center"][0] * scale
                                cy_img = box_info["image_center"][1] * scale
                                L_img = box_info["image_length"] * scale

                                x_start_mosaic = x_tile + cx_img - L_img / 2
                                y_start_mosaic = y_tile + cy_img - L_img / 2
                                x_end_mosaic = x_tile + cx_img + L_img / 2
                                y_end_mosaic = y_tile + cy_img + L_img / 2

                                draw.rectangle(
                                    [x_start_mosaic, y_start_mosaic,
                                     x_end_mosaic - 1, y_end_mosaic - 1],
                                    outline="white", width=int(1 * scale)
                                )

                                if show_box_labels:
                                    label_text = f"({box_info['image_center'][0]:.1f}, {box_info['image_center'][1]:.1f}) | {current_scan_id}"
                                    text_x = x_start_mosaic + 10 * scale
                                    text_y = y_end_mosaic - 20 * scale
                                    draw.text((text_x, text_y), label_text, fill="white", font=font)

                        except Exception as e:
                            print(f"⚠️ Error loading or processing {union_json_path}: {e}")
                    else:
                        print(f"⚠️ unions_output.json not found for scan ID {current_scan_id}")

    mosaic_path = os.path.join(save_dir, f"stitched_8x8_mosaic_highres_x{upscale_factor}.png")
    mosaic.save(mosaic_path, quality=95)
    print(f"✅ High-resolution mosaic saved to: {mosaic_path}")

# === STITCH UNION TIFFS ===
def stitch_union_tiffs(scan_ids, grid_size=(8, 8), base_path=None):
    print("\n🧩 Stitching 8x8 union TIFF mosaic...")
    if base_path is None:
        print("Error: base_path for union TIFFs is not provided.")
        return

    tiles = []
    tile_size = None

    for sid in scan_ids:
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
                tile_size = (512, 512)
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
    # Merge all images
    for folder in scan_folders:
        merge_folder(folder)

    # Create high-resolution stitched mosaic
    stitch_images(
        scan_ids,
        draw_tile_borders=True,
        column_shift_amount=5,
        draw_union_boxes=True,
        show_box_labels=True,
        upscale_factor=3  # <-- increase to 3 or 4 for sharper text
    )

    print("\n🎉 All merging and stitching complete!")
    print("\n--- Starting sub-section: Union TIFF Stitching ---")
    stitch_union_tiffs(scan_ids, base_path=info_path)
    print("--- Sub-section complete ---")
