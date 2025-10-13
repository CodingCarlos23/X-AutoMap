#!/usr/bin/env python3
"""
merge_and_stitch.py
1. Combine Ca_K, Fe_K, and Cu_K TIFF images into RGB composites
   for multiple scan folders.
2. Stitch all merged images into an 8x8 mosaic in numerical order.
"""

import os
import numpy as np
from PIL import Image

# === CONFIG ===
base_root = "/home/codingcarlos/Desktop/Data/Carlos-D/Carlos-D/Automap_2025Q3/xrf_data_mosaic_auto_PM_200um"
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
def stitch_images(scan_ids, grid_size=(8, 8)):
    print("\n🧩 Stitching 8x8 mosaic...")
    from PIL import Image

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

    mosaic_path = os.path.join(save_dir, "stitched_8x8_mosaic.png")
    mosaic.save(mosaic_path)
    print(f"✅ 8x8 mosaic saved to: {mosaic_path}")

# === MAIN ===
if __name__ == "__main__":
    for folder in scan_folders:
        merge_folder(folder)
    stitch_images(scan_ids)
    print("\n🎉 All merging and stitching complete!")
