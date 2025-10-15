#!/usr/bin/env python3
"""
merge_only.py
Combine Ca_K, Fe_K, and Cu_K TIFF images into RGB composites
for multiple scan folders and save them into range-based subfolders in FineImages.
"""

import os
import numpy as np
from PIL import Image

# === CONFIG ===
base_root = "/home/codingcarlos/Desktop/Data/Carlos-D/Carlos-D/Automap_2025Q3/xrf_data_mosaic_auto_PM_200um"
save_root = "/home/codingcarlos/Desktop/Data/FineImages"
os.makedirs(save_root, exist_ok=True)

# === DEFINE SCAN RANGES ===
scan_ranges = [
    (367582, 367589),
    (367589, 367592),
    (367592, 367596),
    (367596, 367600),
]

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

# === MERGE FUNCTION ===
def merge_folder(scan_folder, save_dir):
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

# === MAIN LOOP ===
if __name__ == "__main__":
    for start, end in scan_ranges:
        range_name = f"{start}-{end}"
        save_dir = os.path.join(save_root, range_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n📁 Processing range: {range_name}")
        scan_ids = list(range(start, end))
        scan_folders = [f"output_tiff_scan2D_{sid}" for sid in scan_ids]

        for folder in scan_folders:
            merge_folder(folder, save_dir)

    print("\n🎉 All ranges processed and saved into FineImages subfolders!")
