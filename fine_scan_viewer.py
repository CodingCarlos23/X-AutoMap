#!/usr/bin/env python3
"""
merge_only.py
Combine Ca_K, Fe_K, and Cu_K TIFF images into RGB composites
for multiple scan folders and save to FineImages.
"""

import os
import numpy as np
from PIL import Image

# === CONFIG ===
base_root = "/home/codingcarlos/Desktop/Data/Carlos-D/Carlos-D/Automap_2025Q3/xrf_data_mosaic_auto_PM_200um"
save_dir = "/home/codingcarlos/Desktop/Data/FineImages"
os.makedirs(save_dir, exist_ok=True)

# scan IDs from 367582 to 367589 inclusive
scan_ids = list(range(367582, 367590))
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

# === MAIN ===
if __name__ == "__main__":
    for folder in scan_folders:
        merge_folder(folder)
    print("\n🎉 All images merged and saved to FineImages!")
