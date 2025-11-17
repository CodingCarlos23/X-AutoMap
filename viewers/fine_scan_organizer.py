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
base_root = "/home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3/xrf_data_mosaic_auto_PM_200um"
# /home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3/xrf_data_mosaic_auto_PM_200um
# /home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3/all_xrf
save_root = "/home/codingcarlos/Desktop/Data/FineImages"
os.makedirs(save_root, exist_ok=True)

# === DEFINE SCAN RANGES ===
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

# Create consecutive ranges
scan_ranges = [(scan_ids[i], scan_ids[i+1]) for i in range(len(scan_ids)-1)]

# scan_ranges = [
#     (367582, 367589),
#     (367589, 367592),
#     (367592, 367596),
#     (367596, 367600),
# ]

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