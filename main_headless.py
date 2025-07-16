from utils import *
import pathlib
import os
import json
import time
import tifffile as tiff
import numpy as np
import traceback as trackback
# === Setup ===
notebook_dir = pathlib.Path().resolve()
watch_dir = notebook_dir / "input"
watch_dir.mkdir(exist_ok=True)

def find_unique_tiff_files(folder):
    return list({f for f in os.listdir(folder) if f.lower().endswith((".tiff", ".tif"))})

processed_jsons = set()
precomputed_blobs = {
    "red": {},
    "green": {},
    "blue": {}
}

print("Start")
print("Looking for analysis_params.json and beamline_params.json")

analysis_params_path = watch_dir / "analysis_params.json"
beamline_params_path = watch_dir / "beamline_params.json"

missing_files = []
if not analysis_params_path.exists():
    missing_files.append("analysis_params.json")
if not beamline_params_path.exists():
    missing_files.append("beamline_params.json")

if missing_files:
    for fname in missing_files:
        print(f"File {fname} is not present")
    exit(1)

# Load both JSON files
try:
    with open(analysis_params_path, "r") as f:
        analysis_params = json.load(f)
    with open(beamline_params_path, "r") as f:
        beamline_params = json.load(f)
    print("Loaded analysis_params.json:")
    print(analysis_params)
    print("Loaded beamline_params.json:")
    print(beamline_params)
except Exception as e:
    print(f"Error reading JSON files: {e}")
    exit(1)

# analysis_params = [[key, value] for key, value in analysis_params.items()]
# beamline_params = [[key, value] for key, value in beamline_params.items()]

# Only process TIFF files, since JSON is already loaded above
precomputed = {}
print("\nWaiting for 3 unique .tiff files...")
tiff_files = find_unique_tiff_files(watch_dir)
if len(tiff_files) >= 3:
    first_three = sorted(tiff_files)[:3]
    tiff1_path = watch_dir / first_three[0]
    tiff2_path = watch_dir / first_three[1]
    tiff3_path = watch_dir / first_three[2]

    print("\n✅ Found 3 TIFF files:")
    print(f"TIFF 1: {tiff1_path.name}")
    print(f"TIFF 2: {tiff2_path.name}")
    print(f"TIFF 3: {tiff3_path.name}")
    try:
        tiff1_img = tiff.imread(str(tiff1_path)).astype(np.float32)
        tiff1_norm, tiff1_dilated = normalize_and_dilate(tiff1_img)

        tiff2_img = tiff.imread(str(tiff2_path)).astype(np.float32)
        tiff2_norm, tiff2_dilated = normalize_and_dilate(tiff2_img)

        tiff3_img = tiff.imread(str(tiff3_path)).astype(np.float32)
        tiff3_norm, tiff3_dilated = normalize_and_dilate(tiff3_img)
        
        b1 = detect_blobs(tiff1_dilated, tiff1_norm, analysis_params["min_threshold_intensity"], analysis_params["min_threshold_area"], "red", tiff1_path.name)
        precomputed_blobs["red"][(analysis_params["min_threshold_intensity"], analysis_params["min_threshold_area"])] = b1

        b2 = detect_blobs(tiff2_dilated, tiff2_norm, analysis_params["min_threshold_intensity"], analysis_params["min_threshold_area"], "blue", tiff2_path.name)
        precomputed_blobs["blue"][(analysis_params["min_threshold_intensity"], analysis_params["min_threshold_area"])] = b2

        b3 = detect_blobs(tiff3_dilated, tiff3_norm, analysis_params["min_threshold_intensity"], analysis_params["min_threshold_area"], "green", tiff3_path.name)
        precomputed_blobs["green"][(analysis_params["min_threshold_intensity"], analysis_params["min_threshold_area"])] = b3
        
        unions = find_union_blobs(
            precomputed_blobs,
            analysis_params["microns_per_pixel_x"],
            analysis_params["microns_per_pixel_y"],
            analysis_params["true_origin_x"],
            analysis_params["true_origin_y"]
        )

        formatted_unions = {}

        print("Processsing images now")

        for idx, union in unions.items():
            box_name = f"Union Box #{idx}"
            formatted = {
                "text": box_name,
                "image_center": union["center"],
                "image_length": union["length"],
                "image_area_px²": union["area"],
                "real_center_um": union["real_center_um"],
                "real_size_um": union["real_size_um"],
                "real_area_um²": union["real_area_um\u00b2"],
                "real_top_left_um": union["real_top_left_um"],
                "real_bottom_right_um": union["real_bottom_right_um"]
            }

            formatted_unions[box_name] = formatted

            # Print each formatted union box
            print(f"\n{box_name}")
            print(f"  Text:               {formatted['text']}")
            print(f"  Image Center:       {formatted['image_center']}")
            print(f"  Image Length:       {formatted['image_length']}")
            print(f"  Image Area (px²):   {formatted['image_area_px²']}")
            print(f"  Real Center (µm):   {formatted['real_center_um']}")
            print(f"  Real Size (µm):     {formatted['real_size_um']}")
            print(f"  Real Area (µm²):    {formatted['real_area_um²']}")
            print(f"  Real Top Left (µm): {formatted['real_top_left_um']}")
            print(f"  Real Bottom Right:  {formatted['real_bottom_right_um']}")
            print("-" * 50)

        # Save to file
        output_path = notebook_dir / "unions_output.json"
        with open(output_path, "w") as f:
            json.dump(formatted_unions, f, indent=2)
        print(f"\n✅ Union data saved to: {output_path}")
        
        print("Data send to queue server")
        save_each_blob_as_individual_scan(formatted_unions, px_per_um=1.25, output_dir="headless_scan")
        headless_send_queue("headless_scan", beamline_params)

    except Exception as e:
        print(f"❌ Error processing TIFFs: {e}")
        trackback.print_exc()
else:
    print(f"Currently found {len(tiff_files)} TIFF file(s). Need at least 3. Exiting.")
