from utils import *
import pathlib
import os
import json
import time
import tifffile as tiff
import numpy as np
import traceback as trackback

notebook_dir = pathlib.Path().resolve()
watch_dir = notebook_dir / "data" / "input"
watch_dir.mkdir(exist_ok=True)

processed_jsons = set()
precomputed_blobs = {
    "red": {},
    "green": {},
    "blue": {}
}

print("Looking for analysis_params.json and beamline_params.json")

analysis_params_path = watch_dir / "analysis_params.json"
beamline_params_path = watch_dir / "beamline_params.json"
scan_params_path = watch_dir / "scan_200_params.json"

missing_files = []
if not analysis_params_path.exists():
    missing_files.append("analysis_params.json")
if not beamline_params_path.exists():
    missing_files.append("beamline_params.json")
if not scan_params_path.exists():
    missing_files.append("scan_200_params.json")

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
    with open(scan_params_path, "r") as f:
        scan_params = json.load(f)
    print("Loaded analysis_params.json:")
    print(analysis_params)
    print("Loaded beamline_params.json:")
    print(beamline_params)
    print("Loaded scan_200_params.json:")
    print(scan_params)
except Exception as e:
    print(f"Error reading JSON files: {e}")
    exit(1)

# Only process TIFF files, since JSON is already loaded above
precomputed = {}

print()
#Later one remove the first 2 params and keep the 3rd for headless_send_queue_coarse_scan
initial_scan_path = watch_dir / "initial_scan.json"
headless_send_queue_coarse_scan("data/headless_scan", beamline_params, initial_scan_path)

# # Wait for all required TIFF files before proceeding
# required_elements = analysis_params["element_list"]
# #The path for Tiff we can change to the proper path later
# tiff_paths = wait_for_element_tiffs(required_elements, Path("data/headless_scan")) 

# # Define up to 8 colors for blob detection
# COLOR_ORDER = [
#     'red', 'blue', 'green', 'orange', 'purple', 'cyan', 'olive', 'yellow'
# ]

# print("\nProcessing TIFF files for blob detection...")
# precomputed_blobs = {color: {} for color in COLOR_ORDER}

# # Only process up to 8 TIFFs
# max_tiffs = min(len(tiff_paths), 8)
# processed_elements = list(tiff_paths.keys())[:max_tiffs]

# for idx, element in enumerate(processed_elements):
#     color = COLOR_ORDER[idx]
#     tiff_path = tiff_paths[element]
#     print(f"Processing {tiff_path.name} as color {color}")
#     try:
#         tiff_img = tiff.imread(str(tiff_path)).astype(np.float32)
#         tiff_norm, tiff_dilated = normalize_and_dilate(tiff_img)
#         b = detect_blobs(
#             tiff_dilated,
#             tiff_norm,
#             analysis_params["min_threshold_intensity"],
#             analysis_params["min_threshold_area"],
#             color,
#             tiff_path.name
#         )
#         precomputed_blobs[color][(analysis_params["min_threshold_intensity"], analysis_params["min_threshold_area"])] = b
#     except Exception as e:
#         print(f"❌ Error processing {tiff_path.name}: {e}")
#         trackback.print_exc()

# # Only run unions if at least 2 colors are present
# if len(processed_elements) >= 2:
#     unions = find_union_blobs(
#         precomputed_blobs,
#         scan_params["microns_per_pixel_x"],
#         scan_params["microns_per_pixel_y"],
#         scan_params["true_origin_x"],
#         scan_params["true_origin_y"]
#     )

#     formatted_unions = {}

#     print("Processsing images now")

#     for idx, union in unions.items():
#         box_name = f"Union Box #{idx}"
#         formatted = {
#             "text": box_name,
#             "image_center": union["center"],
#             "image_length": union["length"],
#             "image_area_px²": union["area"],
#             "real_center_um": union["real_center_um"],
#             "real_size_um": union["real_size_um"],
#             "real_area_um²": union["real_area_um\u00b2"],
#             "real_top_left_um": union["real_top_left_um"],
#             "real_bottom_right_um": union["real_bottom_right_um"]
#         }

#         formatted_unions[box_name] = formatted

#         # Print each formatted union box
#         print(f"\n{box_name}")
#         print(f"  Text:               {formatted['text']}")
#         print(f"  Image Center:       {formatted['image_center']}")
#         print(f"  Image Length:       {formatted['image_length']}")
#         print(f"  Image Area (px²):   {formatted['image_area_px²']}")
#         print(f"  Real Center (µm):   {formatted['real_center_um']}")
#         print(f"  Real Size (µm):     {formatted['real_size_um']}")
#         print(f"  Real Area (µm²):    {formatted['real_area_um²']}")
#         print(f"  Real Top Left (µm): {formatted['real_top_left_um']}")
#         print(f"  Real Bottom Right:  {formatted['real_bottom_right_um']}")
#         print("-" * 50)

#     # Save to file
#     output_path = "data/headless_scan/unions_output.json"
#     with open(output_path, "w") as f:
#         json.dump(formatted_unions, f, indent=2)
#     print(f"\n✅ Union data saved to: {output_path}")
#     print()
    #change the output_dir to the proper path later the out_dir from export and save 

###Here we will load the unions_output.json before as formatted_unions before running it into the other functions to send to bluesky

save_each_blob_as_individual_scan(formatted_unions, output_dir="data/headless_scan")
    #change the output_dir to the proper path later the out_dir from export and save 
headless_send_queue_fine_scan("data/headless_scan", beamline_params, output_dir="data/headless_scan")
# else:
    # print("Not enough TIFFs to perform union operation (need at least 2 Elements)")
