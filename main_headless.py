from utils import *
import pathlib
import os
import json
import time
import tifffile as tiff
import numpy as np

# === Setup ===
notebook_dir = pathlib.Path().resolve()
watch_dir = notebook_dir / "input"
watch_dir.mkdir(exist_ok=True)

def find_unique_tiff_files(folder):
    return list({f for f in os.listdir(folder) if f.lower().endswith((".tiff", ".tif"))})

processed_jsons = set()
json_items = []
precomputed_blobs = {
    "red": {},
    "green": {},
    "blue": {}
}
print("Start")
print("Please provide json file")
while True:
    # Step 1: Look for new JSON files
    json_files = [f for f in os.listdir(watch_dir) if f.endswith(".json") and f not in processed_jsons]
    
    if json_files:
        for json_file in json_files:
            file_path = watch_dir / json_file
            print(f"\n--- Reading JSON: {json_file} ---")
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        json_items = [[key, value] for key, value in data.items()]
                        print("Stored JSON key-value pairs:")
                        for pair in json_items:
                            print(pair)
                    else:
                        json_items = [data]  # fallback for non-dict JSON
                        print(json_items)
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                continue
            
            processed_jsons.add(json_file)

            # Step 2: Wait for 3 unique TIFF files
            precomputed = {}
            print("\nWaiting for 3 unique .tiff files...")
            while True:
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
                        
                        b1 = detect_blobs(tiff1_dilated, tiff1_norm, json_items[0][1], json_items[1][1], "red", tiff1_path.name)
                        precomputed_blobs["red"][(json_items[0][1], json_items[1][1])] = b1

                        b2 = detect_blobs(tiff2_dilated, tiff2_norm, json_items[0][1], json_items[1][1], "blue", tiff2_path.name)
                        precomputed_blobs["blue"][(json_items[0][1], json_items[1][1])] = b2

                        b3 = detect_blobs(tiff3_dilated, tiff3_norm, json_items[0][1], json_items[1][1], "green", tiff3_path.name)
                        precomputed_blobs["green"][(json_items[0][1], json_items[1][1])] = b3

                        # print(precomputed_blobs)
                        
                        unions = find_union_blobs(precomputed_blobs, json_items[2][1], json_items[3][1], json_items[4][1], json_items[5][1])

                        formatted_unions = {}


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
                        
                        save_each_blob_as_individual_scan(formatted_unions, px_per_um=1.25, output_dir="output")

                        
                        # Example: send to processing function
                        # process_images(tiff1_dilated, tiff2_dilated, tiff3_dilated)

                    except Exception as e:
                        print(f"❌ Error processing TIFFs: {e}")                            
                        break

                    break
                else:
                    print(f"Currently found {len(tiff_files)} TIFF file(s). Waiting...")
                    time.sleep(5)
    else:
        # print("No new JSON files found.")
        time.sleep(1)
