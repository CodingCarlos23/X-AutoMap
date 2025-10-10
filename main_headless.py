import json
import os
import pathlib
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
# headless_send_queue_coarse_scan("data/headless_scan", beamline_params, initial_scan_path)
mosaic_overlap_scan(dets = None, ylen = 100, xlen = 100, overlap_per = 15, dwell = 0.05,
                    step_size = 500, plot_elem = ["None"],mll = False, beamline_params=beamline_params, initial_scan_path=initial_scan_path)
