import json
import os
import pathlib
import traceback as trackback

import numpy as np
import tifffile as tiff

from app_state import AppState
from utils import (
    detect_blobs,
    find_union_blobs,
    headless_send_queue_coarse_scan,
    headless_send_queue_fine_scan,
    normalize_and_dilate,
    save_each_blob_as_individual_scan,
    wait_for_element_tiffs,
)


def load_json_file(path):
    """Loads a single JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.name}")
    with open(path, "r") as f:
        return json.load(f)


def load_parameters(watch_dir):
    """Loads all required JSON parameter files."""
    print("Looking for initial_scan.json")#, beamline_params.json, and scan_200_params.json")
    try:
        analysis_params = load_json_file(watch_dir / "initial_scan.json")
        # beamline_params = load_json_file(watch_dir / "beamline_params.json")
        # scan_params = load_json_file(watch_dir / "scan_200_params.json")
        print("Loaded initial_scan.json:", analysis_params)
        # print("Loaded beamline_params.json:", beamline_params)
        # print("Loaded scan_200_params.json:", scan_params)
        return analysis_params#, beamline_params, scan_params
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON files: {e}")
        exit(1)


def run_headless_processing():
    """Main function to run the headless processing workflow."""
    state = AppState()

    notebook_dir = pathlib.Path().resolve()
    watch_dir = notebook_dir / "data" / "input"
    watch_dir.mkdir(exist_ok=True)

    # analysis_params, beamline_params, scan_params = load_parameters(watch_dir)
    analysis_params = load_parameters(watch_dir)

    # Populate state from loaded params
    # state.microns_per_pixel_x = scan_params.get("microns_per_pixel_x")
    # state.microns_per_pixel_y = scan_params.get("microns_per_pixel_y")
    # state.true_origin_x = scan_params.get("true_origin_x")
    # state.true_origin_y = scan_params.get("true_origin_y")
    # state.element_colors = analysis_params.get("elem_list", [])


    initial_scan_path = watch_dir / "initial_scan.json"
    # Perform coarse scan
    # print("\nStarting coarse scan...")
    #1 is real run time, 0 is testing
    headless_send_queue_coarse_scan(analysis_params, initial_scan_path, 0) #Where scan used to start here

    #Will need to modify gui for these changes of test vs real params as well 

    #['Fe','Ca','S']

    #Grid Scan
    print("\nGrid scan starts here")
    # mosaic_overlap_scan(dets = None, ylen = 100, xlen = 100, overlap_per = 15, dwell = 0.05,
                        # step_size = 500, plot_elem = ["None"],mll = False, beamline_params=beamline_params, initial_scan_path=initial_scan_path)
    print("Scans Done")
if __name__ == "__main__":
    run_headless_processing()