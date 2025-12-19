
from PIL import Image, ImageDraw, ImageFont
import os
import shutil

def create_scan_type_display_image(image_paths, micron_scales, output_path="scan_type_display.png"):
    """
    Creates an image with a white background, a 3x3 grid of images from TIFF files,
    and column titles "Separate", "Together", "Partial".
    """
    # Image dimensions and colors
    box_size = 100
    padding = 30
    main_title_height = 80
    column_title_height = 50
    legend_height = 30 # New height for the legend
    micron_scale_text_height = 20 # Height for the micron scale text
    scale_bar_height = 10 # Height for the scale bar
    num_columns = 3
    num_rows = 3

    image_width = num_columns * box_size + (num_columns + 1) * padding
    image_height = num_rows * (box_size + micron_scale_text_height + scale_bar_height) + (num_rows + 1) * padding + main_title_height + column_title_height + legend_height

    background_color = (255, 255, 255)  # White
    box_color = (0, 0, 0)  # Black
    text_color = (0, 0, 0)  # Black

    # Create a new image with a white background
    image = Image.new("RGB", (image_width, image_height), background_color)
    draw = ImageDraw.Draw(image)

    # Try to load a font, fallback to default if not found
    try:
        main_font = ImageFont.truetype("arial.ttf", 30)
        subtitle_font = ImageFont.truetype("arial.ttf", 15)
        column_font = ImageFont.truetype("arial.ttf", 20)  # Adjust font size as needed
        micron_font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        main_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        column_font = ImageFont.load_default()
        micron_font = ImageFont.load_default()

    # Main title
    main_title = "Fine Scan Different Types"
    x = image_width // 2
    y = padding
    bbox = draw.textbbox((0, 0), main_title, font=main_font)
    text_width = bbox[2] - bbox[0]
    draw.text((x - text_width // 2, y), main_title, font=main_font, fill=text_color)

    # Subtitle
    subtitle = "Elements Fe Ca Si"
    y_subtitle = y + 40  # Spacing below main title
    bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    text_width = bbox[2] - bbox[0]
    draw.text((x - text_width // 2, y_subtitle), subtitle, font=subtitle_font, fill=text_color)

    # Color Legend
    y_legend = y_subtitle + 20 # Spacing below subtitle
    
    # Calculate total width of legend text to center it
    ca_bbox = draw.textbbox((0,0), "Ca (Red)", font=subtitle_font)
    fe_bbox = draw.textbbox((0,0), "Fe (Green)", font=subtitle_font)
    si_bbox = draw.textbbox((0,0), "Si (Blue)", font=subtitle_font)

    ca_width = ca_bbox[2] - ca_bbox[0]
    fe_width = fe_bbox[2] - fe_bbox[0]
    si_width = si_bbox[2] - si_bbox[0]

    total_legend_width = ca_width + fe_width + si_width + 20 # Add some spacing between elements
    start_x_legend = (image_width - total_legend_width) // 2

    # Draw "Ca (Red)"
    draw.text((start_x_legend, y_legend), "Ca (Red)", font=subtitle_font, fill=(255, 0, 0))
    start_x_legend += ca_width + 10

    # Draw "Fe (Green)"
    draw.text((start_x_legend, y_legend), "Fe (Green)", font=subtitle_font, fill=(0, 128, 0)) # Darker green for visibility
    start_x_legend += fe_width + 10

    # Draw "Si (Blue)"
    draw.text((start_x_legend, y_legend), "Si (Blue)", font=subtitle_font, fill=(0, 0, 255))


    # Column titles
    column_titles = ["Separate", "Together", "Partial"]
    for i, title in enumerate(column_titles):
        x = padding + i * (box_size + padding) + box_size // 2
        y = main_title_height + column_title_height // 2 + legend_height # Adjust y for legend
        # Center the text
        bbox = draw.textbbox((0, 0), title, font=column_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((x - text_width // 2, y - text_height // 2), title, font=column_font, fill=text_color)

    # Draw the 3x3 grid of images
    for col, title in enumerate(column_titles):
        for row in range(num_rows):
            x1 = padding + col * (box_size + padding)
            y1 = main_title_height + column_title_height + legend_height + row * (box_size + padding + micron_scale_text_height + scale_bar_height) # Adjust y for legend and micron scale
            
            file_path = image_paths[title][row]
            
            try:
                # Open and resize the TIFF image
                img = Image.open(file_path)
                img = img.resize((box_size, box_size))
                image.paste(img, (x1, y1))
            except (IOError, FileNotFoundError):
                # Draw a black box if the image can't be loaded
                x2 = x1 + box_size
                y2 = y1 + box_size
                draw.rectangle([x1, y1, x2, y2], fill=box_color)

            # Draw the micron scale bar (caliper style)
            bar_y = y1 + box_size + 9 # Horizontal bar 9 pixels below image
            vertical_line_start_y = y1 + box_size + 4 # Vertical lines start 6 pixels below image
            # Horizontal line
            draw.line([(x1, bar_y), (x1 + box_size, bar_y)], fill=box_color, width=1)
            # Vertical lines with space
            draw.line([(x1, vertical_line_start_y), (x1, bar_y)], fill=box_color, width=1)
            draw.line([(x1 + box_size, vertical_line_start_y), (x1 + box_size, bar_y)], fill=box_color, width=1)

            # Draw the micron scale text
            scale = micron_scales[title][row]
            scale_text = f"{scale}_microns"
            bbox = draw.textbbox((0, 0), scale_text, font=micron_font)
            text_width = bbox[2] - bbox[0]
            draw.text((x1 + (box_size - text_width) // 2, bar_y + 2), scale_text, font=micron_font, fill=text_color)


    # Save the image
    image.save(output_path)
    print(f"Image saved to {output_path}")

def export_higher_res_fine_scan_merges(
    scan_entries,
    output_root="/home/codingcarlos/Documents/AddGoodSamples/higherres",
    output_size=(4096, 4096),
    fine_scan_root="/home/codingcarlos/Desktop/Data/Beamline_Data/Automap_2025Q3/all_xrf",
    elements_map=None,
):
    """
    Copies elemental TIFF files for the specified fine scans, merges them at 4K,
    and writes PNGs named coarse_##_fine_##_type.png under the output root.
    """
    os.makedirs(output_root, exist_ok=True)

    if elements_map is None:
        elements_map = {"r": "Ca", "g": "Fe", "b": "Si"}

    for entry in scan_entries:
        coarse_id = entry["coarse_id"]
        fine_id = entry["fine_id"]
        scan_type = entry["type"]
        element_paths = {
            "r": os.path.join(
                fine_scan_root,
                f"output_tiff_scan2D_{fine_id}",
                f"detsum_{elements_map['r']}_K_norm.tiff",
            ),
            "g": os.path.join(
                fine_scan_root,
                f"output_tiff_scan2D_{fine_id}",
                f"detsum_{elements_map['g']}_K_norm.tiff",
            ),
            "b": os.path.join(
                fine_scan_root,
                f"output_tiff_scan2D_{fine_id}",
                f"detsum_{elements_map['b']}_K_norm.tiff",
            ),
        }

        dest_dir = os.path.join(output_root, f"coarse_{coarse_id}_fine_{fine_id}")
        os.makedirs(dest_dir, exist_ok=True)

        copied_paths = {}
        for element, src_path in element_paths.items():
            if not os.path.exists(src_path):
                print(f"Missing element file for {fine_id}: {src_path}")
                copied_paths = None
                break
            dest_path = os.path.join(dest_dir, os.path.basename(src_path))
            shutil.copy(src_path, dest_path)
            copied_paths[element] = dest_path

        if not copied_paths:
            continue

        try:
            r_img = Image.open(copied_paths["r"])
            g_img = Image.open(copied_paths["g"])
            b_img = Image.open(copied_paths["b"])

            def normalize(arr):
                arr = arr.astype("float32")
                if arr.max() > arr.min():
                    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
                return arr.astype("uint8")

            import numpy as np
            r_np = normalize(np.array(r_img))
            g_np = normalize(np.array(g_img))
            b_np = normalize(np.array(b_img))
            rgb_np = np.stack([r_np, g_np, b_np], axis=-1)
            merged = Image.fromarray(rgb_np, "RGB")

            merged = merged.resize(output_size, Image.LANCZOS)
            out_name = f"coarse_{coarse_id}_fine_{fine_id}_{scan_type}.png"
            out_path = os.path.join(output_root, out_name)
            merged.save(out_path)
            print(f"Saved 4K merged image to {out_path}")
        except Exception as exc:
            print(f"Failed to merge fine scan {fine_id}: {exc}")

if __name__ == "__main__":
    # Define the file paths for the TIFF images
    image_paths = {
        "Separate": [
            "/home/codingcarlos/Documents/AddGoodSamples/Seperate/scan_368604_Coarse_Fine_Scans/merged_detsum_368606.png",
            "/home/codingcarlos/Documents/AddGoodSamples/Seperate/scan_369139_Coarse_Fine_Scans/merged_detsum_369140.png",
            "/home/codingcarlos/Documents/AddGoodSamples/Seperate/scan_369155_Coarse_Fine_Scans/merged_detsum_369158.png"
        ],
        "Together": [
            "/home/codingcarlos/Documents/AddGoodSamples/Together/scan_368612_Coarse_Fine_Scans/merged_detsum_368613.png",
            "/home/codingcarlos/Documents/AddGoodSamples/Together/scan_368950_Coarse_Fine_Scans/merged_detsum_368953.png",
            "/home/codingcarlos/Documents/AddGoodSamples/Together/scan_369089_Coarse_Fine_Scans/merged_detsum_369090.png"
        ],
        "Partial": [
            "/home/codingcarlos/Documents/AddGoodSamples/Partial/scan_369068_Coarse_Fine_Scans/merged_detsum_369071.png",
            "/home/codingcarlos/Documents/AddGoodSamples/Partial/scan_369009_Coarse_Fine_Scans/merged_detsum_369012.png",
            "/home/codingcarlos/Documents/AddGoodSamples/Partial/scan_369116_Coarse_Fine_Scans/merged_detsum_369117.png"
        ]
    }
    # Define the micron scales for each image
    micron_scales = {
        "Separate": [5.8, 1.8, 4.5],
        "Together": [2.2, 1.8, 1.5],
        "Partial": [2.5, 1.2, 3.5]
    }
    create_scan_type_display_image(image_paths, micron_scales, output_path="/home/codingcarlos/Documents/AddGoodSamples/FineScanTypesShowcase.png")

    scan_entries = [
        {
            "coarse_id": 368604,
            "fine_id": 368606,
            "type": "separate",
        },
        {
            "coarse_id": 369139,
            "fine_id": 369140,
            "type": "separate",
        },
        {
            "coarse_id": 369155,
            "fine_id": 369158,
            "type": "separate",
        },
        {
            "coarse_id": 368612,
            "fine_id": 368613,
            "type": "together",
        },
        {
            "coarse_id": 368950,
            "fine_id": 368953,
            "type": "together",
        },
        {
            "coarse_id": 369089,
            "fine_id": 369090,
            "type": "together",
        },
        {
            "coarse_id": 369068,
            "fine_id": 369071,
            "type": "partial",
        },
        {
            "coarse_id": 369009,
            "fine_id": 369012,
            "type": "partial",
        },
        {
            "coarse_id": 369116,
            "fine_id": 369117,
            "type": "partial",
        },
    ]
    export_higher_res_fine_scan_merges(scan_entries)
