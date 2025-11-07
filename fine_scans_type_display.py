
from PIL import Image, ImageDraw, ImageFont

def create_scan_type_display_image(output_path="scan_type_display.png"):
    """
    Creates an image with a white background, a 3x3 grid of black boxes,
    and column titles "Separate", "Together", "Partial".
    """
    # Image dimensions and colors
    box_size = 100
    padding = 30
    main_title_height = 80
    column_title_height = 50
    num_columns = 3
    num_rows = 3

    image_width = num_columns * box_size + (num_columns + 1) * padding
    image_height = num_rows * box_size + (num_rows + 1) * padding + main_title_height + column_title_height

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
    except IOError:
        main_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        column_font = ImageFont.load_default()

    # Main title
    main_title = "Fine Scan Different Types"
    x = image_width // 2
    y = padding
    bbox = draw.textbbox((0, 0), main_title, font=main_font)
    text_width = bbox[2] - bbox[0]
    draw.text((x - text_width // 2, y), main_title, font=main_font, fill=text_color)

    # Subtitle
    subtitle = "Elements Fe Ca Si"
    y += 40  # Spacing below main title
    bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    text_width = bbox[2] - bbox[0]
    draw.text((x - text_width // 2, y), subtitle, font=subtitle_font, fill=text_color)


    # Column titles
    column_titles = ["Separate", "Together", "Partial"]
    for i, title in enumerate(column_titles):
        x = padding + i * (box_size + padding) + box_size // 2
        y = main_title_height + column_title_height // 2
        # Center the text
        bbox = draw.textbbox((0, 0), title, font=column_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text((x - text_width // 2, y - text_height // 2), title, font=column_font, fill=text_color)

    # Draw the 3x3 grid of black boxes
    for row in range(num_rows):
        for col in range(num_columns):
            x1 = padding + col * (box_size + padding)
            y1 = main_title_height + column_title_height + row * (box_size + padding)
            x2 = x1 + box_size
            y2 = y1 + box_size
            draw.rectangle([x1, y1, x2, y2], fill=box_color)

    # Save the image
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    create_scan_type_display_image(output_path="/home/codingcarlos/Documents/AddGoodSamples/FineScanTypesShowcase.png")
