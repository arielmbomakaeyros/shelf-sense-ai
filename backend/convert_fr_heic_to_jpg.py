import os
from pathlib import Path
from pillow_heif import register_heif_opener
from PIL import Image

# Register HEIF opener
register_heif_opener()

def convert_heic_to_jpg(input_folder: str, output_folder: str = None) -> None:
    """Convert all HEIC images in a folder to JPG format."""
    
    # Create output folder if specified, otherwise use input folder
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    else:
        output_folder = input_folder
        
    # Get all HEIC files
    heic_files = Path(input_folder).glob("*.HEIC")
    
    for heic_file in heic_files:
        try:
            # Open HEIC image
            image = Image.open(heic_file)
            
            # Create output path
            output_path = Path(output_folder) / f"{heic_file.stem}.jpg"
            
            # Convert and save as JPG
            image.save(output_path, "JPEG")
            print(f"Converted: {heic_file.name} -> {output_path.name}")
            
        except Exception as e:
            print(f"Error converting {heic_file.name}: {str(e)}")

if __name__ == "__main__":
    # Replace with your input folder path
    input_folder = "images_llm"
    
    # Optional: specify output folder
    # output_folder = "path/to/output"
    
    convert_heic_to_jpg(input_folder)