import fitz  # PyMuPDF
import os
import glob
import numpy as np

#define parameters
THRESHOLD=220
STDDEV_THRESHOLD=15

def is_blank_image_page(pix, threshold=THRESHOLD, stddev_threshold=STDDEV_THRESHOLD):
    """
    Determine if the page is visually blank by analyzing image data.
    A page is considered blank if the mean intensity is very high (close to 255) and the standard deviation is low (indicating uniform color).

    Parameters:
    - pix: A pixmap object of the page.
    - threshold: Intensity threshold to consider the page as blank (default 235).
    - stddev_threshold: Standard deviation threshold for intensity to consider the page as blank (default 15).
    """
    # Convert pixmap samples to a numpy array
    image_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:  # Assuming RGB or RGBA
        gray = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale assuming RGB
    else:  # Grayscale or single channel
        gray = image_array

    mean_intensity = np.mean(gray)
    stddev_intensity = np.std(gray)

    # Determine if the page is blank based on mean intensity and standard deviation
    return mean_intensity > threshold and stddev_intensity < stddev_threshold

def convert_pdf_to_png(pdf_path, output_folder):
    """
    Converts each non-blank page of a PDF to a PNG file.

    Parameters:
    - pdf_path: Path to the PDF file.
    - output_folder: Directory where the output PNG images will be saved.
    """
    doc = fitz.open(pdf_path)  # Open the PDF file
    image_files = []  # List to store output filenames
    for i in range(len(doc)):  # Iterate over each page
        page = doc[i]
        pix = page.get_pixmap()  # Render page to an image
        if not is_blank_image_page(pix):  # Check if the page is not blank
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            image_path = os.path.join(output_folder, f"{base_name}_{i+1}.png")
            pix.save(image_path)  # Save the image if it's not blank
            image_files.append(image_path)  # Add the filename to the list
    doc.close()  # Close the PDF after processing
    return image_files  # Return the list of generated files

def convert_pdf_images_to_png(input_folder, output_folder):
    """
    Converts all non-blank PDF pages in a specified directory to PNG images.

    Parameters:
    - input_folder: Directory containing PDF files.
    - output_folder: Directory where the output PNG images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create output directory if it does not exist

    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))  # Find all PDF files in the input folder

    for pdf_file in pdf_files:
        convert_pdf_to_png(pdf_file, output_folder)  # Convert each non-blank PDF page to PNG

# Paths to the input and output folders
input_folder = "invoices"
output_folder = "invoices_png"

convert_pdf_images_to_png(input_folder, output_folder)  # Run the conversion process
