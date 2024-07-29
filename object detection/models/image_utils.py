from io import BytesIO
import torch
import cv2
import numpy as np
import fitz  # PyMuPDF (also known as PyMuPDF)
import numpy as np

# Set device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def read_and_preprocess_image(file: BytesIO):
    """
    Reads an image from a BytesIO stream and preprocesses it for model inference.

    This function decodes an image from a BytesIO stream, converts it from BGR to RGB,
    normalizes it, and transforms it into a tensor suitable for input to a neural network.

    Args:
        file (BytesIO): The input file stream containing the image data.

    Returns:
        tuple: A tuple containing the preprocessed image tensor and a copy of the original image.

    Raises:
        ValueError: If the image cannot be decoded from the provided BytesIO stream.
    """
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image, ensure it is in a valid format")
    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image_input = torch.tensor(image_input, dtype=torch.float).to(DEVICE)
    image_input = torch.unsqueeze(image_input, 0)
    return image_input, orig_image


# Define parameters
THRESHOLD = 220
STDDEV_THRESHOLD = 15

def is_blank_image_page(pix, threshold=THRESHOLD, stddev_threshold=STDDEV_THRESHOLD):
    """
    Determine if the page is visually blank by analyzing image data.
    """
    # Convert pixmap samples to a numpy array
    image_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:  # Assuming RGB or RGBA
        gray = np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale assuming RGB
    else:
        gray = image_array

    mean_intensity = np.mean(gray)
    stddev_intensity = np.std(gray)

    # Determine if the page is blank based on mean intensity and standard deviation
    return mean_intensity > threshold and stddev_intensity < stddev_threshold

def convert_pdf_to_images(file_stream):
    """
    Converts each non-blank page of a PDF to an image array and returns them.
    
    Args:
        file_stream: A stream of the PDF file (e.g., from an upload).

    Returns:
        List of image arrays of non-blank pages.
    """
    doc = fitz.open(stream=file_stream, filetype="pdf")
    images = []
    for page in doc:  # Iterate over each page
        pix = page.get_pixmap()  # Render page to an image
        if not is_blank_image_page(pix):  # Check if the page is not blank
            img = np.frombuffer(pix.tobytes(), dtype=np.uint8).reshape(pix.height, pix.width, 3 if pix.alpha == 0 else 4)
            images.append(img)
    doc.close()
    return images


def draw_bounding_boxes(orig_image, boxes, pred_classes, scores, COLORS, CLASSES):
    """
    Draws bounding boxes with class names and scores on an image.

    This function overlays rectangles and labels on the image for each detection,
    color-coded by class, showing the detection class and the confidence score.

    Args:
        orig_image (ndarray): The original image on which to draw the bounding boxes.
        boxes (list of tuples): A list of coordinates for each bounding box.
        pred_classes (list of str): A list of predicted class names for each box.
        scores (list of float): A list of scores for each class prediction.
        COLORS (list of tuples): A list of color tuples for drawing boxes.
        CLASSES (list of str): A list of class names corresponding to class indices.

    Returns:
        ndarray: The original image with bounding boxes and labels drawn on it.
    """
    for box, class_name, score in zip(boxes, pred_classes, scores):
        color = COLORS[CLASSES.index(class_name) % len(COLORS)]
        xmin, ymin, xmax, ymax = [int(coord) for coord in box]
        cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax), color, 2)
        label_text = f"{class_name}: {score:.2f}"
        cv2.putText(orig_image, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return orig_image
