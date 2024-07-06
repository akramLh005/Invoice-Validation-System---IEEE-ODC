from io import BytesIO
from fastapi import FastAPI, File, HTTPException, UploadFile
from models.OD_model import DetectionModel
from models.image_utils import read_and_preprocess_image, draw_bounding_boxes

app = FastAPI()

# Define parameters
threshold = 0.3
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
CLASSES = ['__background__', 'stamp', 'signature']
NUM_CLASSES = len(CLASSES)

# Load the model
model_path = './checkpoints/best_model.pth'
model = DetectionModel(model_path,NUM_CLASSES)


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Receives an image file, processes it for object detection, and returns a JSON-formatted response indicating the presence of each predefined object class in the image.

    This endpoint utilizes the DetectionModel to perform object detection. The result is structured as a JSON object, providing a clear indication of which objects have been detected based on a predefined threshold.

    Args:
        file (UploadFile): The image file uploaded by the user.

    Returns:
        JSONResponse: A JSON object indicating detected objects and their presence.
    """

    if file.filename == '' or file.size == 0:
        raise HTTPException(status_code=400, detail="No file uploaded or the file is empty")

    # Read and preprocess the received image
    file_contents = await file.read()
    file_stream = BytesIO(file_contents)
    image_input, orig_image = read_and_preprocess_image(file_stream)

    outputs = model.predict(image_input)

    boxes = outputs[0]['boxes'].data.cpu().numpy()
    scores = outputs[0]['scores'].data.cpu().numpy()
    labels = outputs[0]['labels'].data.cpu().numpy()



    # Filter boxes with a score above a threshold
    filtered_indices = scores >= threshold
    boxes = boxes[filtered_indices]
    scores = scores[filtered_indices]
    pred_classes = [CLASSES[i] for i in labels[filtered_indices]]

    # Initialize response dict with False for each class
    response = {class_name: False for class_name in CLASSES if class_name != '__background__'}

    # The response based on detection scores
    for class_name, score in zip(pred_classes, scores):
        if score >= threshold:
            response[class_name] = True

    #if we want to draw bboxes
    #final_image = draw_bounding_boxes(orig_image, boxes, pred_classes, scores, COLORS, CLASSES)

    return response


