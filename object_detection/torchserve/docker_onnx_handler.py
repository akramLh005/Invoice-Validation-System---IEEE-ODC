from functools import partial
import io
import json
import logging
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
from torchvision.transforms.functional import to_tensor


# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class DetectionModelHandler:
    def __init__(self):
        self.initialized = False
        self.session = None  # ONNX Runtime session


    def initialize(self, context):
        model_path = "/home/model-server/model-store/modelPrunned.onnx"  # Path to the ONNX model
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name  # Ensure this attribute is defined
        self.initialized = True
        logging.info("ONNX model initialized successfully.")


    def preprocess(self, data):
        if isinstance(data, dict):
            image = data.get("data") or data.get("body")
        elif isinstance(data, list) and data:
            image = data[0].get("data") or data[0].get("body")
        else:
            raise ValueError("Data provided is neither dict nor list, or it is empty.")


        if isinstance(image, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image))
            image = np.array(image)
        else:
            raise ValueError("Invalid image format: not bytes or bytearray")


        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))  # Resize to (640, 640)
        image = image.astype(np.float32) / 255.0  # Normalize


        image_tensor = np.transpose(image, (2, 0, 1))  # Change data layout to CHW
        image_tensor = np.expand_dims(image_tensor, axis=0)  # Add batch dimension
        return image_tensor


    def inference(self, model_input):
        inputs = {self.input_name: model_input}
        outputs = self.session.run(None, inputs)
        logging.info("Inference executed successfully")


        # Assuming outputs are numpy arrays and indexing is correct
        if len(outputs[0]) == 0:
            logging.info("No objects detected.")
        else:
            logging.info("Objects detected: %d", len(outputs[0]))


        output_dict = {
            'boxes': outputs[0].tolist(),
            'scores': outputs[1].tolist(),
            'labels': outputs[2].tolist()
        }
        return output_dict


    def postprocess(self, inference_output):
        return json.dumps(inference_output).encode('utf-8')


    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        response = self.postprocess(model_output)
        return [response]
