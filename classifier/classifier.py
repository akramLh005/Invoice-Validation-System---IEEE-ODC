import numpy as np
import onnxruntime as ort
import io
from PIL import Image
from torchvision import transforms
import logging

class ImageClassifier:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def preprocess_image(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        return image.numpy()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def classify_image(self, input_data):
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        result = self.session.run([output_name], {input_name: input_data})
        result_sigmoid = self.sigmoid(result[0])

        # Determine if the result is handwritten or machinewritten
        is_handwritten = result_sigmoid > 0.5
        classification_result = "handwritten" if is_handwritten else "machinewritten"

        # Log the processed output and classification
        logging.info(f"Sigmoid output: {result_sigmoid}")
        logging.info(f"Classified image as: {classification_result}")
        logging.info(f"returned result, is it handwritten? : {is_handwritten}")

        return is_handwritten  # Return True if handwritten, False if machinewritten
