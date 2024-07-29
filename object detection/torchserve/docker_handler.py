from functools import partial
import io
import json
import logging
import numpy as np
import torch
import torchvision
import cv2
from PIL import Image
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.transforms.functional import to_tensor

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DetectionModelHandler:
    def __init__(self):
        self.initialized = False

    def initialize(self, context):
        properties = context.system_properties
        model_path = "/home/model-server/model-store/best_model.pth"
        weights_path = "/home/model-server/model-store/retinanet_resnet50_fpn_v2_coco-5905b1c5.pth"
        self.model = self.load_model(model_path, weights_path, num_classes=3)
        self.model.eval()
        self.initialized = True
        logging.info("Model initialized successfully.")

    def load_model(self, model_path, weights_path, num_classes):
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights=None)  # No pretrained weights
        # Load the pre-trained backbone weights
        state_dict = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)

        # Update the classification head
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )

        # Load the custom trained weights
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        return model

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

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image_tensor = to_tensor(image).unsqueeze_(0).to(DEVICE)
        return image_tensor

    def inference(self, model_input):
        logging.info("Model input size: %s", model_input.size())
        with torch.no_grad():
            outputs = self.model(model_input)
        if outputs[0]['boxes'].nelement() == 0:
            logging.info("No objects detected.")
        else:
            logging.info("Objects detected: %d", outputs[0]['boxes'].shape[0])
        output_dict = {
            'boxes': outputs[0]['boxes'].cpu().tolist(),
            'labels': outputs[0]['labels'].cpu().tolist(),
            'scores': outputs[0]['scores'].cpu().tolist()
        }
        return output_dict

    def postprocess(self, inference_output):
        return json.dumps(inference_output).encode('utf-8')

    def handle(self, data, context):
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        response = self.postprocess(model_output)
        return [response]
