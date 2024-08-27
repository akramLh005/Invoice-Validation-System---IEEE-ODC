from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from PIL import Image
import json
import io

class TransformersHandler:
    def __init__(self):
        self.model = None
        self.processor = None

    def initialize(self, context):
        model_dir = "/home/model-server/model-store/checkpoint-2000-focalloss-azure"
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir)
        self.processor = LayoutLMv3Processor.from_pretrained(model_dir, apply_ocr=True)

    def preprocess(self, data):
        image_data = data[0].get("body") or data[0].get("data")
        image = Image.open(io.BytesIO(image_data))
        return image

    def inference(self, image):
        encoding = self.processor(image, return_tensors="pt")
        outputs = self.model(**encoding)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        return predictions, encoding

    def postprocess(self, predictions, encoding):
        labels = [self.model.config.id2label[pred] for pred in predictions]
        results = {
            "labels": labels,
            "boxes": encoding['bbox'][0].tolist()
        }
        return json.dumps(results)

    def handle(self, data, context):
        image = self.preprocess(data)
        predictions, encoding = self.inference(image)
        response = self.postprocess(predictions, encoding)
        return [response]

