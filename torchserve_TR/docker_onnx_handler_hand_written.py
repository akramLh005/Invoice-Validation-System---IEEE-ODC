import io
from PIL import Image
import onnxruntime as ort
from transformers import LayoutLMv3Processor
from ts.torch_handler.base_handler import BaseHandler

class LayoutLMv3Handler(BaseHandler):
    """
    TorchServe handler for serving LayoutLMv3 models in ONNX format, 
    returning labels and bounding boxes. The handler also includes custom
    label-to-id mappings directly within the class.
    """

    def __init__(self):
        super(LayoutLMv3Handler, self).__init__()
        self.initialized = False
        # Define the id-to-label mappings within the handler
        self.id2label = {
            0: 'O',
            1: 'B-title',
            2: 'I-title',
            3: 'B-date',
            4: 'I-date',
            5: 'B-ieee',
            6: 'I-ieee',
            7: 'B-total',
            8: 'I-total',
            9: 'B-totalValue',
            10: 'I-totalValue'
        }

    def initialize(self, context):
        model_path = "/home/model-server/model-store/layoutlmv3_hand_written.onnx"
        # Load the ONNX model session
        self.ort_session = ort.InferenceSession(model_path)
        # Initialize the LayoutLMv3 processor
        self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)

        self.initialized = True

    def preprocess(self, data):
        # Check the type of the data and prepare the image accordingly
        if isinstance(data[0].get("data"), bytearray):
            image_data = io.BytesIO(data[0].get("data"))
        else:
            image_path = data[0].get("data") or data[0].get("body")
            image_data = open(image_path, 'rb')

        image = Image.open(image_data).convert("RGB")
        encoding = self.processor(image, return_tensors="pt")
        
        # Convert tensors to numpy arrays for ONNX
        return {
            'input_ids': encoding["input_ids"].numpy(),
            'bbox': encoding["bbox"].numpy(),
            'attention_mask': encoding["attention_mask"].numpy()
        }

    def inference(self, input_batch):
        # Run inference using ONNX Runtime
        outputs = self.ort_session.run(None, input_batch)
        return outputs  # Return all outputs including logits

    def postprocess(self, inference_output):
        # Ensure response for each input
        batch_responses = []
        for output in inference_output:
            predictions = output.argmax(-1).squeeze().tolist()
            labels = [self.id2label[pred] for pred in predictions]
            batch_responses.append(labels)
        return batch_responses

_service = LayoutLMv3Handler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)
    return data
