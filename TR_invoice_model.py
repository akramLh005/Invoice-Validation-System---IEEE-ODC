from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor


class InvoiceModel:
    def __init__(self, model_path, processor_name="microsoft/layoutlmv3-base", apply_ocr=True):
        # Load the model and processor with provided parameters
        self.model = LayoutLMv3ForTokenClassification.from_pretrained("./checkpoints/checkpoint-2000-focalloss-azure")
        self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)

    def run_inference(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()


        # Convert prediction IDs to labels
        labels = [self.model.config.id2label[pred] for pred in predictions]
        boxes = inputs['bbox'][0].tolist()  # Extract bounding boxes

        return labels, boxes
