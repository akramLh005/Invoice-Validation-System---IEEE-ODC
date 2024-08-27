import logging
from typing import List
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import httpx
import aiofiles
import tempfile
import os
import json

from classifier.classifier import ImageClassifier
from middleware.cors_setup import add_cors_middleware
from middleware.ip_whitelist import IPWhitelistMiddleware
from utils.postprocessing import process_torch_serve_response, aggregate_results
from utils.pdf_to_png import convert_pdf_to_png

app = FastAPI()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

#IP_whitelist
allow_all_ips = "*" in config['allowed_ips'] or "0.0.0.0" in config['allowed_ips']
app.add_middleware(IPWhitelistMiddleware, allowlist=config['allowed_ips'], allow_all=allow_all_ips)

# Add CORS middleware
add_cors_middleware(app)

# URLs for various services
retinanet_url = config['retinanet_torch_serve_url']
LMv3_machine_url = config['LMv3_machine_torch_serve_url']
LMv3_hand_url = config['LMv3_hand_torch_serve_url']

# Initialize the classifier with the ONNX model path
efficientnet_classifier = ImageClassifier("./classifier/efficientnet_classifier_model.onnx")

@app.post("/predict")
async def process_images(file_id: List[str] = Form(...), files: List[UploadFile] = File(...)):
    if len(files) != len(file_id):
        raise HTTPException(status_code=400, detail="Number of file_id and files must be equal.")

    results = []
    async with httpx.AsyncClient() as client:
        for f_id, file in zip(file_id, files):
            file_contents = await file.read()
            file_extension = file.filename.split('.')[-1].lower()

            file_result = {'id': f_id, 'detected_classes': {}, 'field_predictions': []}
            if file_extension in ['png', 'jpeg', 'jpg']:
                # Object Detection
                od_response = await client.post(retinanet_url, files={'data': ('filename', file_contents)})
                if od_response.status_code == 200:
                    od_data = process_torch_serve_response(od_response)
                    file_result['detected_classes'] = od_data['detected_classes']
                
                # Classification and Field Detection
                processed_image = efficientnet_classifier.preprocess_image(file_contents)
                is_handwritten = efficientnet_classifier.classify_image(processed_image)
                document_type = 'handwritten' if is_handwritten else 'machinewritten'
                target_url = LMv3_hand_url if is_handwritten else LMv3_machine_url
                field_response = await client.post(target_url, files={'data': ('filename', file_contents)})
                
                if field_response.status_code == 200:
                    labels = field_response.json()  # Directly using the list
                    unique_fields = {label[2:] if label.startswith(('B-', 'I-')) else label for label in labels if label != 'O'}
                    file_result['field_predictions'].extend(list(unique_fields))

            elif file_extension == 'pdf':
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_path = os.path.join(temp_dir, file.filename)
                    async with aiofiles.open(pdf_path, 'wb') as out_file:
                        await out_file.write(file_contents)
                    images = convert_pdf_to_png(pdf_path, temp_dir)
                    if images is None:
                        images = []  # Ensure images is always a list, even if empty
                    pdf_results = []
                    for image_filename in images:
                        image_path = os.path.join(temp_dir, image_filename)
                        async with aiofiles.open(image_path, 'rb') as image_file:
                            image_contents = await image_file.read()
                            # Object Detection
                            od_response = await client.post(retinanet_url, files={'data': ('filename', image_contents)})
                            if od_response.status_code == 200:
                                od_data = process_torch_serve_response(od_response)
                                # Classification and Field Detection
                                processed_image = efficientnet_classifier.preprocess_image(image_contents)
                                is_handwritten = efficientnet_classifier.classify_image(processed_image)
                                document_type = 'handwritten' if is_handwritten else 'machinewritten'
                                target_url = LMv3_hand_url if is_handwritten else LMv3_machine_url
                                field_response = await client.post(target_url, files={'data': ('filename', image_contents)})
                                if field_response.status_code == 200:
                                    labels = field_response.json()  # Directly using the list
                                    pdf_results.append({"unique_fields": [label[2:] if label.startswith(('B-', 'I-')) else label for label in labels if label != 'O']})
                    aggregate_results(file_result, pdf_results)

            results.append(file_result)

    return JSONResponse(content=results)
