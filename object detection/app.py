import os
import tempfile
from fastapi import FastAPI, File, Request, HTTPException, UploadFile
from starlette.middleware.base import BaseHTTPMiddleware
import httpx
import json
import aiofiles
from utils.pdf_to_png import convert_pdf_to_png

app = FastAPI()

class IPWhitelistMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, allowlist=None, allow_all=False):
        super().__init__(app)
        self.allowlist = allowlist or []
        self.allow_all = allow_all

    async def dispatch(self, request: Request, call_next):
        client_host = request.client.host
        if self.allow_all or client_host in self.allowlist:
            response = await call_next(request)
        else:
            raise HTTPException(status_code=403, detail="Access forbidden")
        return response

# Load the configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Determine if all IPs should be allowed based on the configuration
allow_all_ips = "*" in config['allowed_ips'] or "0.0.0.0" in config['allowed_ips']

# Initialize middleware with the list of allowed IPs from the config and the allow_all setting
app.add_middleware(IPWhitelistMiddleware, allowlist=config['allowed_ips'], allow_all=allow_all_ips)

# Access configuration values
threshold = config['threshold']
COLORS = config['colors']
CLASSES = config['classes']
torch_serve_url = config['retinanet_torch_serve_url']

def process_torch_serve_response(response):
    """
    Processes the JSON response from TorchServe and organizes detections by class.
    """
    detection_results = response.json()

    # Initialize response data structure
    response_data = {
        'detected_classes': {class_name: {'present': False, 'detections': []} for class_name in CLASSES if class_name != '__background__'}
    }

    # Process detections and organize by class
    for box, score, label in zip(detection_results['boxes'], detection_results['scores'], detection_results['labels']):
        class_name = CLASSES[label]
        if score >= threshold:
            response_data['detected_classes'][class_name]['present'] = True
            response_data['detected_classes'][class_name]['detections'].append({
                'box': box,
                'score': score
            })

    return response_data

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Receives a file (PNG, JPEG, or PDF), processes each image or PDF page for object detection, and returns a list of detection results.
    """
    file_extension = file.filename.split('.')[-1].lower()
    if file_extension in ['png', 'jpeg', 'jpg']:
        file_contents = await file.read()
        files = {'data': ('filename', file_contents)}
        results = []
        async with httpx.AsyncClient() as client:
            response = await client.post(torch_serve_url, files=files)
            if response.status_code == 200:
                processed_data = process_torch_serve_response(response)
                results.append(processed_data)
            else:
                results.append({'error': 'Failed to process image', 'filename': file.filename})
        return results
    elif file_extension == 'pdf':
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, file.filename)
            # Use aiofiles to write the PDF file asynchronously
            async with aiofiles.open(pdf_path, 'wb') as out_file:
                await out_file.write(await file.read())
            convert_pdf_to_png(pdf_path, temp_dir)  # Convert PDF to PNG
            results = []
            for image_filename in os.listdir(temp_dir):
                if image_filename.endswith('.png'):
                    image_path = os.path.join(temp_dir, image_filename)
                    with open(image_path, 'rb') as image_file:
                        files = {'data': ('filename', image_file.read())}
                        async with httpx.AsyncClient() as client:
                            response = await client.post(torch_serve_url, files=files)
                            if response.status_code == 200:
                                processed_data = process_torch_serve_response(response)
                                results.append(processed_data)
                            else:
                                results.append({'error': 'Failed to process image', 'filename': image_filename})
        return results
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PNG, JPEG, or PDF file.")
