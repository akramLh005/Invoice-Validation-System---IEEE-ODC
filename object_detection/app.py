import asyncio
import os
import tempfile
from typing import List
from fastapi import FastAPI, File, Form, Request, HTTPException, UploadFile
from starlette.middleware.base import BaseHTTPMiddleware
import httpx
import json
import aiofiles
from utils import *
from utils.postprocessing import process_torch_serve_response, aggregate_results
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

torch_serve_url = config['retinanet_torch_serve_url']

@app.post("/detect")
async def detect_objects(file_id: List[str] = Form(...), files: List[UploadFile] = File(...)):
    """
    Receives files (PNG, JPEG, or PDF) and their IDs, processes each image or PDF page for object detection,
    and returns a list of detection results as JSON objects, each associated with the given file ID.
    """
    results = []
    for file_id, file in zip(file_id, files):
        file_extension = file.filename.split('.')[-1].lower()
        file_result = {'id': file_id, 'detected_classes': {}}

        if file_extension in ['png', 'jpeg', 'jpg']:
            file_contents = await file.read()
            files = {'data': ('filename', file_contents)}
            async with httpx.AsyncClient() as client:
                response = await client.post(torch_serve_url, files=files)
                if response.status_code == 200:
                    processed_data = process_torch_serve_response(response)
                    file_result['detected_classes'].update(processed_data)
                else:
                    file_result['error'] = 'Failed to process image'
            results.append(file_result)
        
        elif file_extension == 'pdf':
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_path = os.path.join(temp_dir, file.filename)
                async with aiofiles.open(pdf_path, 'wb') as out_file:
                    await out_file.write(await file.read())
                convert_pdf_to_png(pdf_path, temp_dir)  # Convert PDF to PNG
                pdf_results = []
                for image_filename in os.listdir(temp_dir):
                    if image_filename.endswith('.png'):
                        image_path = os.path.join(temp_dir, image_filename)
                        with open(image_path, 'rb') as image_file:
                            files = {'data': ('filename', image_file.read())}
                            async with httpx.AsyncClient() as client:
                                response = await client.post(torch_serve_url, files=files)
                                if response.status_code == 200:
                                    processed_data = process_torch_serve_response(response)
                                    pdf_results.append(processed_data)
                                else:
                                    pdf_results.append({'error': 'Failed to process image', 'filename': image_filename})
                # Aggregate results for the entire PDF file
                aggregate_results(file_result, pdf_results)
            results.append(file_result)

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

    return results

# async def send_batch_request(batch):
#     """Send a batch of image requests and process the responses."""
#     async with httpx.AsyncClient() as client:
#         responses = await asyncio.gather(*[
#             client.post(torch_serve_url, files={'data': ('filename', image.read())}) 
#             for image in batch
#         ])
#     return [process_torch_serve_response(response) if response.status_code == 200 else {'error': 'Failed to process image'}
#             for response in responses]

# @app.post("/2detect")
# async def detect_objects(file_id: List[str] = Form(...), files: List[UploadFile] = File(...)):
#     results = []
#     batch = []
#     for file_id, file in zip(file_ids, files):
#         file_extension = file.filename.split('.')[-1].lower()
#         file_result = {'id': file_id, 'detected_classes': {}}

#         if file_extension in ['png', 'jpeg', 'jpg']:
#             file_contents = await file.read()
#             batch.append(('filename', file_contents))
#             if len(batch) == 4:
#                 file_results = await send_batch_request(batch)
#                 results.extend(file_results)
#                 batch = []
        
#         elif file_extension == 'pdf':
#             with tempfile.TemporaryDirectory() as temp_dir:
#                 pdf_path = os.path.join(temp_dir, file.filename)
#                 async with aiofiles.open(pdf_path, 'wb') as out_file:
#                     await out_file.write(await file.read())
#                 convert_pdf_to_png(pdf_path, temp_dir)  # Convert PDF to PNG
#                 pdf_results = []
#                 pdf_images = []
#                 for image_filename in os.listdir(temp_dir):
#                     if image_filename.endswith('.png'):
#                         with open(os.path.join(temp_dir, image_filename), 'rb') as image_file:
#                             pdf_images.append(image_file)
#                             if len(pdf_images) == 4:
#                                 pdf_results.extend(await send_batch_request(pdf_images))
#                                 pdf_images = []
#                 if pdf_images:
#                     pdf_results.extend(await send_batch_request(pdf_images))
#                 aggregate_results(file_result, pdf_results)
#             results.append(file_result)

#         else:
#             raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

#     if batch:  # Check if there is a residual batch to send
#         results.extend(await send_batch_request(batch))

#     return results


