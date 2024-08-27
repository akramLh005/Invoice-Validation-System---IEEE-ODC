import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import JSONResponse
import requests


# Load the configuration
with open('config.json') as config_file:
    config = json.load(config_file)

TORCHSERVE_URL = config['LMv3_torch_serve_url']

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        # Send the image to TorchServe for inference
        response = requests.post(TORCHSERVE_URL, files={"data": image_bytes})
        response.raise_for_status()
        
        # Parse the response from TorchServe
        data = response.json()
        labels = data.get('labels', [])
        
        # Process labels to remove IOB tags and collect unique field names
        unique_fields = set()
        for label in labels:
            if label.startswith(('B-', 'I-')):
                normalized_label = label[2:]
            else:
                normalized_label = label
            unique_fields.add(normalized_label)

        unique_fields.discard('O')

        return JSONResponse(content={"fields": list(unique_fields)})

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=str(e))

