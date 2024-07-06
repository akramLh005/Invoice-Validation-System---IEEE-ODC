from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io,time
from TR_invoice_model import InvoiceModel
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'




app = FastAPI()
model = InvoiceModel("./checkpoint-2000-focalloss-azure", "microsoft/layoutlmv3-base", True)

@app.post("/")
async def Hello():
    return "hello World"

@app.post("/test")
async def predict(file: UploadFile = File(...)):
    return {"filename": file.filename}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    start_time = time.time()

    labels, boxes = model.run_inference(image)
    inference_time = time.time() - start_time

    response_data = {"inference time in seconds ": inference_time}
    return JSONResponse(content=response_data)


@app.post("/predict2")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    labels, boxes = model.run_inference(image)

    # Process labels to remove IOB tags and collect unique field names
    unique_fields = set()
    for label in labels:
        # Normalize label by removing IOB-prefix if present ('B-', 'I-')
        normalized_label = label[2:] if label.startswith(('B-', 'I-')) else label
        unique_fields.add(normalized_label)

    return JSONResponse(content={"fields": list(unique_fields)})