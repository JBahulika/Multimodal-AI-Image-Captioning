from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import json
import torch

app = FastAPI(title="Multimodal AI Caption Generator")

# Allow frontend to call API locally or from a deployed site
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your caption dataset (optional use)
with open("captions_set_01.json", "r") as f:
    captions_data = json.load(f)

# Load pretrained model + processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the simple upload UI."""
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/generate_caption/")
async def generate_caption(file: UploadFile):
    """Accept image upload and return caption."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

        return JSONResponse({"caption": caption})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
def health_check():
    return {"status": "running"}
