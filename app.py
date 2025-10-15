from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import json
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# --- Configuration ---
MODEL_ID = "openai/clip-vit-large-patch14"
CAPTIONS_PATH = "captions_set_01.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# --- Helper Functions ---

def _norm(s: str) -> str:
    """Normalize for dedupe: collapse spaces + lowercase."""
    return " ".join(s.split()).strip().casefold()

def load_captions(source: str) -> list[str]:
    """Load and deduplicate captions from a JSON file."""
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Captions file not found: {p.resolve()}")
    
    with p.open("r", encoding="utf-8") as f:
        captions = json.load(f)

    seen = set()
    unique = [c.strip() for c in captions if isinstance(c, str) and (key := _norm(c)) not in seen and not seen.add(key)]
    
    if not unique:
        raise ValueError("No captions loaded from file.")
        
    return unique

def compute_caption_embeddings(captions, clip_model, processor):
    """Compute CLIP text embeddings for all captions."""
    clip_model.eval()
    batch_size = 256
    all_text_features = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i+batch_size]
        text_inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(clip_model.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
        all_text_features.append(text_features.detach().cpu().numpy())
    return np.vstack(all_text_features)

def compute_image_embedding(image, clip_model, processor):
    """Compute CLIP image embedding for a PIL image."""
    clip_model.eval()
    inputs = processor(images=image, return_tensors="pt").to(clip_model.device)
    inputs['pixel_values'] = inputs['pixel_values'].to(clip_model.dtype)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.detach().cpu().numpy()

# --- FINAL ALGORITHM (Supporting Shuffle) ---
def get_caption_suggestions(image_features_np, caption_features_np, captions, top_k=15, pool_size=300, diversity_threshold=0.94):
    """
    Finds a large, relevant, and diverse pool of captions to send to the frontend for shuffling.
    """
    sims = cosine_similarity(image_features_np, caption_features_np)[0]
    
    sorted_relevant_indices = np.argsort(sims)[-pool_size:][::-1]
    
    if len(sorted_relevant_indices) == 0:
        return {"captions": []}

    final_indices = [sorted_relevant_indices[0]]
    for idx in sorted_relevant_indices[1:]:
        if len(final_indices) >= top_k:
            break
        current_embedding = caption_features_np[idx].reshape(1, -1)
        selected_embeddings = caption_features_np[final_indices]
        similarity_to_selected = cosine_similarity(current_embedding, selected_embeddings)[0]
        
        if np.max(similarity_to_selected) < diversity_threshold:
            final_indices.append(idx)

    results = []
    for idx in final_indices:
        results.append({"caption": captions[idx]})
        
    return {"captions": results}

# --- FastAPI Application ---
app = FastAPI(title="Multimodal AI Caption Generator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- One-Time Model and Data Loading ---
print("Server starting up...")
print(f"Loading CLIP model: {MODEL_ID} onto {DEVICE.upper()}")
model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=TORCH_DTYPE).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
print("Model loaded successfully.")

print(f"Loading captions from {CAPTIONS_PATH}...")
candidate_captions = load_captions(source=CAPTIONS_PATH)
print(f"Loaded {len(candidate_captions)} unique captions.")

print("Computing caption embeddings (this may take a moment)...")
caption_embeds = compute_caption_embeddings(candidate_captions, model, processor)
print(f"Computed {len(caption_embeds)} caption embeddings.")
print("--- Server is ready to accept requests ---")

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/generate_caption/")
async def generate_caption(file: UploadFile):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_embed = compute_image_embedding(image, model, processor)
        caption_results = get_caption_suggestions(image_embed, caption_embeds, candidate_captions)
        return JSONResponse(caption_results)
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)