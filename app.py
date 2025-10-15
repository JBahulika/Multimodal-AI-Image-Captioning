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
import random
from pathlib import Path

# --- Configuration ---
MODEL_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
CAPTIONS_PATH = "captions_set_01.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# --- Helper Functions (from your Notebook) ---

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
        image_features = clip_model.get_image_features(**inputs)
    return image_features.detach().cpu().numpy()

def pick_diverse_captions(image_features_np, caption_features_np, captions, top_k=5, pool_size=100, diversity_threshold=0.96):
    """
    Selects a randomized, diverse set of top_k captions.
    ACCURACY BOOST: Increased pool_size to 100 for more relevant initial candidates.
    """
    sims = cosine_similarity(image_features_np, caption_features_np)[0]
    
    initial_indices = np.argsort(sims)[-pool_size:][::-1]
    
    if len(initial_indices) == 0:
        return {"best_caption": "No matching captions found.", "top_captions": []}

    diverse_indices = [initial_indices[0]]
    remaining_indices = list(initial_indices[1:])
    random.shuffle(remaining_indices)
    
    for idx in remaining_indices:
        if len(diverse_indices) >= top_k:
            break
        current_embedding = caption_features_np[idx].reshape(1, -1)
        selected_embeddings = caption_features_np[diverse_indices]
        similarity_to_selected = cosine_similarity(current_embedding, selected_embeddings)[0]
        
        if np.max(similarity_to_selected) < diversity_threshold:
            diverse_indices.append(idx)

    final_scores = sims[diverse_indices]
    sorted_order = np.argsort(final_scores)[::-1]
    final_diverse_indices = np.array(diverse_indices)[sorted_order]
    
    results = []
    for idx in final_diverse_indices:
        results.append({
            "caption": captions[idx],
            "score": float(sims[idx])
        })
        
    return {
        "best_caption": results[0] if results else None,
        "other_suggestions": results[1:]
    }

# --- FastAPI Application ---
app = FastAPI(title="Multimodal AI Caption Generator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- One-Time Model and Data Loading ---
print("Server starting up...")
print(f"Loading CLIP model: {MODEL_ID} onto {DEVICE.upper()}")
clip_model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=TORCH_DTYPE).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
print("Model loaded successfully.")

print(f"Loading captions from {CAPTIONS_PATH}...")
candidate_captions = load_captions(source=CAPTIONS_PATH)
print(f"Loaded {len(candidate_captions)} unique captions.")

print("Computing caption embeddings (this may take a moment)...")
caption_embeds = compute_caption_embeddings(candidate_captions, clip_model, processor)
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
        image_embed = compute_image_embedding(image, clip_model, processor)
        caption_results = pick_diverse_captions(image_embed, caption_embeds, candidate_captions)
        return JSONResponse(caption_results)
    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)