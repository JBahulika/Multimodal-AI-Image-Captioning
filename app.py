import streamlit as st
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import random
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
import json

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="AI Image Captioner",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Model and Processor Loading (with Caching)
# ==============================================================================
@st.cache_resource
def load_model():
    """Load the CLIP model and processor."""
    model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 🔑 FIX: force float32 on CPU (Streamlit Cloud)
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch_dtype).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    return model, processor, device

# ==============================================================================
# Caption Loading and Embedding (with Caching)
# ==============================================================================
@st.cache_data
def load_and_embed_captions(_model, _processor, captions_path="captions_set_01.json"):
    """Load captions from file and compute their embeddings."""
    def _norm(s: str) -> str:
        return " ".join(s.split()).strip().casefold()

    p = Path(captions_path)
    if not p.exists():
        st.error(f"Caption file not found at: {p.resolve()}")
        return None, None
    with p.open("r", encoding="utf-8") as f:
        captions_data = json.load(f)

    seen = set()
    unique_captions = []
    for c in captions_data:
        if not isinstance(c, str):
            continue
        key = _norm(c)
        if key not in seen:
            seen.add(key)
            unique_captions.append(c.strip())

    text_inputs = _processor(
        text=unique_captions,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    text_inputs = {k: v.to(_model.device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_features = _model.get_text_features(**text_inputs)

    return unique_captions, text_features.detach().cpu().numpy()

# ==============================================================================
# Core Functions
# ==============================================================================
def compute_image_embedding(image, model, processor):
    """Compute CLIP image embedding for a PIL image."""
    model.eval()
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    # 🔑 FIX: respect dtype from model
    inputs['pixel_values'] = inputs['pixel_values'].to(model.dtype)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    return image_features.detach().cpu().numpy()

def pick_diverse_caption(image_embed, caption_embeds, captions, top_k, temp, pool_size, diversity_thresh):
    """Selects a randomized, diverse set of captions."""
    sims = cosine_similarity(image_embed, caption_embeds)[0]

    initial_indices = np.argsort(sims)[-pool_size:][::-1]
    if len(initial_indices) == 0:
        return "No captions found", [], []

    diverse_indices = [initial_indices[0]]
    remaining_indices = list(initial_indices[1:])
    random.shuffle(remaining_indices)

    for idx in remaining_indices:
        if len(diverse_indices) >= top_k:
            break
        current_embedding = caption_embeds[idx].reshape(1, -1)
        selected_embeddings = caption_embeds[diverse_indices]
        similarity_to_selected = cosine_similarity(current_embedding, selected_embeddings)[0]
        if np.max(similarity_to_selected) < diversity_thresh:
            diverse_indices.append(idx)

    final_scores = sims[diverse_indices]
    sorted_order = np.argsort(final_scores)[::-1]
    final_diverse_indices = np.array(diverse_indices)[sorted_order]
    final_diverse_scores = final_scores[sorted_order]

    if len(final_diverse_indices) == 0:
        return "Could not find any diverse captions.", [], []

    scaled_scores = final_diverse_scores / temp
    scaled_scores -= np.max(scaled_scores)
    exp_scores = np.exp(scaled_scores)
    probs = exp_scores / np.sum(exp_scores)

    chosen_idx = np.random.choice(final_diverse_indices, p=probs)

    return captions[chosen_idx], final_diverse_indices.tolist(), final_diverse_scores.tolist()

# ==============================================================================
# Streamlit UI
# ==============================================================================
st.title("📸 AI Image Captioner")
st.markdown("Upload an image and our AI will find the most creative and diverse captions for it from our library. Adjust the settings in the sidebar for different results!")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Caption Controls")

    temp = st.slider(
        "Temperature (Creativity)", 
        min_value=0.1, max_value=3.0, value=1.5, step=0.1,
        help="Higher values lead to more surprising and creative captions."
    )

    top_k = st.slider(
        "Number of Diverse Candidates",
        min_value=3, max_value=10, value=5,
        help="How many diverse options to generate before the final choice."
    )

    diversity_thresh = st.slider(
        "Diversity Threshold",
        min_value=0.80, max_value=1.0, value=0.97, step=0.01,
        help="Lower values force the candidates to be more different from each other."
    )

    pool_size = st.slider(
        "Initial Candidate Pool",
        min_value=10, max_value=50, value=30,
        help="The number of top captions to consider for diversity filtering."
    )

# --- Main App Logic ---
with st.spinner("Warming up the AI... This might take a moment."):
    model, processor, device = load_model()
    captions, caption_embeds = load_and_embed_captions(model, processor)

st.success("AI is ready! Please upload an image.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Image")
        st.image(image, use_column_width=True)

    with st.spinner("The AI is thinking... 🧠"):
        image_embed = compute_image_embedding(image, model, processor)
        chosen_caption, top_indices, top_scores = pick_diverse_caption(
            image_embed, caption_embeds, captions, top_k, temp, pool_size, diversity_thresh
        )

    with col2:
        st.subheader("AI's Choice")
        st.markdown(f"""
        <div style="border: 2px solid #28a745; border-radius: 5px; padding: 10px; background-color: #e9f5ec;">
            <p style="font-size: 20px; font-weight: bold; color: #155724;">{chosen_caption}</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader(f"Top {top_k} Diverse Candidates")
        diverse_captions = [captions[i] for i in top_indices]
        for i, (caption, score) in enumerate(zip(diverse_captions, top_scores)):
            st.info(f"**{i+1}.** {caption} *(Similarity: {score:.2f})*")
