$code = @'
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Leaf Disease Detector", layout="centered")
st.title("🌿 Leaf Disease Detector")
st.write("Upload a leaf image and the model will predict the disease/class (top-5 shown).")

# HF model info (edit if you used a different repo/filename)
MODEL_REPO = "CODERZ0/leaf-disease-model"
MODEL_FILENAME = "leaf_model_classweighted.keras"
LOCAL_MODEL_PATH = os.path.join("models", MODEL_FILENAME)

@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    # 1) try local model first
    if os.path.exists(LOCAL_MODEL_PATH):
        model_path = LOCAL_MODEL_PATH
    else:
        # 2) else try to download from HF hub (will be cached)
        hf_token = None
        try:
            hf_token = st.secrets["HF_TOKEN"]
        except Exception:
            hf_token = None
        if hf_token:
            model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, token=hf_token)
        else:
            model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)

    # load model
    model = tf.keras.models.load_model(model_path)

    # load label mapping (class_indices.json) which maps label->index
    with open("class_indices.json", "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    # invert to index -> label (ensure keys are ints)
    inv_map = {int(v): k for k, v in class_indices.items()}
    return model, inv_map

# load (cached)
with st.spinner("Loading model (this happens once)..."):
    try:
        model, inv_map = load_model_and_labels()
    except Exception as e:
        st.error("Failed to load model. See details below.")
        st.exception(e)
        st.stop()

# UI: uploader
uploaded = st.file_uploader("Upload a leaf image (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    try:
        img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error("Couldn't open the uploaded file as an image.")
        st.exception(e)
        st.stop()

    st.image(img, caption="Uploaded image", use_column_width=True)

    # Preprocess for model (resize to 224x224 - change if your model expects other)
    size = (224, 224)
    img_resized = img.resize(size)
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)

    with st.spinner("Running model inference..."):
        preds = model.predict(x)  # shape (1, N)
        probs = preds[0]
        # get top-5
        top_k = min(5, probs.shape[0])
        top_idx = np.argsort(probs)[-top_k:][::-1]
        st.subheader("Top predictions:")
        for idx in top_idx:
            label = inv_map.get(int(idx), str(idx))
            confidence = float(probs[int(idx)])
            st.write(f"- **{label}** — {confidence:.3f}")

    st.success("Done")
else:
    st.info("Upload an image to get predictions. You can also run the `predict_demo.py` script locally.")
'@

$code | Set-Content -Path .\app.py -Encoding UTF8
Write-Host "app.py overwritten with improved Streamlit UI."
