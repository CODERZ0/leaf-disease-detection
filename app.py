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

MODEL_REPO = "CODERZ0/leaf-disease-model"
MODEL_FILENAME = "leaf_model_classweighted.keras"
LOCAL_MODEL_PATH = os.path.join("models", MODEL_FILENAME)

@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    if os.path.exists(LOCAL_MODEL_PATH):
        model_path = LOCAL_MODEL_PATH
    else:
        hf_token = None
        try:
            hf_token = st.secrets["HF_TOKEN"]
        except Exception:
            hf_token = None
        if hf_token:
            model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, token=hf_token)
        else:
            model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)

    model = tf.keras.models.load_model(model_path)

    with open("class_indices.json", "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    inv_map = {int(v): k for k, v in class_indices.items()}
    return model, inv_map

with st.spinner("Loading model (this happens once)..."):
    try:
        model, inv_map = load_model_and_labels()
    except Exception as e:
        st.error("Failed to load model.")
        st.exception(e)
        st.stop()

uploaded = st.file_uploader("Upload a leaf image (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded:
    try:
        img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error("Couldn't open the image.")
        st.exception(e)
        st.stop()

    st.image(img, caption="Uploaded image", use_column_width=True)
    size = (224, 224)
    img_resized = img.resize(size)
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)

    with st.spinner("Predicting..."):
        preds = model.predict(x)
        probs = preds[0]
        top_k = min(5, probs.shape[0])
        top_idx = np.argsort(probs)[-top_k:][::-1]

        st.subheader("Top Predictions:")
        for idx in top_idx:
            label = inv_map.get(int(idx), str(idx))
            st.write(f"- **{label}** — {probs[idx]:.3f}")
    st.success("Done ✅")
else:
    st.info("Upload a leaf image to begin.")
