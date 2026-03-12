import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import cv2
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Leaf Disease Detector", layout="centered")
st.title("🌿 Leaf Disease Detector")
st.write("Upload a leaf image OR scan using camera and the model will predict the disease/class (top-5 shown).")

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


# -----------------------------
# Upload OR Camera Input
# -----------------------------

uploaded = st.file_uploader("Upload a leaf image (jpg/png)", type=["jpg", "jpeg", "png"])
camera_photo = st.camera_input("Or scan a leaf using your camera")

image = None

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

elif camera_photo is not None:
    image = Image.open(camera_photo).convert("RGB")


# -----------------------------
# Prediction
# -----------------------------

if image is not None:

    st.image(image, caption="Leaf Image", use_column_width=True)

    # ---------- Leaf Validation (Green Detection) ----------
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / mask.size

    if green_ratio < 0.05:
        st.error("❌ Invalid image. Please upload a leaf image.")
        st.stop()

    # ---------- Model Prediction ----------
    size = (224, 224)
    img_resized = image.resize(size)

    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)

    with st.spinner("Predicting..."):
        preds = model.predict(x)
        probs = preds[0]

        max_prob = np.max(probs)

        # Confidence threshold
        if max_prob < 0.60:
            st.error("❌ This does not appear to be a leaf image.")
            st.stop()

        top_k = min(5, probs.shape[0])
        top_idx = np.argsort(probs)[-top_k:][::-1]

        st.subheader("Top Predictions:")

        for idx in top_idx:
            label = inv_map.get(int(idx), str(idx))
            st.write(f"- **{label}** — {probs[idx]:.3f}")

    st.success("Done ✅")

else:
    st.info("Upload a leaf image or use the camera to begin.")
