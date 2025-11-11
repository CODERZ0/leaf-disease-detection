# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

st.set_page_config(page_title="Leaf Disease Detector", layout="centered")

MODEL_PATH = "models/leaf_model_classweighted.keras"
IMG_SIZE = 224

@st.cache_resource
def load_model_and_labels():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    inv_map = {int(v): k for k, v in class_indices.items()}
    return model, inv_map

def preprocess_pil(pil_image, size=IMG_SIZE):
    img = pil_image.resize((size, size))
    arr = image.img_to_array(img)
    arr = eff_preprocess(arr)   # EfficientNet preprocessing
    return np.expand_dims(arr, axis=0)

st.title("🌿 Leaf Disease Detector")
st.write("Upload a leaf image and the model will predict the disease/class (top-5 shown).")

try:
    model, inv_map = load_model_and_labels()
except Exception as e:
    st.error(str(e))
    st.stop()

uploaded = st.file_uploader("Upload a leaf image", type=["jpg","jpeg","png"])
if uploaded:
    pil = image.load_img(uploaded)
    st.image(pil, caption="Uploaded image", use_column_width=True)
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            x = preprocess_pil(pil, IMG_SIZE)
            preds = model.predict(x)[0]
            top5 = np.argsort(preds)[::-1][:5]
            st.markdown("**Top predictions:**")
            for i in top5:
                label = inv_map[int(i)]
                prob = float(preds[int(i)])
                st.write(f"- **{label}** — {prob:.3f}")
        st.success("Done")
