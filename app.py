# ----- start: model load snippet -----
import streamlit as st
import tensorflow as tf
import json
from huggingface_hub import hf_hub_download
import os

# Change these values to match your HF model repo/file
MODEL_REPO = "CODERZ0/leaf-disease-model"
MODEL_FILENAME = "leaf_model_classweighted.keras"

@st.cache_resource(show_spinner=False)
def load_model_and_labels():
    # If you set HF_TOKEN in Streamlit secrets (for private repo), use it:
    hf_token = None
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        hf_token = None

    # Download model (huggingface-hub will cache it)
    if hf_token:
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, token=hf_token)
    else:
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)

    # Load the Keras model
    model = tf.keras.models.load_model(model_path)

    # Load labels from local file (class_indices.json kept in repo)
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    inv_map = {int(v): k for k, v in class_indices.items()}
    return model, inv_map

# call once at start
model, inv_map = load_model_and_labels()
# ----- end snippet -----
