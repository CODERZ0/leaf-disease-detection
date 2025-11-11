# predict_demo.py
import json
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from argparse import ArgumentParser
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

parser = ArgumentParser()
parser.add_argument("--img", required=True, help="path to image file")
parser.add_argument("--model", default="models/leaf_model_finetuned.keras", help="path to model file (.h5 or .keras)")
parser.add_argument("--size", type=int, default=224, help="image size (square)")
args = parser.parse_args()

if not os.path.isfile(args.img):
    raise SystemExit(f"Image not found: {args.img}")
if not os.path.isfile(args.model):
    raise SystemExit(f"Model not found: {args.model}")
if not os.path.isfile("class_indices.json"):
    raise SystemExit("class_indices.json not found. Make sure you trained the model and saved class indices.")

print("Loading model...")
model = tf.keras.models.load_model(args.model)

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
inv_map = {int(v): k for k, v in class_indices.items()}

def load_image(path, size):
    img = image.load_img(path, target_size=(size, size))
    arr = image.img_to_array(img)
    arr = eff_preprocess(arr)   # EfficientNet preprocessing
    return np.expand_dims(arr, axis=0)

x = load_image(args.img, args.size)
pred = model.predict(x)[0]
idx = int(np.argmax(pred))
label = inv_map[idx]
conf = float(pred[idx])
print(f"Prediction: {label}  (confidence: {conf:.3f})")

# show top-5 predictions
top5 = np.argsort(pred)[::-1][:5]
print("\nTop-5 predictions:")
for i in top5:
    print(f"  {inv_map[int(i)]:<60} {pred[int(i)]:.4f}")
