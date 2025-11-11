# evaluate_and_report.py
import os, json, numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "models/leaf_model.keras"  # or .h5 if you prefer
IMG = (224,224)
BATCH = 32
DATA_DIR = "data"

if not os.path.isfile(MODEL_PATH):
    raise SystemExit(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
with open("class_indices.json") as f:
    class_indices = json.load(f)
inv = {int(v):k for k,v in class_indices.items()}
labels = [inv[i] for i in sorted(inv.keys())]

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(DATA_DIR,"test"), target_size=IMG, batch_size=BATCH,
    class_mode='categorical', shuffle=False
)

y_true = test_gen.classes
preds = model.predict(test_gen, verbose=1)
y_pred = preds.argmax(axis=1)

print("\nClassification report:\n")
print(classification_report(y_true, y_pred, target_names=labels))
cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix shape:", cm.shape)
