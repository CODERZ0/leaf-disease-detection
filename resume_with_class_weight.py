# resume_with_class_weight.py
import os, json, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

MODEL_IN = "models/leaf_model_finetuned.keras"   # use your fine-tuned model
MODEL_OUT = "models/leaf_model_classweighted.keras"
DATA_DIR = "data"
IMG = (224,224)
BATCH = 8      # reduce to 4 if you get memory errors
EPOCHS = 8

# Load class weights
if not os.path.isfile("class_weight.json"):
    raise SystemExit("class_weight.json not found. Run compute_class_weights.py first.")
with open("class_weight.json") as f:
    class_weight = json.load(f)
print("Loaded class_weight (sample):", dict(list(class_weight.items())[:6]))

# Data generators
train_aug = ImageDataGenerator(rescale=1./255, rotation_range=30, zoom_range=0.2,
                               shear_range=0.2, horizontal_flip=True)
val_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG,
    batch_size=BATCH,
    class_mode='categorical'
)

val_gen = val_aug.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG,
    batch_size=BATCH,
    class_mode='categorical',
    shuffle=False
)

# Load model and unfreeze all layers
model = tf.keras.models.load_model(MODEL_IN)
for layer in model.layers:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-7),
    ModelCheckpoint(MODEL_OUT, save_best_only=True)
]

print("\n🚀 Starting fine-tuning with class weights...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight
)

print("\n✅ Training finished. Model saved to", MODEL_OUT)
