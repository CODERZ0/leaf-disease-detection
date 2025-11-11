import os, json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ==============================
# SIMPLE FINE-TUNE CONFIG
# ==============================
MODEL_IN = "models/leaf_model.keras"        # existing model
MODEL_OUT = "models/leaf_model_finetuned.keras"
DATA_DIR = "data"
IMG = (224,224)
BATCH = 8
EPOCHS = 6

# Load and prepare data
train_aug = ImageDataGenerator(rescale=1./255, rotation_range=30, zoom_range=0.2,
                               shear_range=0.2, horizontal_flip=True)
val_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG, batch_size=BATCH, class_mode='categorical'
)
val_gen = val_aug.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG, batch_size=BATCH, class_mode='categorical', shuffle=False
)

# Load model
model = tf.keras.models.load_model(MODEL_IN)

# Unfreeze top 50 layers for fine-tuning
base = None
for layer in model.layers:
    if 'efficientnet' in layer.name:
        base = layer
        break
if base:
    for layer in base.layers:
        layer.trainable = True

# Compile with smaller learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Add callbacks
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-7),
    ModelCheckpoint(MODEL_OUT, save_best_only=True)
]

# Train again (fine-tuning)
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

print("\n✅ Fine-tuning complete. Model saved to", MODEL_OUT)
