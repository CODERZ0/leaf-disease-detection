import os, json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ---------------------- CONFIG ----------------------
DATA_DIR = "data"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16        # reduce to 8 if memory low
EPOCHS = 10
MODEL_OUT = "models/leaf_model.h5"
# ----------------------------------------------------

def build_generators():
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')

    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True
    )
    val_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(
        train_dir, target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, class_mode='categorical'
    )
    val_gen = val_aug.flow_from_directory(
        val_dir, target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, class_mode='categorical',
        shuffle=False
    )
    return train_gen, val_gen

def build_model(num_classes):
    base = EfficientNetB0(weights='imagenet', include_top=False,
                          input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.4)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(base.input, out)
    for layer in base.layers:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError("Data folder not found. Run split_dataset.py first!")

    train_gen, val_gen = build_generators()
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    with open('class_indices.json', 'w') as f:
        json.dump(train_gen.class_indices, f)

    model = build_model(train_gen.num_classes)
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(MODEL_OUT, save_best_only=True)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("✅ Training complete. Model saved to", MODEL_OUT)
