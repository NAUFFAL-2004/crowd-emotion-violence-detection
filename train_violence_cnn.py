# train_violence_cnn.py

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

BASE_DIR = "data/violence"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
MODEL_PATH = "models/violence_model_cnn.h5"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

def build_violence_model() -> Sequential:
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128, (3,3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))  # binary: violent / non_violent

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)

    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )
    print("Violence train images:", train_gen.n, "val images:", val_gen.n)

    print("Violence classes:", train_gen.class_indices)  # {'non_violent': 0, 'violent': 1}

    model = build_violence_model()
    model.summary()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    model.save(MODEL_PATH)
    print(f"Violence model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
