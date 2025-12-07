# train_emotion_vgg16.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

BASE_DIR = "data/emotions"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
MODEL_PATH = "models/emotion_model_vgg16.h5"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10   # increase if you have GPU + more time
LEARNING_RATE = 1e-4

def build_emotion_model(num_classes: int) -> Model:
    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False  # freeze for transfer learning

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)

    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
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
        class_mode="categorical"
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )
    print("Emotion classes:", train_gen.class_indices)

# save mapping to json so we can reuse it in realtime script
    with open("models/emotion_classes.json", "w") as f:
       json.dump(train_gen.class_indices, f)

    print("Emotion train images:", train_gen.n, "val images:", val_gen.n)


    num_classes = len(train_gen.class_indices)
    print("Emotion classes:", train_gen.class_indices)

    model = build_emotion_model(num_classes)
    model.summary()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    model.save(MODEL_PATH)
    print(f"Emotion model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
