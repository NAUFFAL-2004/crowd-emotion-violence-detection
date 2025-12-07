import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque

# ---------- Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMOTION_MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model_vgg16.h5")
VIOLENCE_MODEL_PATH = os.path.join(BASE_DIR, "models", "violence_model_cnn.h5")
FACE_CASCADE_PATH = os.path.join(BASE_DIR, "haarcascades", "haarcascade_frontalface_default.xml")
EMOTION_CLASSES_PATH = os.path.join(BASE_DIR, "models", "emotion_classes.json")

# ---------- Parameters ----------
EMOTION_IMG_SIZE = (224, 224)
VIOLENCE_IMG_SIZE = (128, 128)

VIOLENCE_THRESHOLD = 0.8     # higher = fewer false alarms
FRAME_SKIP = 1     
     # process every 3rd frame
SMOOTHING_FRAMES = 10       # temporal smoothing window

# store recent violence probabilities
violence_history = deque(maxlen=SMOOTHING_FRAMES)

# ---------- Load Models Safely ----------
def load_models():
    print("Emotion model path:", EMOTION_MODEL_PATH)
    print("Violence model path:", VIOLENCE_MODEL_PATH)
    print("Face cascade path:", FACE_CASCADE_PATH)

    if not os.path.exists(EMOTION_MODEL_PATH):
        raise FileNotFoundError(f"Emotion model not found: {EMOTION_MODEL_PATH}")
    if not os.path.exists(VIOLENCE_MODEL_PATH):
        raise FileNotFoundError(f"Violence model not found: {VIOLENCE_MODEL_PATH}")
    if not os.path.exists(FACE_CASCADE_PATH):
        raise FileNotFoundError(f"Face cascade not found: {FACE_CASCADE_PATH}")

    emotion_model = load_model(EMOTION_MODEL_PATH)
    violence_model = load_model(VIOLENCE_MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haarcascade XML file.")

    # ----- Load emotion class mapping safely -----
    if os.path.exists(EMOTION_CLASSES_PATH):
        with open(EMOTION_CLASSES_PATH, "r") as f:
            class_dict = json.load(f)
        emotion_classes = [None] * len(class_dict)
        for label, idx in class_dict.items():
            emotion_classes[idx] = label
    else:
        # fallback if json not present
        emotion_classes = ["angry", "happy", "neutral", "sad", "scared"]

    print("Emotion classes:", emotion_classes)
    return emotion_model, violence_model, face_cascade, emotion_classes


# ---------- Preprocessing ----------
def preprocess_for_emotion(face_img):
    face_resized = cv2.resize(face_img, EMOTION_IMG_SIZE)
    face_resized = face_resized.astype("float32") / 255.0
    return np.expand_dims(face_resized, axis=0)


def preprocess_for_violence(frame):
    img = cv2.resize(frame, VIOLENCE_IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


# ---------- Main Loop ----------
def main():
    emotion_model, violence_model, face_cascade, emotion_classes = load_models()

    # Webcam or CCTV URL
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed. Exiting...")
            break

        frame_count += 1
        display_frame = frame.copy()

        # Only process every Nth frame (boost FPS)
        if frame_count % FRAME_SKIP == 0:

            # ---------- Violence Prediction ----------
            v_input = preprocess_for_violence(frame)
            violence_prob = float(violence_model.predict(v_input, verbose=0)[0][0])

            violence_history.append(violence_prob)
            avg_violence = sum(violence_history) / len(violence_history)
            is_violent = avg_violence >= VIOLENCE_THRESHOLD

            # ---------- Face Detection ----------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,          # good balance: sensitivity vs speed
                    minNeighbors=3,           # a bit more relaxed → more faces found
                    minSize=(40, 40),         # allow smaller faces (people a bit far)
                    flags=cv2.CASCADE_SCALE_IMAGE
            )
            print("Faces detected this frame:", len(faces))


            for (x, y, w, h) in faces:
                face_color = frame[y:y+h, x:x+w]

                # Emotion prediction
                e_input = preprocess_for_emotion(face_color)
                preds = emotion_model.predict(e_input, verbose=0)[0]

                emotion_idx = int(np.argmax(preds))
                emotion_label = emotion_classes[emotion_idx]
                emotion_score = float(np.max(preds))

                # Draw face bounding box
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Emotion label
                text = f"{emotion_label} ({emotion_score:.2f})"
                cv2.putText(display_frame, text,
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 0), 2)

            # ---------- Violence Banner ----------
            label = f"Violence: {avg_violence:.2f}"
            color = (0, 0, 255) if is_violent else (0, 255, 0)

            cv2.putText(display_frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color, 2)

            if is_violent:
                cv2.putText(display_frame, "!!! VIOLENCE DETECTED !!!",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 3)

        # ---------- Display ----------
        cv2.imshow("Crowd Emotion & Violence Detection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

