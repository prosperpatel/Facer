import cv2
import numpy as np
from PIL import Image
import os

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Face detector
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def getImagesAndLabels(path):
    imagePaths = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        try:
            # Convert image to grayscale
            pilImage = Image.open(imagePath).convert("L")
            imageNp = np.array(pilImage, "uint8")

            fileName = os.path.split(imagePath)[-1]

            parts = fileName.split(".")

            # Expected format:
            # User.<ID>.<Sample>.jpg
            if len(parts) < 4:
                print(f"Skipping invalid filename: {fileName}")
                continue

            try:
                userId = int(parts[1])
            except ValueError:
                print(f"Skipping invalid ID file: {fileName}")
                continue

            faces = detector.detectMultiScale(imageNp)

            if len(faces) == 0:
                print(f"No face detected: {fileName}")
                continue

            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y+h, x:x+w])
                ids.append(userId)

        except Exception as e:
            print(f"Error processing {imagePath}: {e}")

    return faceSamples, ids


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "dataset")
TRAINNER_PATH = os.path.join(BASE_DIR, "trainner")

os.makedirs(TRAINNER_PATH, exist_ok=True)

print("Training faces...")

faces, ids = getImagesAndLabels(DATASET_PATH)

if len(faces) == 0:
    raise Exception(
        "No valid faces found in dataset folder. "
        "Check image names and ensure faces are visible."
    )

recognizer.train(faces, np.array(ids))

MODEL_FILE = os.path.join(TRAINNER_PATH, "trainner.yml")

recognizer.write(MODEL_FILE)

print("\nTraining Complete!")
print(f"Faces trained: {len(faces)}")
print(f"Unique IDs: {len(set(ids))}")
print(f"Model saved to: {MODEL_FILE}")
