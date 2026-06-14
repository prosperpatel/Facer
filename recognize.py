import cv2
import os

# Load face detector
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Create recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trainner", "trainner.yml")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

recognizer.read(MODEL_PATH)

# Student ID -> Name mapping
names = {
    1: "Prosper",
    2: "Student2",
    3: "Student3"
}

# Start camera
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise Exception("Unable to access webcam")

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, img = cam.read()

    if not ret:
        print("Failed to capture frame")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face_roi = gray[y:y+h, x:x+w]

        try:
            student_id, confidence = recognizer.predict(face_roi)

            # Lower confidence = better match
            if confidence < 60:
                name = names.get(student_id, f"ID {student_id}")
            else:
                name = "Unknown"

            cv2.putText(
                img,
                f"{name}",
                (x, y - 10),
                font,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                img,
                f"Conf: {confidence:.2f}",
                (x, y + h + 25),
                font,
                0.5,
                (255, 255, 0),
                1
            )

        except Exception as e:
            print("Recognition error:", e)

    cv2.imshow("Face Recognition", img)

    key = cv2.waitKey(1) & 0xFF

    # Press Q to quit
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
