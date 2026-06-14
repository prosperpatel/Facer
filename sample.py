import cv2
import os

# Load face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Get user ID
user_id = input("Enter User ID: ").strip()

if not user_id.isdigit():
    raise ValueError("User ID must be a number.")

# Create dataset folder if it doesn't exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

os.makedirs(DATASET_PATH, exist_ok=True)

# Start camera
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise Exception("Could not access webcam.")

print("\nLooking at camera...")
print("Press 'q' anytime to stop.\n")

sample_num = 0

while True:
    ret, img = cam.read()

    if not ret:
        print("Failed to capture frame.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        sample_num += 1

        face_img = gray[y:y+h, x:x+w]

        file_name = os.path.join(
            DATASET_PATH,
            f"User.{user_id}.{sample_num}.jpg"
        )

        cv2.imwrite(file_name, face_img)

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            img,
            f"Samples: {sample_num}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Capture", img)

    key = cv2.waitKey(100) & 0xFF

    # Stop after 50 images
    if sample_num >= 50:
        break

    # Press Q to quit manually
    if key == ord('q'):
        break

print(f"\nCollected {sample_num} face samples.")

cam.release()
cv2.destroyAllWindows()

print("Dataset creation completed.")
