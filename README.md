# Facer - Face Recognition Attendance System

A simple Face Recognition System built using Python and OpenCV. This project allows users to:

* Capture face samples using a webcam
* Train a face recognition model
* Recognize registered users in real time

The project uses the LBPH (Local Binary Patterns Histograms) face recognition algorithm provided by OpenCV.

---

# Features

* Real-time face detection
* Face dataset collection
* Face model training
* Face recognition using webcam
* Multiple user support
* Python 3 compatible
* OpenCV 4 compatible

---

# Project Structure

```text
Facer/
│
├── dataset/
│   ├── User.1.1.jpg
│   ├── User.1.2.jpg
│   └── ...
│
├── trainner/
│   └── trainner.yml
│
├── sample.py
├── train.py
├── recognize.py
└── README.md
```

---

# Requirements

## Software

* Python 3.9 or higher
* Webcam

## Python Packages

Install dependencies:

```bash
pip install opencv-contrib-python pillow numpy
```

Verify installation:

```bash
python -c "import cv2; print(cv2.__version__)"
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Facer.git
cd Facer
```

Or download the ZIP and extract it.

---

# Step 1 - Capture Face Samples

Run:

```bash
python sample.py
```

Enter a numeric User ID:

```text
Enter User ID: 1
```

The camera will open and automatically capture face samples.

Example generated files:

```text
dataset/
├── User.1.1.jpg
├── User.1.2.jpg
├── User.1.3.jpg
...
├── User.1.50.jpg
```

Recommended:

* Look straight at the camera
* Turn your head slightly left and right
* Use good lighting
* Capture at least 50 samples

---

# Step 2 - Train Model

Run:

```bash
python train.py
```

Expected output:

```text
Training Complete!
Faces trained: 50
Unique IDs: 1
Model saved to: trainner/trainner.yml
```

This creates:

```text
trainner/trainner.yml
```

which contains the trained face recognition model.

---

# Step 3 - Configure Names

Open:

```python
recognize.py
```

Edit:

```python
names = {
    1: "Prosper",
    2: "John",
    3: "Alice"
}
```

Replace names with your users.

Example:

```python
names = {
    1: "Prosper Patel",
    2: "Ritika",
    3: "Teacher"
}
```

---

# Step 4 - Start Face Recognition

Run:

```bash
python recognize.py
```

The webcam will open.

When a known face is detected:

```text
Prosper Patel
Conf: 42.31
```

will appear on the screen.

Unknown faces will be displayed as:

```text
Unknown
```

Press:

```text
q
```

to exit.

---

# Adding More Users

Capture samples using a new ID.

Example:

```text
User 1 = Prosper
User 2 = John
User 3 = Alice
```

Run:

```bash
python sample.py
```

Enter:

```text
2
```

Capture samples.

Then retrain:

```bash
python train.py
```

Update:

```python
names = {
    1: "Prosper",
    2: "John",
    3: "Alice"
}
```

in `recognize.py`.

---

# Troubleshooting

## Error: module 'cv2' has no attribute 'face'

Install:

```bash
pip install opencv-contrib-python
```

Remove regular OpenCV if necessary:

```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

---

## Camera Not Opening

Check:

* Webcam permissions
* Webcam not being used by another application
* Correct camera index

Try:

```python
cv2.VideoCapture(1)
```

instead of:

```python
cv2.VideoCapture(0)
```

if multiple cameras exist.

---

## No Faces Detected

Ensure:

* Good lighting
* Face clearly visible
* Camera focused
* Haar Cascade loaded correctly

---

## Training Fails

Verify dataset contains files like:

```text
User.1.1.jpg
User.1.2.jpg
User.1.3.jpg
```

Do NOT use:

```text
User..1.jpg
User..2.jpg
```

---

# Future Improvements

* Attendance logging
* CSV attendance reports
* MySQL database integration
* Student management dashboard
* Flask web application
* School attendance system
* Face recognition API

---

# Technology Stack

* Python
* OpenCV
* NumPy
* Pillow
* Haar Cascade Face Detection
* LBPH Face Recognition

---

# License

This project is provided for educational and learning purposes. Feel free to modify and extend it according to your requirements.
