import cv2
import numpy as np
import face_recognition
import firebase_admin
import requests
import os
from datetime import datetime
from firebase_admin import credentials, firestore, storage

# ===============================
# 🔥 FIREBASE SETUP
# ===============================

if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'fyp2-92b3f.firebasestorage.app'
    })

db = firestore.client()
bucket = storage.bucket()

print("✅ Firebase connected")

# ===============================
# 🔥 LOAD FAMILY MEMBERS
# ===============================

known_face_encodings = []
known_face_names = []

docs = db.collection("family_members").stream()

for doc in docs:
    data = doc.to_dict()
    name = data.get("name")
    image_url = data.get("image_url")

    if name and image_url:
        response = requests.get(image_url)
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        enc = face_recognition.face_encodings(rgb)
        if enc:
            known_face_encodings.append(enc[0])
            known_face_names.append(name)
            print("✔ Loaded", name)

# ===============================
# 📷 CAMERA SETUP (Lower = Faster)
# ===============================

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

print("📷 Camera started")

# ===============================
# PARAMETERS
# ===============================

TOLERANCE = 0.62
DETECT_EVERY = 3

frame_count = 0
last_face_locations = []
last_face_names = []

# ===============================
# MAIN LOOP
# ===============================

cv2.namedWindow("CCTV Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("CCTV Camera", 960, 540)

while True:
    ret, frame = camera.read()
    if not ret:
        continue

    frame_count += 1

    # Detect every few frames
    if frame_count % DETECT_EVERY == 0:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(
            rgb, face_locations
        )

        last_face_locations = face_locations
        last_face_names = []

        for face_encoding in face_encodings:

            name = "Stranger"

            if known_face_encodings:
                distances = face_recognition.face_distance(
                    known_face_encodings,
                    face_encoding
                )

                best_index = np.argmin(distances)

                print("Distance:", round(distances[best_index], 3))

                if distances[best_index] < TOLERANCE:
                    name = known_face_names[best_index]

            last_face_names.append(name)

    # Draw boxes (always)
    for (top, right, bottom, left), name in zip(
        last_face_locations, last_face_names
    ):
        cv2.rectangle(frame, (left, top),
                      (right, bottom), (0, 255, 0), 2)

        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("CCTV Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
