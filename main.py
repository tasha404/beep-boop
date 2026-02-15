import cv2
import numpy as np
import face_recognition
import firebase_admin
import requests
import os
from datetime import datetime
from firebase_admin import credentials, firestore, storage
from flask import Flask, Response
import threading

# =====================================
# 🔥 FIREBASE SETUP
# =====================================

if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'fyp2-92b3f.firebasestorage.app'
    })

db = firestore.client()
bucket = storage.bucket()

print("✅ Firebase connected")

# =====================================
# 🔥 LOAD FAMILY MEMBERS
# =====================================

known_face_encodings = []
known_face_names = []

docs = db.collection("family_members").stream()

for doc in docs:
    data = doc.to_dict()
    name = data.get("name")
    image_url = data.get("image_url")

    if name and image_url:
        try:
            response = requests.get(image_url)
            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            encodings = face_recognition.face_encodings(rgb_image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"✔ Loaded {name}")

        except:
            pass

# =====================================
# 📷 CAMERA SETUP
# =====================================

camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # VERY IMPORTANT (reduces lag)

if not camera.isOpened():
    print("❌ Camera failed to open")
    exit()

print("📷 Camera started")

# =====================================
# 🔥 VARIABLES
# =====================================

last_name = None
last_alert_time = None
alert_cooldown_seconds = 5

display_frame = None
stream_frame = None

lock = threading.Lock()

# =====================================
# 🎥 DETECTION LOOP
# =====================================

def detection_loop():
    global display_frame, stream_frame, last_name

    frame_count = 0

    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # Always update stream frame immediately
        with lock:
            stream_frame = frame.copy()

        frame_count += 1

        # Run detection only every 4 frames (huge performance boost)
        if frame_count % 4 != 0:
            with lock:
                display_frame = frame.copy()
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):

            name = "Stranger"

            if known_face_encodings:
                matches = face_recognition.compare_faces(
                    known_face_encodings,
                    face_encoding,
                    tolerance=0.5
                )

                face_distances = face_recognition.face_distance(
                    known_face_encodings,
                    face_encoding
                )

                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

            if name != last_name:
                if name == "Stranger":
                    print("⚠ Stranger detected!")
                    send_stranger_alert(frame)
                else:
                    print(f"✔ {name} detected")

                last_name = name

            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        with lock:
            display_frame = frame.copy()

# =====================================
# 🌐 FLASK STREAMING
# =====================================

app = Flask(__name__)

def generate_frames():
    global stream_frame

    while True:
        with lock:
            if stream_frame is None:
                continue

            frame = stream_frame.copy()

        ret, buffer = cv2.imencode(
            '.jpg', frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        )

        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() +
               b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# =====================================
# 🚀 START SYSTEM
# =====================================

if __name__ == "__main__":

    t = threading.Thread(target=detection_loop)
    t.daemon = True
    t.start()

    app.run(host="0.0.0.0", port=5000,
            threaded=True,
            debug=False,
            use_reloader=False)

    camera.release()
    cv2.destroyAllWindows()
