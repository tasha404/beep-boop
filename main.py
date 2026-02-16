import cv2
import numpy as np
import face_recognition
import firebase_admin
import requests
import os
import time
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
        except Exception as e:
            print("Error loading member:", e)

# =====================================
# 📷 CAMERA SETUP (USB Webcam - Far Optimized)
# =====================================

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    print("❌ Camera failed to open")
    exit()

print("📷 Camera started")

# =====================================
# 🔥 VARIABLES
# =====================================

RESIZE_SCALE = 0.8      # Higher = better far detection
TOLERANCE = 0.6         # Slightly relaxed
DETECT_EVERY = 6        # Balance smoothness

last_face_locations = []
last_face_names = []
last_face_time = 0
face_display_duration = 1.2

last_alert_time = None
alert_cooldown_seconds = 10

display_frame = None
stream_frame = None
frame_count = 0

# =====================================
# 🚨 ALERT FUNCTION
# =====================================

def send_stranger_alert(frame):
    global last_alert_time
    now = datetime.now()

    if last_alert_time:
        if (now - last_alert_time).seconds < alert_cooldown_seconds:
            return

    try:
        filename = f"stranger_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)

        blob = bucket.blob(f"strangers/{filename}")
        blob.upload_from_filename(filename)
        blob.make_public()

        db.collection("alerts").add({
            "type": "stranger",
            "title": "Stranger Detected",
            "timestamp": firestore.SERVER_TIMESTAMP,
            "status": "unread",
            "image_url": blob.public_url
        })

        os.remove(filename)
        last_alert_time = now
        print("🚨 Stranger alert sent")

    except Exception as e:
        print("Alert error:", e)

# =====================================
# 🎥 DETECTION LOOP
# =====================================

def detection_loop():
    global display_frame, stream_frame
    global last_face_locations, last_face_names
    global last_face_time, frame_count

    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        stream_frame = frame.copy()
        frame_count += 1

        if frame_count % DETECT_EVERY == 0:

            small = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(
                rgb_small,
                model="hog"
            )

            face_encodings = face_recognition.face_encodings(
                rgb_small,
                face_locations
            )

            face_names = []

            for face_encoding in face_encodings:
                name = "Stranger"

                if known_face_encodings:
                    matches = face_recognition.compare_faces(
                        known_face_encodings,
                        face_encoding,
                        tolerance=TOLERANCE
                    )

                    face_distances = face_recognition.face_distance(
                        known_face_encodings,
                        face_encoding
                    )

                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]

                face_names.append(name)

                if name == "Stranger":
                    send_stranger_alert(frame)

            last_face_locations = face_locations
            last_face_names = face_names
            last_face_time = time.time()

        # Draw boxes
        if time.time() - last_face_time < face_display_duration:

            scale = int(1 / RESIZE_SCALE)

            for (top, right, bottom, left), name in zip(
                last_face_locations, last_face_names
            ):
                cv2.rectangle(
                    frame,
                    (left * scale, top * scale),
                    (right * scale, bottom * scale),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    name,
                    (left * scale, top * scale - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        display_frame = frame

# =====================================
# 🌐 FLASK STREAM
# =====================================

app = Flask(__name__)

def generate_frames():
    global stream_frame

    while True:
        if stream_frame is None:
            continue

        ret, buffer = cv2.imencode(
            '.jpg',
            stream_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        )

        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() +
               b'\r\n')

@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# =====================================
# 🚀 START SYSTEM
# =====================================

if __name__ == "__main__":

    t = threading.Thread(target=detection_loop)
    t.daemon = True
    t.start()

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
