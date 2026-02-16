import cv2
import numpy as np
import face_recognition
import firebase_admin
import requests
import os
import time
from datetime import datetime
from firebase_admin import credentials, firestore, storage

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

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not camera.isOpened():
    print("❌ Camera failed")
    exit()

print("📷 Camera started")

# =====================================
# ⚙ PARAMETERS
# =====================================

PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360
TOLERANCE = 0.6
DETECT_EVERY = 6
MIN_FACE_SIZE = 40

frame_count = 0
last_face_locations = []
last_face_names = []

last_alert_time = None
alert_cooldown_seconds = 10

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

    except:
        pass

# =====================================
# 🚀 MAIN LOOP (NO THREADING)
# =====================================

cv2.namedWindow("CCTV Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("CCTV Camera", 960, 540)

while True:
    ret, frame = camera.read()
    if not ret:
        continue

    frame_count += 1

    # Detect only every few frames
    if frame_count % DETECT_EVERY == 0:

        small = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(
            rgb_small, face_locations
        )

        last_face_locations = []
        last_face_names = []

        scale_x = frame.shape[1] / PROCESS_WIDTH
        scale_y = frame.shape[0] / PROCESS_HEIGHT

        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):

            if (right - left) < MIN_FACE_SIZE:
                continue

            name = "Stranger"

            if known_face_encodings:
                distances = face_recognition.face_distance(
                    known_face_encodings,
                    face_encoding
                )

                best_index = np.argmin(distances)

                if distances[best_index] < TOLERANCE:
                    name = known_face_names[best_index]

            # Scale back to 720p
            top = int(top * scale_y)
            bottom = int(bottom * scale_y)
            left = int(left * scale_x)
            right = int(right * scale_x)

            last_face_locations.append((top, right, bottom, left))
            last_face_names.append(name)

            if name == "Stranger":
                send_stranger_alert(frame)

    # Draw last known detections
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
