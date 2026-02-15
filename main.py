import cv2
import numpy as np
import face_recognition
import firebase_admin
import requests
import os
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
# 🔥 LOAD REGISTERED FAMILY MEMBERS
# =====================================

known_face_encodings = []
known_face_names = []

print("Loading family members...")

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

            if image is None:
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)

            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"✔ Loaded {name}")

        except Exception as e:
            print(f"❌ Error loading {name}: {e}")

print("Total known faces:", len(known_face_encodings))
print("-----------------------------------")

# =====================================
# 📷 WEBCAM SETUP
# =====================================

video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not video_capture.isOpened():
    print("❌ Cannot open webcam")
    exit()

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("📷 Webcam started")

# =====================================
# ⚡ PERFORMANCE SETTINGS
# =====================================

frame_count = 0
encode_every_n_frames = 5
last_alert_time = None
alert_cooldown_seconds = 5

# Stable recognition memory
last_name = None
last_face_location = None

# =====================================
# 🚨 SEND ALERT FUNCTION
# =====================================

def send_stranger_alert(frame):
    global last_alert_time

    now = datetime.now()

    if last_alert_time:
        if (now - last_alert_time).seconds < alert_cooldown_seconds:
            return

    filename = f"stranger_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, frame)

    try:
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

        print("🚨 Stranger alert sent to Firebase")

    except Exception as e:
        print("❌ Firebase error:", e)

    finally:
        if os.path.exists(filename):
            os.remove(filename)

    last_alert_time = now

# =====================================
# 🔥 MAIN LOOP
# =====================================

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("❌ Frame read failed")
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    # Run encoding only every few frames
    if frame_count % encode_every_n_frames == 0 and face_locations:

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

            # Only print when identity changes
            if name != last_name:

                if name == "Stranger":
                    print("⚠ Stranger detected!")
                    send_stranger_alert(frame)
                else:
                    print(f"✔ {name} detected")

                last_name = name

            last_face_location = face_location

    # Draw stable box
    if last_face_location:
        top, right, bottom, left = last_face_location
        scale = 2

        cv2.rectangle(
            frame,
            (left * scale, top * scale),
            (right * scale, bottom * scale),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            last_name if last_name else "Scanning...",
            (left * scale, top * scale - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("CCTV Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
