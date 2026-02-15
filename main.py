import cv2
import numpy as np
import face_recognition
import firebase_admin
import requests
import os
from datetime import datetime
from firebase_admin import credentials, firestore, storage

# =====================================
# 🔥 FIREBASE SETUP (SAFE INIT)
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

            h, w = image.shape[:2]
            if w > 600:
                scale = 600 / w
                image = cv2.resize(image, (int(w * scale), int(h * scale)))

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)

            if len(encodings) > 0:
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

video_capture = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

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
encode_every_n_frames = 3
last_alert_time = None
alert_cooldown_seconds = 5

# =====================================
# 🚨 FUNCTION: SEND ALERT TO FIREBASE
# =====================================

def send_stranger_alert(frame):
    global last_alert_time

    now = datetime.now()

    if last_alert_time is not None:
        time_diff = (now - last_alert_time).seconds
        if time_diff < alert_cooldown_seconds:
            return

    timestamp_string = now.strftime("%Y%m%d_%H%M%S")
    filename = f"stranger_{timestamp_string}.jpg"

    cv2.imwrite(filename, frame)

    try:
        # Upload to Firebase Storage
        blob = bucket.blob(f"strangers/{filename}")
        blob.upload_from_filename(filename)
        blob.make_public()

        image_url = blob.public_url

        # Save to Firestore (CORRECT STRUCTURE)
        db.collection("alerts").add({
            "type": "stranger",
            "title": "Stranger Detected",
            "timestamp": firestore.SERVER_TIMESTAMP,
            "status": "unread",
            "image_url": image_url
        })

        print("🚨 Stranger alert sent to Firebase")

    except Exception as e:
        print("❌ Error sending alert:", e)

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
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    # Draw face boxes
    for face_location in face_locations:
        scale_back = 2
        top, right, bottom, left = face_location
        cv2.rectangle(
            frame,
            (left * scale_back, top * scale_back),
            (right * scale_back, bottom * scale_back),
            (0, 255, 0),
            2
        )

    # Only encode every few frames
    if frame_count % encode_every_n_frames == 0 and len(face_locations) > 0:

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):

            name = "Stranger"

            if len(known_face_encodings) > 0:
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

            scale_back = 2
            top, right, bottom, left = face_location

            cv2.putText(
                frame,
                name,
                (left * scale_back, top * scale_back - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # 🚨 Stranger detected
            if name == "Stranger":
                print("⚠ Stranger detected!")
                send_stranger_alert(frame)

    cv2.imshow("CCTV Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

