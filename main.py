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

cred = credentials.Certificate("firebase_key.json")

firebase_admin.initialize_app(cred, {
    'storageBucket': 'fyp2-92b3f.firebasestorage.app'
})

db = firestore.client()
bucket = storage.bucket()

print("✅ Firebase connected")

# =====================================
# 🔥 LOAD KNOWN FACES FROM events
# =====================================

known_face_encodings = []
known_face_names = []

print("🔄 Loading known faces...")

docs = db.collection("events").stream()

for doc in docs:
    data = doc.to_dict()
    name = data.get("name")
    image_url = data.get("imageUrl")

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
                print(f"✅ Loaded {name}")

        except Exception as e:
            print(f"Error loading {name}: {e}")

print("✅ All known faces loaded")
print("-----------------------------------")

# =====================================
# 📷 UGREEN WEBCAM SETUP
# =====================================

video_capture = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L)

if not video_capture.isOpened():
    print("❌ Cannot open UGREEN webcam")
    exit()

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("📷 UGREEN Webcam started (/dev/video2)")

# =====================================
# 🔥 PERFORMANCE SETTINGS
# =====================================

frame_count = 0
encode_every_n_frames = 3
last_alert_time = None

# =====================================
# 🔥 MAIN LOOP
# =====================================

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Fix mirror effect
    frame = cv2.flip(frame, 1)

    frame_count += 1

    # Resize for detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    # Always draw boxes
    for face_location in face_locations:
        scale_back = 1 / 0.3
        top, right, bottom, left = face_location
        top = int(top * scale_back)
        right = int(right * scale_back)
        bottom = int(bottom * scale_back)
        left = int(left * scale_back)

        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)

    # Encode only every few frames (performance control)
    if frame_count % encode_every_n_frames == 0 and len(face_locations) > 0:

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):

            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )

            name = "Stranger"

            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            scale_back = 1 / 0.3
            top, right, bottom, left = face_location
            top = int(top * scale_back)
            right = int(right * scale_back)
            bottom = int(bottom * scale_back)
            left = int(left * scale_back)

            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

            # 🚨 Stranger detection
            if name == "Stranger":
                now = datetime.now()

                if last_alert_time is None or (now - last_alert_time).seconds > 5:

                    timestamp = now.strftime("%Y%m%d_%H%M%S")
                    filename = f"stranger_{timestamp}.jpg"

                    cv2.imwrite(filename, frame)

                    print("⚠ Stranger detected!")

                    blob = bucket.blob(f"strangers/{filename}")
                    blob.upload_from_filename(filename)
                    blob.make_public()

                    image_url = blob.public_url

                    db.collection("alerts").add({
                        "type": "Stranger Detected",
                        "time": timestamp,
                        "image_url": image_url
                    })

                    print("✅ Stranger uploaded and alert saved")

                    os.remove(filename)

                    last_alert_time = now

    cv2.imshow("CCTV Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
