import cv2
import numpy as np
import face_recognition
import firebase_admin
import requests
import os
from datetime import datetime
from picamera2 import Picamera2
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

print("🔄 Loading known faces from events...")

docs = db.collection("events").stream()

for doc in docs:
    data = doc.to_dict()

    name = data.get("name")
    image_url = data.get("imageUrl")

    if name and image_url:
        try:
            print(f"Downloading image for {name}...")

            response = requests.get(image_url)
            image_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if image is None:
                continue

            # Resize large images before encoding
            h, w = image.shape[:2]
            if w > 800:
                scale = 800 / w
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
# 📷 CAMERA SETUP
# =====================================

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"size": (640, 480)}
))
picam2.start()

print("📷 Camera started")

# =====================================
# 🔥 PERFORMANCE SETTINGS
# =====================================

frame_count = 0
process_every_n_frames = 2
last_alert_time = None

# =====================================
# 🔥 MAIN LOOP
# =====================================

while True:
    frame = picam2.capture_array()

    # Fix blue tint (XBGR → BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    frame_count += 1

    # Skip some frames to reduce CPU usage
    if frame_count % process_every_n_frames != 0:
        cv2.imshow("CCTV Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Smaller resize for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

    if len(face_locations) > 0:
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

            # Scale coordinates back
            top, right, bottom, left = face_location
            top = int(top / 0.4)
            right = int(right / 0.4)
            bottom = int(bottom / 0.4)
            left = int(left / 0.4)

            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

            # 🚨 Stranger detected
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

cv2.destroyAllWindows()
picam2.stop()
