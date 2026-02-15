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

display_frame = None   # Frame with detection overlay
stream_frame = None    # Clean frame for mobile

lock = threading.Lock()

# =====================================
# 🚨 ALERT FUNCTION
# =====================================

def send_stranger_alert(frame):
    global last_alert_time

    now = datetime.now()

    if last_alert_time:
        if (now - last_alert_time).seconds < alert_cooldown_seconds:
            return

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

    print("🚨 Stranger alert sent")
    os.remove(filename)
    last_alert_time = now

# =====================================
# 🎥 DETECTION LOOP
# =====================================

def detection_loop():
    global display_frame, stream_frame, last_name

    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # Save clean frame for mobile
        clean_copy = frame.copy()

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

            # Draw ONLY on display frame
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        with lock:
            display_frame = frame.copy()
            stream_frame = clean_copy.copy()

# =====================================
# 🌐 FLASK STREAMING (CLEAN FRAME)
# =====================================

app = Flask(__name__)

def generate_frames():
    global stream_frame

    while True:
        with lock:
            if stream_frame is None:
                continue

            ret, buffer = cv2.imencode('.jpg', stream_frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# =====================================
# 🚀 START SYSTEM
# =====================================

if __name__ == "__main__":

    # Start AI thread
    t = threading.Thread(target=detection_loop)
    t.daemon = True
    t.start()

    # Start Flask in another thread
    flask_thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
    )
    flask_thread.daemon = True
    flask_thread.start()

    # Show detection window on Raspberry Pi display
    while True:
        with lock:
            if display_frame is not None:
                cv2.imshow("CCTV Detection (Raspberry Pi)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
