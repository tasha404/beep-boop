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
# 📷 CAMERA SETUP (720p)
# =====================================

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not camera.isOpened():
    print("❌ Camera failed to open")
    exit()

print("📷 Camera started at 720p")

# =====================================
# ⚙ PARAMETERS
# =====================================

PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360

TOLERANCE = 0.6
DETECT_EVERY = 5
MIN_FACE_SIZE = 40

last_alert_time = None
alert_cooldown_seconds = 10

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
    global stream_frame, frame_count

    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        frame_count += 1

        if frame_count % DETECT_EVERY == 0:

            # Resize only for processing
            small_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(
                rgb_small,
                model="hog"
            )

            face_encodings = face_recognition.face_encodings(
                rgb_small,
                face_locations
            )

            scale_x = frame.shape[1] / PROCESS_WIDTH
            scale_y = frame.shape[0] / PROCESS_HEIGHT

            for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings
            ):

                face_width = right - left
                if face_width < MIN_FACE_SIZE:
                    continue

                name = "Stranger"

                if known_face_encodings:
                    distances = face_recognition.face_distance(
                        known_face_encodings,
                        face_encoding
                    )

                    best_index = np.argmin(distances)
                    best_distance = distances[best_index]

                    print("Distance:", round(best_distance, 3))

                    if best_distance < TOLERANCE:
                        name = known_face_names[best_index]

                # Scale back to 720p
                top = int(top * scale_y)
                bottom = int(bottom * scale_y)
                left = int(left * scale_x)
                right = int(right * scale_x)

                # Draw on FULL frame
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if name == "Stranger":
                    send_stranger_alert(frame)

        stream_frame = frame

        time.sleep(0.003)

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
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# =====================================
# 🚀 START SYSTEM (Display + Stream)
# =====================================

if __name__ == "__main__":

    t = threading.Thread(target=detection_loop)
    t.daemon = True
    t.start()

    flask_thread = threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    )
    flask_thread.daemon = True
    flask_thread.start()

    cv2.namedWindow("CCTV Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CCTV Detection", 960, 540)

    while True:
        if stream_frame is not None:
            cv2.imshow("CCTV Detection", stream_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
