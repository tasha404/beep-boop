from flask import Flask, Response
import cv2
import time

app = Flask(__name__)

# Open camera safely
camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("❌ Camera failed to open")
else:
    print("✅ Camera opened successfully")

# Set resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def generate_frames():
    while True:
        success, frame = camera.read()

        if not success:
            print("⚠ Frame read failed, retrying...")
            time.sleep(0.1)
            continue   # DO NOT break

        # Resize frame
        frame = cv2.resize(frame, (320, 240))

        # Compress JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)

        if not ret:
            continue

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(0.1)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/frame')
def frame():
    success, frame = camera.read()

    if not success:
        return "Camera error", 500

    frame = cv2.resize(frame, (320, 240))
    ret, buffer = cv2.imencode('.jpg', frame)

    return Response(buffer.tobytes(),
                    mimetype='image/jpeg')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
