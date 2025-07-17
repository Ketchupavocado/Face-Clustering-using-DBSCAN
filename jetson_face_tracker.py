import os
import cv2
import dlib
import time
import threading
import numpy as np
import face_recognition
from flask import Flask, Response

# Configs
KNOWN_DIR = "ClusteredFaces"
MIN_FRAC = 0.02
MAX_FRAC = 0.25
DETECT_INTERVAL = 10

# Load known faces
known_encs, known_labels = [], []
for person in os.listdir(KNOWN_DIR):
    p = os.path.join(KNOWN_DIR, person)
    if os.path.isdir(p):
        for img_file in os.listdir(p):
            if img_file.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(p, img_file)
                img = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_encs.append(encs[0])
                    known_labels.append(person)
print(f"[INFO] Loaded {len(known_encs)} known faces")

# Flask setup
app = Flask(__name__)
output_frame = None
lock = threading.Lock()

# Camera
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

trackers = []
labels = []
frame_count = 0

def process_frames():
    global output_frame, frame_count, trackers, labels
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        frame_count += 1

        if frame_count % DETECT_INTERVAL == 0:
            trackers = []
            labels = []
            boxes = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, boxes)

            for (top, right, bottom, left), enc in zip(boxes, encs):
                box_area = (right - left) * (bottom - top)
                screen_area = h * w
                frac = box_area / screen_area

                if not (MIN_FRAC <= frac <= MAX_FRAC):
                    continue

                name = "Unknown"
                matches = face_recognition.compare_faces(known_encs, enc, tolerance=0.45)
                if True in matches:
                    best_idx = np.argmin(face_recognition.face_distance(known_encs, enc))
                    name = known_labels[best_idx]

                # dlib tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(left, top, right, bottom)
                tracker.start_track(rgb, rect)

                trackers.append(tracker)
                labels.append(name)
        else:
            new_trackers = []
            new_labels = []
            for tracker, label in zip(trackers, labels):
                tracker.update(rgb)
                pos = tracker.get_position()
                x, y, x2, y2 = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                new_trackers.append(tracker)
                new_labels.append(label)
            trackers = new_trackers
            labels = new_labels

        with lock:
            output_frame = frame.copy()

@app.route("/")
def index():
    return """
    <html>
    <head><title>MJPEG Face Stream</title></head>
    <body>
        <h1>Jetson Face Tracker</h1>
        <img src="/video_feed" width="640" />
    </body>
    </html>
    """

def generate():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    threading.Thread(target=process_frames, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
