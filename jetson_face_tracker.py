# It looks like your code is already saving faces and logging them, but you reported that the directories and files aren't appearing in the filesystem despite the logs. That typically indicates either the wrong working directory, or that the script lacks write permissions to the target path.

# Here are some fixes and additional logging to help debug and ensure the directories/files are written:

import os
import cv2
import dlib
import uuid
import time
import threading
import numpy as np
import face_recognition
from flask import Flask, Response, render_template_string

# === Configs ===
KNOWN_DIR = os.path.abspath("ClusteredFaces")
UNKNOWN_DIR = KNOWN_DIR
SAVE_FRAMES = 5
MIN_FRAC = 0.02
MAX_FRAC = 0.25
DETECT_INTERVAL = 10
OVERLAP_THRESHOLD = 0.4

# === Globals ===
known_encs, known_labels = [], []
trackers, labels, save_buffers, folder_ids = [], [], [], []
output_frame = None
lock = threading.Lock()
frame_count = 0

app = Flask(__name__)
cap = cv2.VideoCapture(0)

# === Load Known Faces ===
def load_known_faces():
    print(f"[INFO] Loading known faces from: {KNOWN_DIR}")
    if not os.path.exists(KNOWN_DIR):
        os.makedirs(KNOWN_DIR)
    for person in os.listdir(KNOWN_DIR):
        p = os.path.join(KNOWN_DIR, person)
        if os.path.isdir(p) and not person.startswith("Unknown"):
            for img_file in os.listdir(p):
                if img_file.lower().endswith(('.jpg', '.png')):
                    img_path = os.path.join(p, img_file)
                    img = face_recognition.load_image_file(img_path)
                    encs = face_recognition.face_encodings(img)
                    if encs:
                        known_encs.append(encs[0])
                        known_labels.append(person)
    print(f"[INFO] Loaded {len(known_encs)} known faces")

# === Helpers ===
def overlaps(rect1, rect2):
    x1, y1, x2, y2 = rect1
    xx1, yy1, xx2, yy2 = rect2
    xa = max(x1, xx1)
    ya = max(y1, yy1)
    xb = min(x2, xx2)
    yb = min(y2, yy2)
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (xx2 - xx1) * (yy2 - yy1)
    iou = inter_area / float(box1_area + box2_area - inter_area + 1e-5)
    return iou > OVERLAP_THRESHOLD

def get_next_unknown_id():
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    existing = [f for f in os.listdir(UNKNOWN_DIR) if f.startswith("Unknown_")]
    numbers = [int(f.split("_")[1]) for f in existing if f.split("_")[1].isdigit()]
    next_id = max(numbers, default=0) + 1
    return next_id

def save_unknown_face(face_img, folder_id):
    save_dir = os.path.join(UNKNOWN_DIR, f"Unknown_{folder_id}")
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{uuid.uuid4()}.jpg"
    full_path = os.path.join(save_dir, fname)
    success = cv2.imwrite(full_path, face_img)
    if success:
        print(f"[INFO] Saved unknown face: {full_path}")
    else:
        print(f"[ERROR] Failed to save unknown face: {full_path}")

# === Processing ===
def process_frames():
    global output_frame, frame_count, trackers, labels, save_buffers, folder_ids

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        frame_count += 1

        new_trackers, new_labels, new_buffers, new_folder_ids = [], [], [], []

        if frame_count % DETECT_INTERVAL == 0:
            face_locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)

            for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
                box_area = (right - left) * (bottom - top)
                screen_area = h * w
                frac = box_area / screen_area
                if not (MIN_FRAC <= frac <= MAX_FRAC):
                    continue

                name, folder_id = "Unknown", None
                matches = face_recognition.compare_faces(known_encs, enc, tolerance=0.45)
                if True in matches:
                    best_idx = np.argmin(face_recognition.face_distance(known_encs, enc))
                    name = known_labels[best_idx]
                else:
                    folder_id = get_next_unknown_id()

                new_box = (left, top, right, bottom)
                if any(overlaps(new_box, (int(t.get_position().left()), int(t.get_position().top()), int(t.get_position().right()), int(t.get_position().bottom()))) for t in trackers):
                    continue

                tracker = dlib.correlation_tracker()
                tracker.start_track(rgb, dlib.rectangle(left, top, right, bottom))

                new_trackers.append(tracker)
                new_labels.append(name)
                new_buffers.append([] if name == "Unknown" else None)
                new_folder_ids.append(folder_id)

        for i, tracker in enumerate(trackers):
            tracker.update(rgb)
            pos = tracker.get_position()
            x1, y1, x2, y2 = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())

            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                continue

            face_img = frame[y1:y2, x1:x2]
            label, folder_id = labels[i], folder_ids[i]

            if label == "Unknown":
                save_buffers[i].append(face_img)
                if len(save_buffers[i]) >= SAVE_FRAMES:
                    for img in save_buffers[i]:
                        save_unknown_face(img, folder_id)
                    continue

            new_trackers.append(tracker)
            new_labels.append(label)
            new_buffers.append(save_buffers[i] if label == "Unknown" else None)
            new_folder_ids.append(folder_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        trackers, labels, save_buffers, folder_ids = new_trackers, new_labels, new_buffers, new_folder_ids

        with lock:
            output_frame = frame.copy()

# === Flask ===
@app.route("/")
def index():
    return render_template_string("""
    <html><head><title>Jetson Face Tracker</title></head>
    <body><h2>Jetson Tracker Feed</h2><img src="/video_feed" width="640"></body></html>
    """)

@app.route("/video_feed")
def video_feed():
    def generate():
        global output_frame
        while True:
            with lock:
                if output_frame is None:
                    continue
                ret, buf = cv2.imencode(".jpg", output_frame)
                if not ret:
                    continue
                frame = buf.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    load_known_faces()
    threading.Thread(target=process_frames, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)
