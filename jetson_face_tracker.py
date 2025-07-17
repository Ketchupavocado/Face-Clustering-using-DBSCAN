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
KNOWN_DIR = "ClusteredFaces"
UNKNOWN_DIR = "ClusteredFaces"
SAVE_FRAMES = 5
MIN_FRAC = 0.02
MAX_FRAC = 0.25
DETECT_INTERVAL = 10
OVERLAP_THRESHOLD = 0.4

# === Globals ===
known_encs, known_labels = [], []
trackers, labels, save_buffers = [], [], []
output_frame = None
lock = threading.Lock()
frame_count = 0

# === Setup Flask ===
app = Flask(__name__)
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


# === Load Known Faces ===
def load_known_faces():
    print("[INFO] Loading known faces...")
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


# === Face Overlap Helper ===
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


# === Save to Unknown_X folder ===
def save_unknown_face(face_img, folder_id):
    save_dir = os.path.join(UNKNOWN_DIR, f"Unknown_{folder_id}")
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{str(uuid.uuid4())}.jpg"
    cv2.imwrite(os.path.join(save_dir, fname), face_img)


# === Main Processing Loop ===
def process_frames():
    global output_frame, frame_count, trackers, labels, save_buffers

    unknown_id_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        frame_count += 1

        new_trackers, new_labels, new_buffers = [], [], []

        if frame_count % DETECT_INTERVAL == 0:
            face_locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)

            for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
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

                # Check for overlapping boxes
                new_box = (left, top, right, bottom)
                if any(overlaps(new_box, (int(t.get_position().left()),
                                          int(t.get_position().top()),
                                          int(t.get_position().right()),
                                          int(t.get_position().bottom()))) for t in trackers):
                    continue

                # dlib tracker
                tracker = dlib.correlation_tracker()
                tracker.start_track(rgb, dlib.rectangle(left, top, right, bottom))

                new_trackers.append(tracker)
                new_labels.append(name)
                new_buffers.append([] if name == "Unknown" else None)

        else:
            for i, tracker in enumerate(trackers):
                tracker.update(rgb)
                pos = tracker.get_position()
                x1, y1, x2, y2 = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())

                if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    continue

                face_img = frame[y1:y2, x1:x2]
                label = labels[i]

                if label == "Unknown":
                    save_buffers[i].append(face_img)
                    if len(save_buffers[i]) >= SAVE_FRAMES:
                        unknown_id_counter += 1
                        for img in save_buffers[i]:
                            save_unknown_face(img, unknown_id_counter)
                        continue  # drop this tracker after saving
                new_trackers.append(tracker)
                new_labels.append(label)
                new_buffers.append(save_buffers[i] if label == "Unknown" else None)

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        trackers = new_trackers
        labels = new_labels
        save_buffers = new_buffers

        with lock:
            output_frame = frame.copy()


# === Flask Routes ===
@app.route("/")
def index():
    return render_template_string("""
    <html>
    <head><title>Jetson Face Tracker</title></head>
    <body>
        <h2>Jetson Tracker Feed</h2>
        <img src="/video_feed" width="640">
        <p>Press Delete key in terminal to clear trackers (not supported in browser)</p>
    </body>
    </html>
    """)


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

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# === Keyboard Tracker Reset ===
def keyboard_listener():
    import keyboard
    global trackers, labels, save_buffers
    while True:
        if keyboard.is_pressed("delete"):
            print("[INFO] Clearing all trackers...")
            trackers.clear()
            labels.clear()
            save_buffers.clear()
        time.sleep(0.5)


# === Run ===
if __name__ == "__main__":
    load_known_faces()
    threading.Thread(target=process_frames, daemon=True).start()
    try:
        import keyboard
        threading.Thread(target=keyboard_listener, daemon=True).start()
    except ImportError:
        print("[WARN] 'keyboard' module not available, delete key disabled.")
    app.run(host="0.0.0.0", port=5000, threaded=True)
