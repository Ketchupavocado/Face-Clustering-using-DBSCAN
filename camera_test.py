import cv2
import face_recognition
import os
import time
import uuid
import shutil

# === CONFIG ===
TARGET_FPS = 30
FRAME_DELAY = 1 / TARGET_FPS
KNOWN_FACES_DIR = "ClusteredFaces"
UNKNOWN_BASE = "Unknown"
MAX_TRACKERS = 5

def load_known_faces():
    known_encodings = []
    known_labels = []

    for label in os.listdir(KNOWN_FACES_DIR):
        label_path = os.path.join(KNOWN_FACES_DIR, label)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            filepath = os.path.join(label_path, filename)
            try:
                img = face_recognition.load_image_file(filepath)
                enc = face_recognition.face_encodings(img)
                if enc:
                    known_encodings.append(enc[0])
                    known_labels.append(label)
            except Exception as e:
                print(f"[WARN] Skipping {filepath}: {e}")

    print(f"[INFO] Loaded {len(known_encodings)} known faces.")
    return known_encodings, known_labels

def create_tracker():
    try:
        return cv2.legacy.TrackerKCF_create()
    except AttributeError:
        return cv2.TrackerKCF_create()

def is_overlapping(boxA, boxB, threshold=0.4):
    # boxA and boxB = (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return False

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou > threshold

def save_unknown_face(frame, location, unknown_id):
    top, right, bottom, left = location
    face_img = frame[top:bottom, left:right]
    dir_path = os.path.join(KNOWN_FACES_DIR, f"{UNKNOWN_BASE}_{unknown_id}")
    os.makedirs(dir_path, exist_ok=True)
    filename = os.path.join(dir_path, f"{uuid.uuid4().hex}.jpg")
    cv2.imwrite(filename, face_img)
    print(f"[INFO] Saved unknown face to {filename}")

def run_face_recognition():
    known_encodings, known_labels = load_known_faces()
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)
    print("[INFO] Starting video stream...")

    trackers = []
    names = []
    unknown_counter = 1

    while True:
        ret, frame = vs.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame.shape[:2]

        # Update trackers
        new_trackers = []
        new_names = []
        tracker_bboxes = []

        for i, tracker in enumerate(trackers):
            success, bbox = tracker.update(frame)
            if not success:
                print(f"[WARN] Tracker {i} lost tracking.")
                continue

            x, y, w, h = [int(v) for v in bbox]
            if w <= 0 or h <= 0:
                print(f"[WARN] Invalid box size: {bbox}")
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, names[i], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            new_trackers.append(tracker)
            new_names.append(names[i])
            tracker_bboxes.append((x, y, w, h))

        trackers = new_trackers
        names = new_names

        # Detect new faces if under limit
        if len(trackers) < MAX_TRACKERS:
            face_locations = face_recognition.face_locations(rgb)
            face_encodings = face_recognition.face_encodings(rgb, face_locations)

            for location, encoding in zip(face_locations, face_encodings):
                top, right, bottom, left = location
                w, h = right - left, bottom - top

                # Validate bounding box
                if left < 0 or top < 0 or right > frame_w or bottom > frame_h:
                    continue
                if w <= 0 or h <= 0:
                    continue

                bbox = (left, top, w, h)

                # Skip if already tracked
                if any(is_overlapping(bbox, tb) for tb in tracker_bboxes):
                    continue

                matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
                name = "Unknown"

                if True in matches:
                    matched_idx = matches.index(True)
                    name = known_labels[matched_idx]
                else:
                    name = f"{UNKNOWN_BASE}_{unknown_counter}"
                    save_unknown_face(frame, location, unknown_counter)
                    unknown_counter += 1

                tracker = create_tracker()
                success = tracker.init(frame, bbox)
                if success:
                    trackers.append(tracker)
                    names.append(name)
                    tracker_bboxes.append(bbox)
                    print(f"[INFO] Creating tracker for face at {bbox}")
                else:
                    print("[WARN] Failed to initialize tracker")

        # Display frame
        cv2.imshow("Live Face Recognition", frame)

        # Throttle FPS
        time.sleep(FRAME_DELAY)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_recognition()
