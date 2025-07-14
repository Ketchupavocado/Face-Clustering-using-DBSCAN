import cv2
import face_recognition
import numpy as np
import os
import uuid
from datetime import datetime

# === CONFIGURATION ===
CLUSTERED_DIR = 'ClusteredFaces/'          # Your clustered output (e.g., from DBSCAN)
UNKNOWN_BASE_DIR = 'UnknownFaces/'         # Base folder to store unknowns
SAVE_UNKNOWN_EVERY_N_FRAMES = 30           # Save every N frames (if still visible)


# === Load Known Encodings and Labels ===
def load_known_faces(clustered_dir):
    known_encodings = []
    known_labels = []

    for label in os.listdir(clustered_dir):
        person_dir = os.path.join(clustered_dir, label)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = face_recognition.load_image_file(img_path)
            face_locations = face_recognition.face_locations(img)
            if face_locations:
                encoding = face_recognition.face_encodings(img, face_locations)[0]
                known_encodings.append(encoding)
                known_labels.append(label)

    print(f"[INFO] Loaded {len(known_encodings)} known faces from '{clustered_dir}'")
    return known_encodings, known_labels


# === Tracker Class ===
class TrackedFace:
    def __init__(self, tracker, label, face_id, unknown_folder=None, last_saved_frame=0):
        self.tracker = tracker
        self.label = label
        self.face_id = face_id  # UUID
        self.unknown_folder = unknown_folder  # "Unknown_1", etc.
        self.last_saved_frame = last_saved_frame
        self.bbox = None


# === Create Tracker (with fallback for OpenCV versions) ===
def create_tracker():
    if hasattr(cv2, 'TrackerKCF_create'):
        return cv2.TrackerKCF_create()
    elif hasattr(cv2, 'legacy'):
        return cv2.legacy.TrackerKCF_create()
    else:
        raise RuntimeError("OpenCV is missing KCF tracker. Install opencv-contrib-python.")


# === Save Unknown Face ===
def save_unknown_face(frame, bbox, folder_name):
    folder_path = os.path.join(UNKNOWN_BASE_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    x, y, w, h = [int(v) for v in bbox]
    face_crop = frame[y:y + h, x:x + w]
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(folder_path, filename)
    cv2.imwrite(path, face_crop)
    print(f"[INFO] Saved unknown face to {path}")


# === Generate Unique Unknown Folder Name ===
def generate_unknown_folder(existing_folders):
    i = 1
    while True:
        name = f"Unknown_{i}"
        if name not in existing_folders:
            return name
        i += 1


# === Check if Face Already Being Tracked ===
def is_face_already_tracked(bbox, tracked_faces, iou_threshold=0.4):
    x1, y1, w1, h1 = bbox
    for tf in tracked_faces:
        if tf.bbox is None:
            continue
        x2, y2, w2, h2 = tf.bbox

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / float(union_area) if union_area != 0 else 0

        if iou > iou_threshold:
            return True
    return False


# === Main Recognition Loop ===
def recognize_from_camera(known_encodings, known_labels):
    cap = cv2.VideoCapture(0)
    tracked_faces = []
    frame_count = 0

    # Track used unknown folders
    existing_unknown_folders = set(os.listdir(UNKNOWN_BASE_DIR)) if os.path.exists(UNKNOWN_BASE_DIR) else set()

    print("[INFO] Starting video stream. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Update tracked faces
        updated_tracked = []
        for tf in tracked_faces:
            success, bbox = tf.tracker.update(frame)
            if success:
                tf.bbox = bbox
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{tf.label}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Save unknown face to its unique folder
                if tf.label == "Unknown" and (frame_count - tf.last_saved_frame) >= SAVE_UNKNOWN_EVERY_N_FRAMES:
                    save_unknown_face(frame, bbox, tf.unknown_folder)
                    tf.last_saved_frame = frame_count

                updated_tracked.append(tf)
        tracked_faces = updated_tracked

        # Detect new faces every 15 frames or if no trackers exist
        if frame_count % 15 == 0 or len(tracked_faces) == 0:
            rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                x, y, w, h = left, top, right - left, bottom - top
                bbox = (x, y, w, h)

                # Skip face if already being tracked
                if is_face_already_tracked(bbox, tracked_faces):
                    continue

                matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)
                name = "Unknown"
                unknown_folder = None

                if True in matches:
                    best_match = np.argmin(face_recognition.face_distance(known_encodings, encoding))
                    name = known_labels[best_match]

                if name == "Unknown":
                    unknown_folder = generate_unknown_folder(existing_unknown_folders)
                    existing_unknown_folders.add(unknown_folder)

                tracker = create_tracker()
                tracker.init(frame, bbox)
                tracked_faces.append(
                    TrackedFace(tracker, name, str(uuid.uuid4()), unknown_folder, frame_count)
                )

        cv2.imshow("Live Face Recognition with Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# === Entry Point ===
if __name__ == "__main__":
    encodings, labels = load_known_faces(CLUSTERED_DIR)
    recognize_from_camera(encodings, labels)
