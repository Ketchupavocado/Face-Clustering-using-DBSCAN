import os
import cv2
import face_recognition
import numpy as np
import threading
from datetime import datetime

# Directory structure
CLUSTERED_FACES_DIR = "ClusteredFaces"
UNKNOWN_FACES_DIR = "UnknownFaces"

# Face size thresholds (relative to frame area)
MIN_FACE_AREA_RATIO = 0.02  # 2%
MAX_FACE_AREA_RATIO = 0.25  # 25%

# Track encountered unknowns
next_unknown_id = 1
unknown_encodings = []
unknown_labels = {}

# Load known faces
print("[INFO] Loading known faces...")
known_encodings = []
known_names = []

os.makedirs(CLUSTERED_FACES_DIR, exist_ok=True)
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

for name in os.listdir(CLUSTERED_FACES_DIR):
    person_dir = os.path.join(CLUSTERED_FACES_DIR, name)
    if not os.path.isdir(person_dir):
        continue
    for filename in os.listdir(person_dir):
        filepath = os.path.join(person_dir, filename)
        try:
            image = face_recognition.load_image_file(filepath)
            encs = face_recognition.face_encodings(image)
            if len(encs) > 0:
                known_encodings.append(encs[0])
                known_names.append(name)
        except Exception as e:
            print(f"[WARN] Skipping image {filepath}: {e}")

print(f"[INFO] Loaded {len(known_encodings)} known faces.")

def get_unknown_folder(face_encoding):
    global next_unknown_id

    for idx, enc in enumerate(unknown_encodings):
        if face_recognition.compare_faces([enc], face_encoding, tolerance=0.5)[0]:
            return unknown_labels[idx]

    label = f"Unknown_{next_unknown_id}"
    unknown_encodings.append(face_encoding)
    unknown_labels[len(unknown_encodings) - 1] = label
    os.makedirs(os.path.join(UNKNOWN_FACES_DIR, label), exist_ok=True)
    next_unknown_id += 1
    return label

def save_face_image(folder, image):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}.jpg"
    save_path = os.path.join(folder, filename)
    cv2.imwrite(save_path, image)

def run():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("[ERROR] Cannot open camera")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(frame_rgb)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_width * frame_height

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            w, h = right - left, bottom - top
            area_ratio = (w * h) / frame_area

            if not (MIN_FACE_AREA_RATIO <= area_ratio <= MAX_FACE_AREA_RATIO):
                continue

            name = "Unknown"
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(known_encodings, encoding))
                name = known_names[best_match_index]
            else:
                name = get_unknown_folder(encoding)
                face_img = frame[top:bottom, left:right]
                folder_path = os.path.join(UNKNOWN_FACES_DIR, name)
                threading.Thread(target=save_face_image, args=(folder_path, face_img)).start()

            color = (0, 255, 0) if not name.startswith("Unknown") else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
