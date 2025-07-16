import cv2
import os
import time
import numpy as np
import face_recognition

# Configuration
CAMERA_INDEX = 0
KNOWN_FACES_DIR = "clustered_faces"
UNKNOWN_FACES_DIR = "clustered_faces"
FRAME_SAVE_INTERVAL = 5  # seconds between debug frame saves
DEBUG_SAVE_FRAMES = False

# Load known faces
print("[INFO] Loading known faces...")
known_encodings = []
known_names = []

for person in os.listdir(KNOWN_FACES_DIR):
    person_path = os.path.join(KNOWN_FACES_DIR, person)
    if not os.path.isdir(person_path):
        continue
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person)
print(f"[INFO] Loaded {len(known_encodings)} known faces.")

# Start video
print("[INFO] Starting camera... Press Ctrl+C to stop")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("[ERROR] Camera failed to open.")

last_save_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for (box, encoding) in zip(boxes, encodings):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            name = "Unknown"

            if True in matches:
                matched_idx = matches.index(True)
                name = known_names[matched_idx]
            else:
                name = f"Unknown_{int(time.time())}"
                face_image = frame[box[0]:box[2], box[3]:box[1]]
                save_path = os.path.join(UNKNOWN_FACES_DIR, name)
                os.makedirs(save_path, exist_ok=True)
                file_path = os.path.join(save_path, f"{int(time.time())}.jpg")
                cv2.imwrite(file_path, face_image)
                print(f"[NEW] Saved unknown face to: {file_path}")

            print(f"[TRACK] Detected: {name}, Box: {box}")

        # Optional frame saving for debug
        if DEBUG_SAVE_FRAMES and (time.time() - last_save_time > FRAME_SAVE_INTERVAL):
            cv2.imwrite(f"debug_frame_{int(time.time())}.jpg", frame)
            last_save_time = time.time()

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user. Exiting...")

finally:
    cap.release()
    print("[INFO] Camera released.")
