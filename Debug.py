# test_face_detection.py

import cv2
import face_recognition

cap = cv2.VideoCapture(0)  # Use camera index if necessary

print("[INFO] Starting camera... Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB for face_recognition
    face_locations = face_recognition.face_locations(rgb_frame)

    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Face Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
