import cv2
import face_recognition

def run_face_size_test():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_height * frame_width

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)

        for (top, right, bottom, left) in face_locations:
            w = right - left
            h = bottom - top
            box_area = w * h
            area_ratio = box_area / frame_area * 100  # percentage of frame area

            # Draw green rectangle around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Display percentage text on top-left corner of box
            text = f"{area_ratio:.2f}% of screen"
            cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Size Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_size_test()
