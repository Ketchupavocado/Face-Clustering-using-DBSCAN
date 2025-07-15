import cv2, face_recognition, numpy as np, os, uuid, time, shutil, dlib
from datetime import datetime

CLUSTERED_DIR = 'ClusteredFaces/'
UNKNOWN_BASE_DIR = 'UnknownFaces/'
SAVE_UNKNOWN_EVERY_N_FRAMES = 30
RELOAD_CHECK_INTERVAL = 10  # seconds

def load_known_faces():
    encs, labs = [], []
    valid_exts = ('.jpg', '.jpeg', '.png')

    for label in os.listdir(CLUSTERED_DIR):
        p = os.path.join(CLUSTERED_DIR, label)
        if os.path.isdir(p):
            for img_file in os.listdir(p):
                if not img_file.lower().endswith(valid_exts):
                    continue  # skip non-image files
                imgp = os.path.join(p, img_file)
                try:
                    img = face_recognition.load_image_file(imgp)
                    locs = face_recognition.face_locations(img)
                    if locs:
                        enc = face_recognition.face_encodings(img, locs)[0]
                        encs.append(enc)
                        labs.append(label)
                except Exception as e:
                    print(f"[WARN] Skipping {imgp}: {e}")
    print(f"[INFO] {len(encs)} known faces loaded.")
    return encs, labs


def move_labeled_unknowns(known_labels):
    moved = False
    for folder in os.listdir(UNKNOWN_BASE_DIR):
        src = os.path.join(UNKNOWN_BASE_DIR, folder)
        if os.path.isdir(src) and folder not in known_labels:
            dst = os.path.join(CLUSTERED_DIR, folder)
            shutil.move(src, dst)
            print(f"[INFO] {folder} moved to {dst} and added to known.")
            moved = True
    return moved

def create_tracker():
    return dlib.correlation_tracker()

class TrackedFace:
    def __init__(self, tracker, label, face_id, folder, last_saved=0):
        self.tracker = tracker
        self.label = label
        self.face_id = face_id
        self.folder = folder
        self.last_saved = last_saved
        self.bbox = None

def generate_unknown_folder(existing):
    i=1
    while True:
        n=f"Unknown_{i}"
        if n not in existing:
            existing.add(n)
            return n
        i+=1

def prompt_for_name(current_label):
    print(f"Rename face (current label: '{current_label}'). Enter new name or press Enter to skip:")
    new_name = input()
    if new_name.strip() == '':
        return current_label
    return new_name.strip()

def recognize_camera():
    if not os.path.exists(UNKNOWN_BASE_DIR):
        os.makedirs(UNKNOWN_BASE_DIR)

    known_encs, known_labs = load_known_faces()
    existing_unknowns = set(os.listdir(UNKNOWN_BASE_DIR))
    tracked=[]
    frame_count=0
    last_reload=time.time()

    selected_face = None  # Track which face was clicked

    cap = cv2.VideoCapture(0)
    print("[INFO] Press 'q' to quit.")

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_face
        if event == cv2.EVENT_LBUTTONDOWN:
            for tf in tracked:
                if tf.bbox:
                    bx, by, bw, bh = tf.bbox
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        selected_face = tf
                        print(f"[INFO] Face clicked: {tf.label}")
                        break

    cv2.namedWindow("Recognition+Tracking")
    cv2.setMouseCallback("Recognition+Tracking", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count+=1

        # Reload known if new folders appeared
        if time.time() - last_reload > RELOAD_CHECK_INTERVAL:
            if move_labeled_unknowns(known_labs):
                known_encs, known_labs = load_known_faces()
            last_reload = time.time()

        # Update trackers
        for tf in tracked[:]:
            pos = tf.tracker.get_position()
            x, y, x2, y2 = map(int, [pos.left(), pos.top(), pos.right(), pos.bottom()])
            tf.bbox = (x, y, x2-x, y2-y)
            cv2.rectangle(frame, (x, y), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, tf.label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Save unknown face snapshots periodically
            if tf.folder and (frame_count - tf.last_saved) >= SAVE_UNKNOWN_EVERY_N_FRAMES:
                cx, cy, cw, ch = tf.bbox
                crop = frame[cy:cy+ch, cx:cx+cw]
                fname = datetime.now().strftime('%Y%m%d_%H%M%S') + ".jpg"
                dest = os.path.join(UNKNOWN_BASE_DIR, tf.folder, fname)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                cv2.imwrite(dest, crop)
                tf.last_saved = frame_count
                print(f"[INFO] Saved unknown snapshot: {dest}")

        # Handle face rename on click
        if selected_face:
            # Pause video display
            cv2.imshow("Recognition+Tracking", frame)
            cv2.waitKey(1)  # Refresh window before input

            new_label = prompt_for_name(selected_face.label)
            if new_label != selected_face.label:
                old_folder = selected_face.folder
                if old_folder is None and selected_face.label != "Unknown":
                    # It was known, no folder to rename, just update label
                    selected_face.label = new_label
                else:
                    # Rename the folder
                    src = os.path.join(UNKNOWN_BASE_DIR, old_folder) if old_folder else None
                    dst = os.path.join(CLUSTERED_DIR, new_label)
                    if src and os.path.exists(src):
                        if os.path.exists(dst):
                            print(f"[WARN] Folder {dst} exists. Merging images.")
                            for f in os.listdir(src):
                                shutil.move(os.path.join(src, f), dst)
                            os.rmdir(src)
                        else:
                            shutil.move(src, dst)
                    else:
                        # If no folder (e.g., just created new face), create folder now
                        if not os.path.exists(dst):
                            os.makedirs(dst)

                    # Update label and folder for tracked face
                    selected_face.label = new_label
                    selected_face.folder = new_label

                # Reload known faces after renaming
                known_encs, known_labs = load_known_faces()
                print(f"[INFO] Face renamed to {new_label} and database updated.")
            else:
                print("[INFO] Rename cancelled.")

            selected_face = None  # reset after rename

        # Every 15 frames or if no trackers -> detect
        if frame_count % 15 == 0 or not tracked:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)

            for (t, r, b, l), enc in zip(locs, encs):
                w, h = r-l, b-t
                overlap = False
                for tf in tracked:
                    if tf.bbox:
                        x, y, ww, hh = tf.bbox
                        xi, yi = max(l, x), max(t, y)
                        x2, y2 = min(r, x + ww), min(b, y + hh)
                        inter = max(0, x2 - xi) * max(0, y2 - yi)
                        if inter / (w*h + ww*hh - inter) > .4:
                            overlap = True
                            break
                if overlap:
                    continue

                matches = face_recognition.compare_faces(known_encs, enc, tolerance=0.45)
                name = "Unknown"
                folder = None
                if True in matches:
                    idx = np.argmin(face_recognition.face_distance(known_encs, enc))
                    name = known_labs[idx]
                else:
                    folder = generate_unknown_folder(existing_unknowns)

                tracker = create_tracker()
                tracker.start_track(frame, dlib.rectangle(l, t, r, b))
                tracked.append(TrackedFace(tracker, name, str(uuid.uuid4()), folder, frame_count))

        cv2.imshow("Recognition+Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_camera()
