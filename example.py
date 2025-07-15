import os
import cv2
import face_recognition
import numpy as np
import shutil
import time
from datetime import datetime

CLUSTERED_DIR = 'ClusteredFaces/'
UNKNOWN_DIR = 'UnknownFaces/'
RELOAD_INTERVAL = 10         # seconds
SAVE_UNKNOWN_EVERY_N = 30    # frames
KEEP_BEST_N = 2              # images to keep per person

# Relative min/max face box area ratios (relative to frame area)
MIN_FACE_BOX_AREA_RATIO = 0.01  # 1% of frame area
MAX_FACE_BOX_AREA_RATIO = 0.15  # 15% of frame area

os.makedirs(CLUSTERED_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ==== Load known faces from ClusteredFaces/ ====
def load_known_faces():
    encs, labels = [], []
    for label in os.listdir(CLUSTERED_DIR):
        path = os.path.join(CLUSTERED_DIR, label)
        if os.path.isdir(path):
            for fname in os.listdir(path):
                if fname.lower().endswith(('.jpg', '.png')):
                    img_path = os.path.join(path, fname)
                    try:
                        img = face_recognition.load_image_file(img_path)
                        locs = face_recognition.face_locations(img)
                        if locs:
                            enc = face_recognition.face_encodings(img, locs)[0]
                            encs.append(enc)
                            labels.append(label)
                    except:
                        pass
    print(f"[INFO] Loaded {len(encs)} known faces.")
    return encs, labels

# ==== Tracker wrapper ====
class FaceTrack:
    def __init__(self, tracker, label, folder, last_save_frame):
        self.tracker = tracker
        self.label = label
        self.folder = folder
        self.last_saved = last_save_frame
        self.bbox = None

# ==== Check box overlap ====
def intersect(a, b, iou=0.4):
    if a is None or b is None: return False
    x1, y1, w1, h1 = a; x2, y2, w2, h2 = map(int, b)
    xi, yi = max(x1,x2), max(y1,y2)
    x2i, y2i = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter = max(0, x2i-xi) * max(0, y2i-yi)
    union = w1*h1 + w2*h2 - inter
    return inter/union > iou

# ==== Get new Unknown name ====
def get_new_unknown(existing):
    i = 1
    while f"Unknown_{i}" in existing:
        i += 1
    name = f"Unknown_{i}"
    existing.add(name)
    os.makedirs(os.path.join(UNKNOWN_DIR, name), exist_ok=True)
    return name

# ==== Rename handler ====
def prompt_name(current):
    new = input(f"Enter new name for '{current}' (or press Enter to skip): ").strip()
    return new if new else current

# ==== Move renamed unknowns ====
def update_named_unknowns(known_labels):
    moved = False
    for folder in os.listdir(UNKNOWN_DIR):
        src = os.path.join(UNKNOWN_DIR, folder)
        dst = os.path.join(CLUSTERED_DIR, folder)
        if os.path.isdir(src):
            if os.path.exists(dst):
                # Merge contents instead of crashing
                for file in os.listdir(src):
                    shutil.move(os.path.join(src, file), os.path.join(dst, file))
                shutil.rmtree(src)
                print(f"[INFO] Merged '{folder}' into existing folder.")
            else:
                shutil.move(src, dst)
                print(f"[INFO] Moved '{folder}' to clustered.")
            moved = True
    return moved

# ==== Cleanup helpers ====
def keep_best_clustered_images(clustered_dir=CLUSTERED_DIR, keep_n=KEEP_BEST_N):
    for label in os.listdir(clustered_dir):
        person_dir = os.path.join(clustered_dir, label)
        if not os.path.isdir(person_dir): continue

        imgs = [os.path.join(person_dir, f) for f in os.listdir(person_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(imgs) <= keep_n:
            continue

        imgs.sort(key=lambda x: os.path.getsize(x), reverse=True)
        for img_path in imgs[keep_n:]:
            os.remove(img_path)
            print(f"[CLEANUP] Deleted: {img_path}")

def cleanup_unknown_faces(unknown_dir=UNKNOWN_DIR):
    for folder in os.listdir(unknown_dir):
        path = os.path.join(unknown_dir, folder)
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"[CLEANUP] Deleted unknown folder: {path}")

def hybrid_cleanup():
    print("[INFO] Performing hybrid cleanup...")
    keep_best_clustered_images()
    cleanup_unknown_faces()
    print("[INFO] Cleanup complete.")

# ==== Main camera face tracker ====
def run():
    cap = cv2.VideoCapture(0)
    known_encs, known_labels = load_known_faces()
    tracked = []
    frame_count = 0
    last_reload = time.time()
    existing_unknowns = set(os.listdir(UNKNOWN_DIR))
    clicked = None

    def on_click(event, x, y, flags, param):
        nonlocal clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            for t in tracked:
                if t.bbox:
                    x0, y0, w, h = map(int, t.bbox)
                    if x0 <= x <= x0+w and y0 <= y <= y0+h:
                        clicked = t
                        break

    cv2.namedWindow('Live')
    cv2.setMouseCallback('Live', on_click)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        # Reload knowns if new labels found
        if time.time() - last_reload > RELOAD_INTERVAL:
            if update_named_unknowns(known_labels):
                known_encs, known_labels = load_known_faces()
            last_reload = time.time()

        # Update trackers
        for t in tracked[:]:
            ok, box = t.tracker.update(frame)
            if ok:
                t.bbox = box
            else:
                tracked.remove(t)

        # Handle rename click
        if clicked:
            new_label = prompt_name(clicked.label)
            if new_label != clicked.label:
                if clicked.folder:
                    src = os.path.join(UNKNOWN_DIR, clicked.folder)
                    dst = os.path.join(CLUSTERED_DIR, new_label)
                    shutil.move(src, dst)
                    clicked.folder = None
                clicked.label = new_label
                known_encs, known_labels = load_known_faces()
                print(f"[INFO] Renamed to '{new_label}' and retrained.")
            clicked = None

        # Detect new faces every 15 frames
        if frame_count % 15 == 0 or not tracked:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)

            frame_area = frame.shape[0] * frame.shape[1]

            for (t, r, b, l), e in zip(locs, encs):
                box = (l, t, r-l, b-t)
                area = (r - l) * (b - t)

                if area < MIN_FACE_BOX_AREA_RATIO * frame_area or area > MAX_FACE_BOX_AREA_RATIO * frame_area:
                    continue  # skip faces too small or too large relative to frame size

                if any(intersect(box, t_.bbox) for t_ in tracked):
                    continue

                matches = face_recognition.compare_faces(known_encs, e, tolerance=0.45)
                name = "Unknown"
                folder = None
                if True in matches:
                    name = known_labels[np.argmin(face_recognition.face_distance(known_encs, e))]
                else:
                    folder = get_new_unknown(existing_unknowns)

                tracker = cv2.TrackerCSRT_create()
                if tracker is None:
                    print("[ERROR] Tracker is None. Skipping this face.")
                    continue

                tracker.init(frame, box)
                tracked.append(FaceTrack(tracker, name, folder, frame_count))

        # Draw & save faces
        for t in tracked:
            if t.bbox:
                x, y, w, h = map(int, t.bbox)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, t.label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if t.label.startswith("Unknown") and frame_count - t.last_saved > SAVE_UNKNOWN_EVERY_N:
                    crop = frame[y:y+h, x:x+w]
                    fname = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
                    cv2.imwrite(os.path.join(UNKNOWN_DIR, t.folder, fname), crop)
                    t.last_saved = frame_count

        cv2.imshow('Live', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hybrid_cleanup()  # Cleanup on exit

if __name__ == "__main__":
    run()
