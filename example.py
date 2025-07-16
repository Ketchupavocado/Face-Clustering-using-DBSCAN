import os
import cv2
import face_recognition
import numpy as np
import shutil
import time
from datetime import datetime

CLUSTERED_DIR = 'ClusteredFaces/'
UNKNOWN_DIR = 'UnknownFaces/'
RELOAD_INTERVAL = 10     # seconds
SAVE_UNKNOWN_EVERY_N = 50  # frames
KEEP_BEST_N = 2           # images to keep per person
MIN_BOX_FRAC = 0.02       # 2% of frame area
MAX_BOX_FRAC = 0.25       # 25% of frame area
MAX_TRACKER_AGE = 150     # max frames before tracker removal

os.makedirs(CLUSTERED_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ==== Load known faces from ClusteredFaces/ ====
def load_known_faces():
    print("[INFO] Loading known faces...")
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

# ==== Tracker class ====
class FaceTrack:
    def __init__(self, tracker, label, folder, last_saved, bbox, enc=None):
        self.tracker = tracker
        self.label = label
        self.folder = folder
        self.last_saved = last_saved
        self.bbox = bbox
        self.enc = enc

# ==== IOU box overlap ====
def intersect(a, b, iou=0.4):
    if a is None or b is None:
        return False
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = map(int, b)
    xi, yi = max(x1,x2), max(y1,y2)
    x2i, y2i = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter = max(0, x2i-xi) * max(0, y2i-yi)
    union = w1*h1 + w2*h2 - inter
    return inter/union > iou

# ==== Deduplication ====
def deduplicate_trackers(trackers, iou_thresh=0.5, enc_thresh=0.4):
    to_remove = set()
    for i, t1 in enumerate(trackers):
        for j, t2 in enumerate(trackers):
            if i >= j or i in to_remove or j in to_remove:
                continue
            if t1.bbox and t2.bbox and intersect(t1.bbox, t2.bbox, iou_thresh):
                if t1.enc is not None and t2.enc is not None:
                    dist = face_recognition.face_distance([t1.enc], t2.enc)[0]
                    if dist < enc_thresh:
                        # Keep the one with earlier save time
                        if t1.last_saved >= t2.last_saved:
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
    return [t for idx, t in enumerate(trackers) if idx not in to_remove]

# ==== New unknown folder ====
def get_new_unknown(existing):
    i = 1
    while f"Unknown_{i}" in existing:
        i += 1
    name = f"Unknown_{i}"
    existing.add(name)
    os.makedirs(os.path.join(UNKNOWN_DIR, name), exist_ok=True)
    return name

# ==== Merge renamed unknowns ====
def update_named_unknowns(known_labels):
    moved = False
    for folder in os.listdir(UNKNOWN_DIR):
        src = os.path.join(UNKNOWN_DIR, folder)
        dst = os.path.join(CLUSTERED_DIR, folder)
        if os.path.isdir(src):
            if os.path.exists(dst):
                for file in os.listdir(src):
                    shutil.move(os.path.join(src, file), os.path.join(dst, file))
                shutil.rmtree(src)
                print(f"[INFO] Merged '{folder}' into existing folder.")
            else:
                shutil.move(src, dst)
                print(f"[INFO] Moved '{folder}' to clustered.")
            moved = True
    return moved

# ==== Cleanup ====
def keep_best_clustered_images(clustered_dir=CLUSTERED_DIR, keep_n=2):
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

# ==== Main ====
def run():
    cap = cv2.VideoCapture(0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_w * frame_h

    known_encs, known_labels = load_known_faces()
    tracked = []
    frame_count = 0
    last_reload = time.time()
    existing_unknowns = set(os.listdir(UNKNOWN_DIR))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('d'):  # Delete key
            print("[KEY] Deleted all trackers")
            tracked.clear()

        if time.time() - last_reload > RELOAD_INTERVAL:
            if update_named_unknowns(known_labels):
                known_encs, known_labels = load_known_faces()
            last_reload = time.time()

        # Update all trackers
        for t in tracked[:]:
            ok, box = t.tracker.update(frame)
            if ok:
                t.bbox = box
            else:
                tracked.remove(t)

        # Remove stale trackers
        tracked = [t for t in tracked if frame_count - t.last_saved < MAX_TRACKER_AGE]

        # Deduplicate overlapping trackers
        tracked = deduplicate_trackers(tracked)

        # Detect every 5 frames
        if frame_count % 5 == 0 or not tracked:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)
            for (top, right, bottom, left), e in zip(locs, encs):
                box = (left, top, right - left, bottom - top)
                box_area = box[2] * box[3]
                rel_area = box_area / frame_area
                if rel_area < MIN_BOX_FRAC or rel_area > MAX_BOX_FRAC:
                    continue
                if any(intersect(box, t.bbox) for t in tracked):
                    continue

                matches = face_recognition.compare_faces(known_encs, e, tolerance=0.45)
                name = "Unknown"
                folder = None
                if True in matches:
                    name = known_labels[np.argmin(face_recognition.face_distance(known_encs, e))]
                else:
                    folder = get_new_unknown(existing_unknowns)

                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, box)
                tracked.append(FaceTrack(tracker, name, folder, frame_count, box, e))

        # Draw and save
        for t in tracked:
            if t.bbox:
                x, y, w, h = map(int, t.bbox)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, t.label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                if t.label.startswith("Unknown") and frame_count - t.last_saved > SAVE_UNKNOWN_EVERY_N:
                    crop = frame[y:y+h, x:x+w]
                    fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.jpg"
                    save_path = os.path.join(UNKNOWN_DIR, t.folder, fname)
                    cv2.imwrite(save_path, crop)
                    t.last_saved = frame_count

        cv2.imshow("Face Tracker", frame)

    cap.release()
    cv2.destroyAllWindows()
    hybrid_cleanup()

if __name__ == "__main__":
    run()
