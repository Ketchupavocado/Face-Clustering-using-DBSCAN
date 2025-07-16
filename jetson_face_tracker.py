import os, cv2, time, shutil
import face_recognition, numpy as np
from datetime import datetime
from threading import Thread

CLUSTERED_DIR = 'ClusteredFaces/'
UNKNOWN_DIR = 'UnknownFaces/'
os.makedirs(CLUSTERED_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

RELOAD_INTERVAL = 10
SAVE_UNKNOWN_FRAMES = 30
MIN_BOX_FRAC, MAX_BOX_FRAC = 0.02, 0.25
DETECT_INTERVAL = 5
TRACKER_MAX_AGE = 150

class FaceTrack:
    def __init__(self, tracker, label, folder, last_saved, bbox, enc):
        self.tracker = tracker
        self.label = label
        self.folder = folder
        self.last_saved = last_saved
        self.bbox = bbox
        self.enc = enc

def intersect(a,b,iou=0.4):
    if a is None or b is None: return False
    x1,y1,w1,h1 = a; x2,y2,w2,h2 = b
    xi,yi = max(x1,x2), max(y1,y2)
    x2i,y2i = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter = max(0, x2i-xi)*max(0, y2i-yi)
    return inter / (w1*h1 + w2*h2 - inter) > iou

def load_known_faces():
    encs, labels = [], []
    for label in os.listdir(CLUSTERED_DIR):
        p = os.path.join(CLUSTERED_DIR, label)
        if os.path.isdir(p):
            for fn in os.listdir(p):
                if fn.lower().endswith(('.jpg','.png')):
                    try:
                        img = face_recognition.load_image_file(os.path.join(p,fn))
                        locs = face_recognition.face_locations(img)
                        if locs:
                            encs.append(face_recognition.face_encodings(img,locs)[0])
                            labels.append(label)
                    except: pass
    print(f"[LOAD] {len(encs)} known faces")
    return encs, labels

def get_new_unknown(existing):
    i=1
    while f"Unknown_{i}" in existing: i+=1
    name=f"Unknown_{i}"; existing.add(name)
    os.makedirs(os.path.join(UNKNOWN_DIR, name), exist_ok=True)
    return name

def update_named_unknowns():
    moved=False
    for fld in os.listdir(UNKNOWN_DIR):
        src=os.path.join(UNKNOWN_DIR,fld); dst=os.path.join(CLUSTERED_DIR,fld)
        if os.path.isdir(src):
            if os.path.exists(dst):
                for f in os.listdir(src):
                    shutil.move(os.path.join(src,f), os.path.join(dst,f))
                shutil.rmtree(src); print(f"[MERGE] {fld}")
            else:
                shutil.move(src,dst); print(f"[MOVE] {fld}")
            moved=True
    return moved

def cleanup_on_exit():
    def keep_best():
        for lbl in os.listdir(CLUSTERED_DIR):
            p=os.path.join(CLUSTERED_DIR,lbl)
            if os.path.isdir(p):
                files=[os.path.join(p,f) for f in os.listdir(p)]
                files=[f for f in files if f.lower().endswith(('.jpg','.png'))]
                if len(files)>2:
                    files.sort(key=lambda x:os.path.getsize(x), reverse=True)
                    for rm in files[2:]: os.remove(rm)
    keep_best()
    for fld in os.listdir(UNKNOWN_DIR):
        shutil.rmtree(os.path.join(UNKNOWN_DIR,fld))
    print("Cleanup done.")

def jetson_camera_source():
    gst = ("nvarguscamerasrc ! video/x-raw(memory:NVMM),"
           "width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx !"
           "videoconvert ! appsink")
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return cap

def run():
    cap = jetson_camera_source()
    known_encs, known_labels = load_known_faces()
    tracks=[]
    frame_count=0; last_reload=time.time()
    existing_unknowns=set(os.listdir(UNKNOWN_DIR))

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count+=1
        area=frame.shape[0]*frame.shape[1]

        # reload knowns
        if time.time()-last_reload>RELOAD_INTERVAL:
            if update_named_unknowns():
                known_encs, known_labels=load_known_faces()
            last_reload=time.time()

        # update trackages
        for t in tracks[:]:
            ok, box = t.tracker.update(frame)
            if ok: t.bbox=box
            else: tracks.remove(t)

        # prune old
        tracks=[t for t in tracks if frame_count - t.last_saved < TRACKER_MAX_AGE]

        # detection
        if frame_count % DETECT_INTERVAL == 0 or not tracks:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)
            for (t_,r,b,l),enc in zip(locs,encs):
                box=(l,t_,r-l,b-t_)
                fbox = box[2]*box[3]
                if fbox/area<MIN_BOX_FRAC or fbox/area>MAX_BOX_FRAC: continue
                skip=False
                for t in tracks:
                    if t.bbox and intersect(box,t.bbox,0.5):
                        if t.enc is not None and face_recognition.face_distance([t.enc],enc)[0]<0.45:
                            skip=True; break
                if skip: continue
                matches=face_recognition.compare_faces(known_encs,enc,tolerance=0.45)
                name = known_labels[np.argmin(face_recognition.face_distance(known_encs,enc))] if True in matches else "Unknown"
                folder = None
                if name=="Unknown":
                    folder = get_new_unknown(existing_unknowns)
                tracker=cv2.TrackerCSRT_create()
                tracker.init(frame,box)
                tracks.append(FaceTrack(tracker,name,folder,frame_count,box,enc))

        # save unknowns
        for t in tracks:
            if hasattr(t,'bbox') and t.bbox and t.label=="Unknown" and frame_count - t.last_saved > SAVE_UNKNOWN_FRAMES:
                x,y,w,h = map(int, t.bbox)
                crop = frame[y:y+h, x:x+w]
                if crop.size>0:
                    fname=datetime.now().strftime("%Y%m%d_%H%M%S%f")+".jpg"
                    path=os.path.join(UNKNOWN_DIR, t.folder, fname)
                    cv2.imwrite(path, crop)
                    t.last_saved=frame_count

        # optional: send frames via HTTP, MQTT, etc.

    cap.release()
    cleanup_on_exit()

if __name__=="__main__":
    run()
