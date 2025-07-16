import os, cv2, time, numpy as np, face_recognition
from flask import Flask, Response, render_template_string

# Config
KNOWN_DIR = "ClusteredFaces"
MIN_FRAC, MAX_FRAC = 0.02, 0.25
DETECT_EVERY = 5

# Load known faces
known_encs, known_labels = [], []
for person in os.listdir(KNOWN_DIR):
    p = os.path.join(KNOWN_DIR, person)
    if os.path.isdir(p):
        for fn in os.listdir(p):
            if fn.lower().endswith(('.jpg','.png')):
                img = face_recognition.load_image_file(os.path.join(p,fn))
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_encs.append(encs[0])
                    known_labels.append(person)
print(f"[LOAD] {len(known_encs)} known")

# Flask setup
app = Flask(__name__)
cap = cv2.VideoCapture(0)

HTML = '''
<html><body>
<h1>MJPEG Face Tracker</h1>
<img src="{{ url_for('video_feed') }}" width="640"/>
</body></html>
'''

@app.route("/")
def index():
    return render_template_string(HTML)

def gen():
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % DETECT_EVERY == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, boxes)
            h,w = frame.shape[:2]
            for (top, right, bottom, left), enc in zip(boxes, encs):
                frac = (right-left)*(bottom-top)/(h*w)
                if frac < MIN_FRAC or frac > MAX_FRAC:
                    continue
                matches = face_recognition.compare_faces(known_encs, enc, tolerance=0.45)
                name = known_labels[np.argmin(face_recognition.face_distance(known_encs, enc))] if True in matches else "Unknown"
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        ret, out = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + out.tobytes() + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
