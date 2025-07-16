import os, cv2, asyncio
import face_recognition
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole
from aiortc.contrib.signaling import BYE
from aiortc.contrib.signaling import object_to_string, object_from_string
import numpy as np

# Face settings
KNOWN_DIR = "ClusteredFaces"
MIN_FRAC, MAX_FRAC = 0.02, 0.25
DETECT_EVERY = 5

# Load known faces
known_encs, known_labels = [], []
for person in os.listdir(KNOWN_DIR):
    pdir = os.path.join(KNOWN_DIR, person)
    if os.path.isdir(pdir):
        for fn in os.listdir(pdir):
            if fn.lower().endswith(('.jpg','.png')):
                img = face_recognition.load_image_file(os.path.join(pdir,fn))
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_encs.append(encs[0])
                    known_labels.append(person)
print(f"[LOAD] {len(known_encs)} known faces")

# VideoTrack subclass to perform detection
class FaceTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.counter = 0

    async def recv(self):
        frame = await self.next_timestamp()
        ret, img = self.cap.read()
        if not ret:
            raise asyncio.CancelledError

        self.counter += 1
        rgb = img[:, :, ::-1]
        if self.counter % DETECT_EVERY == 0:
            boxes = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, boxes)
            h, w = img.shape[:2]
            area = h*w
            for (top, right, bottom, left), enc in zip(boxes, encs):
                frac = (right-left)*(bottom-top)/area
                if frac < MIN_FRAC or frac > MAX_FRAC:
                    continue
                matches = face_recognition.compare_faces(known_encs, enc, tolerance=0.45)
                name = known_labels[np.argmin(face_recognition.face_distance(known_encs, enc))] if True in matches else "Unknown"
                cv2.rectangle(img, (left, top), (right, bottom), (0,255,0), 2)
                cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # convert to VideoFrame
        from av import VideoFrame
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

# Flask + aiortc signaling routes
routes = web.RouteTableDef()

@routes.get("/")
async def index(request):
    content = open("static/index.html", "r").read()
    return web.Response(content_type="text/html", text=content)

@routes.post("/offer")
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pc.addTrack(FaceTrack())

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    })

app = web.Application()
app.add_routes(routes)
app.router.add_static("/static/", "static/")

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=5000)
