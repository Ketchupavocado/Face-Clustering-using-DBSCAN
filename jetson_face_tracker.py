from flask import Flask, Response
import cv2
import threading

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # USB camera

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # MJPEG stream format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>Jetson USB Camera</title>
      </head>
      <body>
        <h1>Live Camera Stream</h1>
        <img src="/video_feed" width="720" height="480">
      </body>
    </html>
    """

def run_flask():
    app.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == '__main__':
    t = threading.Thread(target=run_flask)
    t.daemon = True
    t.start()

    print("[INFO] MJPEG stream running at http://<jetson-ip>:5000")
    t.join()
