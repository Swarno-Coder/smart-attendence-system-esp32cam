import tornado.httpserver
import tornado.websocket
import tornado.concurrent
import tornado.ioloop
import tornado.web
import tornado.gen
import threading
import asyncio
import socket
import numpy as np
import imutils
import copy
import time
import cv2
import os
from collections import deque
from datetime import datetime

# ============== FACE DETECTION CONFIG ==============
FACE_DETECTION_ENABLED = True
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
MIN_FACE_SIZE = (30, 30)
DETECTION_SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 4
COOLDOWN_SECONDS = 3
# ===================================================

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if face_cascade.empty():
    print("WARNING: Could not load face cascade classifier!")
    FACE_DETECTION_ENABLED = False
else:
    print("Face cascade loaded successfully")

lock = threading.Lock()
connectedDevices = set()
hq_captures = {}


class WSHandler(tornado.websocket.WebSocketHandler):
    def __init__(self, *args, **kwargs):
        super(WSHandler, self).__init__(*args, **kwargs)
        self.outputFrame = None
        self.frame = None
        self.rawFrame = None
        self.id = None
        self.executor = tornado.concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.last_hq_capture_time = 0
        self.faces_detected = []
        self.hq_frame = None
        self.last_stream_frame = None  # Store last QVGA frame before HQ capture
        self.awaiting_hq_frame = False  # Flag: next binary is HQ frame
        self.stream_paused = False  # Track if stream is paused

    def detect_faces(self, frame):
        if not FACE_DETECTION_ENABLED or frame is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=DETECTION_SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def process_frames(self, is_hq=False):
        if self.frame is None:
            return
        
        self.rawFrame = self.frame.copy()
        self.faces_detected = self.detect_faces(self.rawFrame)
        
        frame = imutils.rotate_bound(self.frame.copy(), 90)
        
        # Draw face rectangles
        if len(self.faces_detected) > 0:
            h, w = self.rawFrame.shape[:2]
            for (x, y, fw, fh) in self.faces_detected:
                new_x = y
                new_y = w - x - fw
                new_w = fh
                new_h = fw
                color = (0, 0, 255) if is_hq else (0, 255, 0)  # Red for HQ, Green for stream
                cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), color, 2)
                label = "HQ CAPTURE" if is_hq else "FACE"
                cv2.putText(frame, label, (new_x, new_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add status overlay
        status = "HQ CAPTURE" if is_hq else ("PAUSED" if self.stream_paused else "STREAMING")
        cv2.putText(frame, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            return
        
        encoded_bytes = encodedImage.tobytes()
        
        if is_hq:
            self.hq_frame = encoded_bytes
            hq_captures[self.id] = {
                'image': encoded_bytes,
                'timestamp': datetime.now(),
                'faces': len(self.faces_detected)
            }
            print(f"HQ frame stored for {self.id}: {len(self.faces_detected)} face(s), {len(encoded_bytes)} bytes")
        else:
            self.outputFrame = encoded_bytes
            if not self.stream_paused:
                self.last_stream_frame = encoded_bytes

    def open(self):
        print('new connection')
        connectedDevices.add(self)
        print(f'Total connected devices: {len(connectedDevices)}')

    def on_message(self, message):
        # Handle text messages (commands/markers)
        if isinstance(message, str):
            if self.id is None:
                self.id = message
                print(f'Device registered with ID: {self.id}')
            elif message == "HQ_FRAME_START":
                self.awaiting_hq_frame = True
                self.stream_paused = True
                print(f"HQ frame incoming for {self.id}...")
            return
        
        # Handle binary messages (frames)
        self.frame = cv2.imdecode(np.frombuffer(message, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.frame is None:
            print('Failed to decode frame')
            return
        
        frame_size = len(message)
        
        if self.awaiting_hq_frame:
            # This is the HQ VGA frame
            print(f"Received HQ frame: {frame_size} bytes")
            tornado.ioloop.IOLoop.current().run_in_executor(
                self.executor, lambda: self.process_frames(is_hq=True)
            )
            self.awaiting_hq_frame = False
            
            # Send RESUME_STREAM command
            tornado.ioloop.IOLoop.current().add_callback(self.send_resume_stream)
        else:
            # Normal QVGA streaming frame - process synchronously for reliable face detection
            self.frame_count = getattr(self, 'frame_count', 0) + 1
            
            # Log every 30 frames
            if self.frame_count % 30 == 0:
                print(f"[{self.id}] Received {self.frame_count} QVGA frames ({frame_size} bytes)")
            
            # Process frame synchronously (face detection is fast enough for QVGA)
            self.process_frames(is_hq=False)
            
            # Now check for faces AFTER processing
            if FACE_DETECTION_ENABLED and len(self.faces_detected) > 0 and not self.stream_paused:
                current_time = time.time()
                if current_time - self.last_hq_capture_time > COOLDOWN_SECONDS:
                    self.last_hq_capture_time = current_time
                    self.write_message("CAPTURE_HQ")
                    self.stream_paused = True
                    print(f"[{self.id}] Face detected! Sending CAPTURE_HQ")
    
    def send_resume_stream(self):
        """Send RESUME_STREAM command after a brief delay"""
        try:
            self.write_message("RESUME_STREAM")
            self.stream_paused = False
            print(f"Sent RESUME_STREAM to {self.id}")
        except Exception as e:
            print(f"Error sending RESUME_STREAM: {e}")

    def on_close(self):
        print('connection closed')
        connectedDevices.discard(self)

    def check_origin(self, origin):
        return True
    

class StreamHandler(tornado.web.RequestHandler):
    @tornado.gen.coroutine
    def get(self, slug):
        print(f'Stream requested for device: {slug}')
        
        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0')
        self.set_header('Pragma', 'no-cache')
        self.set_header('Content-Type', 'multipart/x-mixed-replace;boundary=--jpgboundary')
        self.set_header('Connection', 'close')

        client = None
        for c in connectedDevices:
            if c.id == slug:
                client = c
                break
        
        if client is None:
            self.write(f'Device {slug} not connected')
            return
        
        frame_count = 0
        while client in connectedDevices:
            # Use last_stream_frame if paused, otherwise use current outputFrame
            jpgData = client.outputFrame if not client.stream_paused else client.last_stream_frame
            
            if jpgData is None:
                yield tornado.gen.sleep(0.05)
                continue
            
            try:
                self.write(b"--jpgboundary\r\n")
                self.write(b"Content-Type: image/jpeg\r\n")
                self.write(("Content-Length: %d\r\n\r\n" % len(jpgData)).encode())
                self.write(jpgData)
                self.write(b"\r\n")
                yield self.flush()
                frame_count += 1
            except Exception as e:
                print(f'Stream error: {e}')
                break
            
            yield tornado.gen.sleep(0.05)


class ButtonHandler(tornado.web.RequestHandler):
    def post(self):
        data = self.get_argument("data")
        for client in connectedDevices:
            client.write_message(data)

    def get(self):
        self.write("This is a POST-only endpoint.")


class TemplateHandler(tornado.web.RequestHandler):
    def get(self):
        deviceIds = [d.id for d in connectedDevices]
        self.render(os.path.sep.join(
            [os.path.dirname(__file__), "templates", "index.html"]), 
            url="http://localhost:3000/video_feed/", deviceIds=deviceIds)


class CaptureHQHandler(tornado.web.RequestHandler):
    def get(self, device_id):
        for client in connectedDevices:
            if client.id == device_id:
                client.write_message("CAPTURE_HQ")
                client.stream_paused = True
                self.write({"status": "ok", "message": f"CAPTURE_HQ sent to {device_id}"})
                return
        self.set_status(404)
        self.write({"status": "error", "message": f"Device {device_id} not found"})


class HQFrameHandler(tornado.web.RequestHandler):
    def get(self, device_id):
        if device_id in hq_captures:
            capture = hq_captures[device_id]
            self.set_header("Content-Type", "image/jpeg")
            self.set_header("X-Faces-Detected", str(capture['faces']))
            self.set_header("X-Capture-Time", capture['timestamp'].isoformat())
            self.write(capture['image'])
            return
        self.set_status(404)
        self.write({"status": "error", "message": "No HQ capture available"})


class DevicesHandler(tornado.web.RequestHandler):
    def get(self):
        devices = []
        for c in connectedDevices:
            if c.id:
                devices.append({
                    "id": c.id, 
                    "connected": True,
                    "faces_detected": len(c.faces_detected) if hasattr(c, 'faces_detected') else 0,
                    "stream_paused": c.stream_paused,
                    "has_hq_capture": c.id in hq_captures
                })
        self.write({"devices": devices, "count": len(devices), "detection_enabled": FACE_DETECTION_ENABLED})


class CapturesHandler(tornado.web.RequestHandler):
    def get(self):
        captures = []
        for device_id, capture in hq_captures.items():
            captures.append({
                "device_id": device_id,
                "timestamp": capture['timestamp'].isoformat(),
                "faces": capture['faces']
            })
        self.write({"captures": captures, "count": len(captures)})


class CapturesViewHandler(tornado.web.RequestHandler):
    def get(self):
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Face Detection Pipeline</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #eee; padding: 20px; margin: 0; min-height: 100vh; }
        h1 { color: #00d4ff; margin-bottom: 5px; }
        .subtitle { color: #888; margin-bottom: 20px; }
        .container { display: flex; gap: 20px; flex-wrap: wrap; }
        .box { background: rgba(22, 33, 62, 0.9); border-radius: 15px; padding: 20px; box-shadow: 0 8px 32px rgba(0,212,255,0.15); backdrop-filter: blur(10px); border: 1px solid rgba(0,212,255,0.2); flex: 1; min-width: 300px; }
        .box h2 { color: #00d4ff; margin-top: 0; font-size: 1.1em; display: flex; align-items: center; gap: 8px; }
        img { max-width: 100%; border-radius: 8px; background: #000; }
        .status { padding: 8px 15px; border-radius: 20px; display: inline-block; font-weight: bold; margin: 10px 0; }
        .status.streaming { background: #00c853; color: #000; }
        .status.paused { background: #ff9800; color: #000; }
        .status.waiting { background: #2196f3; color: #fff; }
        .status.offline { background: #f44336; color: #fff; }
        .info-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 15px; }
        .info-item { background: rgba(0,212,255,0.1); padding: 10px; border-radius: 8px; text-align: center; }
        .info-item .label { font-size: 0.8em; color: #888; }
        .info-item .value { font-size: 1.5em; color: #00d4ff; font-weight: bold; }
        .capture-time { font-size: 0.9em; color: #888; margin-top: 10px; }
        #hq-image { min-height: 200px; display: flex; align-items: center; justify-content: center; color: #666; }
    </style>
</head>
<body>
    <h1>🎯 Face Detection Pipeline</h1>
    <p class="subtitle">ESP32-CAM → Face Detection → HQ Capture → Cloud Ready</p>
    
    <div class="container">
        <div class="box">
            <h2>📹 Live Stream (QVGA)</h2>
            <div id="stream-status" class="status offline">Connecting...</div>
            <img id="stream" src="/video_feed/deviceId" onerror="this.style.opacity='0.3'">
            <div class="info-grid">
                <div class="info-item">
                    <div class="label">Faces</div>
                    <div class="value" id="face-count">0</div>
                </div>
                <div class="info-item">
                    <div class="label">State</div>
                    <div class="value" id="stream-state">-</div>
                </div>
            </div>
        </div>
        
        <div class="box">
            <h2>📸 HQ Capture (VGA)</h2>
            <div id="hq-status" class="status waiting">Waiting for face...</div>
            <div id="hq-image">
                <img id="hq-capture" style="display:none;">
                <span id="hq-placeholder">No capture yet</span>
            </div>
            <div class="capture-time" id="capture-time"></div>
            <div class="info-grid">
                <div class="info-item">
                    <div class="label">Captures</div>
                    <div class="value" id="capture-count">0</div>
                </div>
                <div class="info-item">
                    <div class="label">Faces in HQ</div>
                    <div class="value" id="hq-faces">-</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        async function updateStatus() {
            try {
                const res = await fetch('/devices');
                const data = await res.json();
                
                if (data.devices.length > 0) {
                    const d = data.devices[0];
                    document.getElementById('face-count').textContent = d.faces_detected;
                    document.getElementById('stream-state').textContent = d.stream_paused ? 'PAUSED' : 'LIVE';
                    
                    const statusEl = document.getElementById('stream-status');
                    if (d.stream_paused) {
                        statusEl.className = 'status paused';
                        statusEl.textContent = 'Capturing HQ...';
                    } else {
                        statusEl.className = 'status streaming';
                        statusEl.textContent = 'Streaming';
                    }
                    
                    if (d.has_hq_capture) {
                        document.getElementById('hq-status').className = 'status streaming';
                        document.getElementById('hq-status').textContent = 'Captured!';
                    }
                } else {
                    document.getElementById('stream-status').className = 'status offline';
                    document.getElementById('stream-status').textContent = 'No device';
                }
            } catch (e) {
                console.error('Status update error:', e);
            }
        }
        
        async function updateCaptures() {
            try {
                const res = await fetch('/captures');
                const data = await res.json();
                document.getElementById('capture-count').textContent = data.count;
                
                if (data.count > 0) {
                    const latest = data.captures[data.captures.length - 1];
                    document.getElementById('hq-faces').textContent = latest.faces;
                    document.getElementById('capture-time').textContent = 'Last: ' + new Date(latest.timestamp).toLocaleTimeString();
                    
                    // Refresh HQ image
                    const img = document.getElementById('hq-capture');
                    img.src = '/hq_frame/' + latest.device_id + '?' + Date.now();
                    img.style.display = 'block';
                    document.getElementById('hq-placeholder').style.display = 'none';
                }
            } catch (e) {
                console.error('Captures update error:', e);
            }
        }
        
        setInterval(updateStatus, 500);
        setInterval(updateCaptures, 1000);
        updateStatus();
        updateCaptures();
    </script>
</body>
</html>
        """
        self.write(html)


application = tornado.web.Application([
    (r'/video_feed/([^/]+)', StreamHandler),
    (r'/ws', WSHandler),
    (r'/button', ButtonHandler),
    (r'/capture_hq/([^/]+)', CaptureHQHandler),
    (r'/hq_frame/([^/]+)', HQFrameHandler),
    (r'/devices', DevicesHandler),
    (r'/captures', CapturesHandler),
    (r'/view', CapturesViewHandler),
    (r'/', TemplateHandler),
])


def get_all_ips():
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips and not ip.startswith('127.'):
                ips.append(ip)
    except:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        if ip not in ips:
            ips.append(ip)
        s.close()
    except:
        pass
    return ips


if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(3000, address="0.0.0.0")
    
    print('=' * 60)
    print('*** Face Detection Pipeline Server - Port 3000 ***')
    print('=' * 60)
    print(f'\nFace Detection: {"ENABLED" if FACE_DETECTION_ENABLED else "DISABLED"}')
    print(f'Cooldown: {COOLDOWN_SECONDS}s between captures')
    print('\nState Machine Flow:')
    print('  STREAMING -> Face Detected -> CAPTURE_HQ -> PAUSED')
    print('  -> HQ_FRAME_START -> Receive VGA -> RESUME_STREAM -> STREAMING')
    print('\nDashboard: http://localhost:3000/view')
    print('=' * 60)
    
    tornado.ioloop.IOLoop.current().start()
