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
import json
from collections import deque
from datetime import datetime

# Backend client for face recognition
from backend_client import FaceRecognitionClient, get_client, close_client

# ============== FACE DETECTION CONFIG ==============
FACE_DETECTION_ENABLED = True
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
MIN_FACE_SIZE = (30, 30)
DETECTION_SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 3       # Reduced from 4 for better detection recall
CAPTURE_STABILITY_FRAMES = 3  # Number of valid frames required before capture
COOLDOWN_SECONDS = 3

# ============== BACKEND CONFIG ==============
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
BACKEND_ENABLED = True
# =============================================

# ============== FACE GUIDE CONFIG ==============
# Guide oval settings (as percentage of frame dimensions after rotation)
GUIDE_CENTER_X_PCT = 0.5    # Horizontal center
GUIDE_CENTER_Y_PCT = 0.40   # Slightly above vertical center
GUIDE_WIDTH_PCT = 0.55      # Oval width as % of frame width
GUIDE_HEIGHT_PCT = 0.45     # Oval height as % of frame height
# Relaxed tolerance - face just needs to be "around" the oval
GUIDE_TOLERANCE = 0.5       # Face can be within 50% of guide boundary (Balanced)
MIN_FACE_SIZE_IN_GUIDE = 0.25  # Smaller minimum for more flexibility
MAX_FACE_SIZE_IN_GUIDE = 1.5   # Larger maximum for closer faces
# Recognition timing
RESULT_DISPLAY_SECONDS = 2  # How long to show result before resuming stream
BACKEND_TIMEOUT_SECONDS = 10  # Timeout for backend response
# ===============================================

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
if face_cascade.empty():
    print("WARNING: Could not load face cascade classifier!")
    FACE_DETECTION_ENABLED = False
else:
    print("Face cascade loaded successfully")

lock = threading.Lock()
connectedDevices = set()
backend_client: FaceRecognitionClient = None
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
        self.face_in_guide = False  # Track if face is aligned with guide
        self.stability_counter = 0  # Track consecutive valid frames
        
        # Recognition state management
        # States: 'streaming', 'processing', 'success', 'error', 'timeout'
        self.recognition_state = 'streaming'
        self.recognition_result = None
        self.recognition_person_name = ''
        self.recognition_message = ''
        self.recognition_start_time = 0
        self.state_change_time = 0

    def detect_faces(self, frame):
        if not FACE_DETECTION_ENABLED or frame is None:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Improvement: Contrast enhancement for low light
        
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
        # Face detection on original QVGA frame for performance
        self.faces_detected = self.detect_faces(self.rawFrame)
        
        # Reset alignment flag for this frame processing cycle
        any_face_aligned = False
        
        # Rotate the frame first
        frame_rotated = imutils.rotate_bound(self.frame.copy(), 90)
        orig_h, orig_w = frame_rotated.shape[:2]
        
        # Upscale display frame 2x using bicubic interpolation for better quality
        SCALE_FACTOR = 2
        display_frame = cv2.resize(frame_rotated, (orig_w * SCALE_FACTOR, orig_h * SCALE_FACTOR), 
                                   interpolation=cv2.INTER_CUBIC)
        display_h, display_w = display_frame.shape[:2]
        
        # Guide oval parameters (calculated on upscaled frame dim)
        guide_center_x = int(display_w * GUIDE_CENTER_X_PCT)
        guide_center_y = int(display_h * GUIDE_CENTER_Y_PCT)
        guide_width = int(display_w * GUIDE_WIDTH_PCT)
        guide_height = int(display_h * GUIDE_HEIGHT_PCT)
        guide_axes = (guide_width // 2, guide_height // 2)
        
        face_alignment_status = "Waiting for face..."
        
        if len(self.faces_detected) > 0:
            frame_h_raw, frame_w_raw = self.rawFrame.shape[:2]
            
            for (x, y, w, h) in self.faces_detected:
                # Rotate coordinates logic
                new_x = frame_h_raw - y - h
                new_y = x
                new_w = h
                new_h = w
                
                # Calculate face center relative to guide
                face_center_x = new_x + new_w // 2
                face_center_y = new_y + new_h // 2
                
                # Check alignment with upscaled guide coordinates
                face_center_x_up = face_center_x * SCALE_FACTOR
                face_center_y_up = face_center_y * SCALE_FACTOR
                face_w_up = new_w * SCALE_FACTOR
                face_h_up = new_h * SCALE_FACTOR
                
                # Deviation from guide center (normalized by guide semi-axes)
                dx = abs(face_center_x_up - guide_center_x) / (guide_width / 2)
                dy = abs(face_center_y_up - guide_center_y) / (guide_height / 2)
                
                # Check if face size is appropriate
                face_size = max(new_w, new_h) * SCALE_FACTOR
                guide_ref_size = min(guide_width, guide_height)
                size_ratio = face_size / guide_ref_size
                
                # Determine if face is aligned
                position_ok = dx < GUIDE_TOLERANCE and dy < GUIDE_TOLERANCE
                size_ok = MIN_FACE_SIZE_IN_GUIDE < size_ratio < MAX_FACE_SIZE_IN_GUIDE
                
                color = (0, 0, 255) # Default Red
                
                if position_ok:
                    any_face_aligned = True # Mark as aligned for stability
                    color = (0, 255, 0) # Green (Visual feedback immediate)
                    
                    if self.stability_counter >= CAPTURE_STABILITY_FRAMES:
                         face_alignment_status = "CAPTURING..."
                    else:
                         dots = "." * (self.stability_counter + 1)
                         face_alignment_status = f"Hold still{dots}"
                         
                elif size_ratio < MIN_FACE_SIZE_IN_GUIDE:
                    color = (0, 255, 255)  # Yellow
                    face_alignment_status = "Move closer"
                elif position_ok and not size_ok: 
                     color = (0, 165, 255)  # Orange
                     face_alignment_status = "Move back"
                else:
                    face_alignment_status = "Center your face"
                    
                if is_hq:
                    color = (255, 0, 255)
                
                # Draw Box
                disp_x = new_x * SCALE_FACTOR
                disp_y = new_y * SCALE_FACTOR
                disp_w = new_w * SCALE_FACTOR
                disp_h = new_h * SCALE_FACTOR
                
                cv2.rectangle(display_frame, (disp_x, disp_y), (disp_x + disp_w, disp_y + disp_h), color, 2)
                label = "HQ CAPTURE" if is_hq else ("ALIGNED" if any_face_aligned else "FACE")
                cv2.putText(display_frame, label, (disp_x, disp_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Update stability counter
        if any_face_aligned:
             self.stability_counter = min(self.stability_counter + 1, CAPTURE_STABILITY_FRAMES + 1)
        else:
             self.stability_counter = 0 # Reset if alignment lost
             
        # Set trigger flag only if stable
        self.face_in_guide = (self.stability_counter >= CAPTURE_STABILITY_FRAMES)
        
        # Guide Overlay
        guide_color = (0, 255, 0) if any_face_aligned else (255, 255, 255)
        guide_thick = 4 if self.face_in_guide else 2
        cv2.ellipse(display_frame, (guide_center_x, guide_center_y), guide_axes, 0, 0, 360, guide_color, guide_thick)
        
        # Draw corners
        corner_len = 25
        # Top-left
        cv2.line(display_frame, (guide_center_x - guide_axes[0], guide_center_y - guide_axes[1] + corner_len),
                 (guide_center_x - guide_axes[0], guide_center_y - guide_axes[1]), guide_color, 2)
        cv2.line(display_frame, (guide_center_x - guide_axes[0], guide_center_y - guide_axes[1]),
                 (guide_center_x - guide_axes[0] + corner_len, guide_center_y - guide_axes[1]), guide_color, 2)
        # Top-right
        cv2.line(display_frame, (guide_center_x + guide_axes[0], guide_center_y - guide_axes[1] + corner_len),
                 (guide_center_x + guide_axes[0], guide_center_y - guide_axes[1]), guide_color, 2)
        cv2.line(display_frame, (guide_center_x + guide_axes[0], guide_center_y - guide_axes[1]),
                 (guide_center_x + guide_axes[0] - corner_len, guide_center_y - guide_axes[1]), guide_color, 2)
        # Bottom-left
        cv2.line(display_frame, (guide_center_x - guide_axes[0], guide_center_y + guide_axes[1] - corner_len),
                 (guide_center_x - guide_axes[0], guide_center_y + guide_axes[1]), guide_color, 2)
        cv2.line(display_frame, (guide_center_x - guide_axes[0], guide_center_y + guide_axes[1]),
                 (guide_center_x - guide_axes[0] + corner_len, guide_center_y + guide_axes[1]), guide_color, 2)
        # Bottom-right
        cv2.line(display_frame, (guide_center_x + guide_axes[0], guide_center_y + guide_axes[1] - corner_len),
                 (guide_center_x + guide_axes[0], guide_center_y + guide_axes[1]), guide_color, 2)
        cv2.line(display_frame, (guide_center_x + guide_axes[0], guide_center_y + guide_axes[1]),
                 (guide_center_x + guide_axes[0] - corner_len, guide_center_y + guide_axes[1]), guide_color, 2)
        
        # Status Overlay
        status = "HQ CAPTURE" if is_hq else ("PAUSED" if self.stream_paused else "STREAMING")
        cv2.putText(display_frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Alignment Instructions
        text_size = cv2.getTextSize(face_alignment_status, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = (display_w - text_size[0]) // 2
        text_y = display_h - 20
        cv2.rectangle(display_frame, (text_x - 8, text_y - text_size[1] - 8), 
                     (text_x + text_size[0] + 8, text_y + 8), (0, 0, 0), -1)
        text_color = (0, 255, 0) if any_face_aligned else (255, 255, 255)
        cv2.putText(display_frame, face_alignment_status, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        
        (flag, encodedImage) = cv2.imencode(".jpg", display_frame)
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
            
            # Store raw JPEG bytes for backend
            self.hq_raw_bytes = message
            
            tornado.ioloop.IOLoop.current().run_in_executor(
                self.executor, lambda: self.process_frames(is_hq=True)
            )
            self.awaiting_hq_frame = False
            
            # Send HQ frame to backend for recognition
            if BACKEND_ENABLED and backend_client:
                tornado.ioloop.IOLoop.current().add_callback(
                    self.send_to_backend, message
                )
            
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
            
            # Now check for faces AFTER processing - only capture if face is aligned in guide
            if FACE_DETECTION_ENABLED and self.face_in_guide and not self.stream_paused:
                current_time = time.time()
                if current_time - self.last_hq_capture_time > COOLDOWN_SECONDS:
                    self.last_hq_capture_time = current_time
                    self.write_message("CAPTURE_HQ")
                    self.stream_paused = True
                    self.stability_counter = 0  # Reset stability counter to require fresh confirmation
                    print(f"[{self.id}] Face aligned in guide! Sending CAPTURE_HQ")
    
    def send_resume_stream(self):
        """Send RESUME_STREAM command after a brief delay"""
        try:
            self.write_message("RESUME_STREAM")
            self.stream_paused = False
            print(f"Sent RESUME_STREAM to {self.id}")
        except Exception as e:
            print(f"Error sending RESUME_STREAM: {e}")
    
    async def send_to_backend(self, image_bytes: bytes):
        """
        Send HQ image to backend for face recognition.
        Manages recognition state and handles timeout.
        """
        global backend_client
        
        if not backend_client:
            print(f"[{self.id}] Backend client not available")
            self.recognition_state = 'error'
            self.recognition_message = 'Backend not available'
            self.state_change_time = time.time()
            self.schedule_resume()
            return
        
        # Set processing state
        self.recognition_state = 'processing'
        self.recognition_start_time = time.time()
        self.state_change_time = time.time()
        
        try:
            print(f"[{self.id}] Sending {len(image_bytes)} bytes to backend...")
            
            # Add timeout to backend call
            import asyncio
            try:
                result = await asyncio.wait_for(
                    backend_client.recognize_face(image_bytes),
                    timeout=BACKEND_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                print(f"[{self.id}] Backend timeout after {BACKEND_TIMEOUT_SECONDS}s")
                self.recognition_state = 'timeout'
                self.recognition_message = f'Backend timeout ({BACKEND_TIMEOUT_SECONDS}s)'
                self.state_change_time = time.time()
                self.schedule_resume()
                return
            
            # Store recognition result
            self.last_recognition_result = result
            self.recognition_result = result
            
            if result.get('success'):
                face_match = result.get('face_match', {})
                person_name = face_match.get('person_name', 'Unknown')
                similarity = face_match.get('similarity', 0)
                print(f"[{self.id}] RECOGNIZED: {person_name} (similarity: {similarity:.2f})")
                
                # Set success state
                self.recognition_state = 'success'
                self.recognition_person_name = person_name
                self.recognition_message = f'Welcome, {person_name}!'
                self.state_change_time = time.time()
                
                # Send recognition result back to ESP32-CAM
                self.write_message(json.dumps({
                    "type": "recognition_result",
                    "success": True,
                    "person_name": person_name,
                    "similarity": similarity
                }))
            else:
                liveness = result.get('liveness', {})
                message = result.get('message', 'Unknown error')
                
                if not liveness.get('is_live', True):
                    print(f"[{self.id}] SPOOF DETECTED: {liveness.get('message', 'Not live')}")
                    self.recognition_message = 'Spoof detected - Not a real face'
                else:
                    print(f"[{self.id}] NOT RECOGNIZED: {message}")
                    self.recognition_message = message
                
                # Set error state
                self.recognition_state = 'error'
                self.recognition_person_name = ''
                self.state_change_time = time.time()
                
                # Send failure result back to ESP32-CAM
                self.write_message(json.dumps({
                    "type": "recognition_result",
                    "success": False,
                    "message": message,
                    "is_live": liveness.get('is_live', False)
                }))
                
            # Store in captures with recognition result
            hq_captures[self.id] = {
                'image': self.hq_frame,
                'timestamp': datetime.now(),
                'faces': len(self.faces_detected),
                'recognition': result
            }
            
            print(f"[{self.id}] Processing time: {result.get('processing_time_ms', 0):.1f}ms")
            
            # Schedule auto-resume after showing result
            self.schedule_resume()
            
        except Exception as e:
            print(f"[{self.id}] Backend error: {e}")
            self.recognition_state = 'error'
            self.recognition_message = f'Error: {str(e)}'
            self.state_change_time = time.time()
            self.schedule_resume()
    
    def schedule_resume(self):
        """Schedule stream resume after RESULT_DISPLAY_SECONDS"""
        tornado.ioloop.IOLoop.current().call_later(
            RESULT_DISPLAY_SECONDS,
            self.resume_after_result
        )
    
    def resume_after_result(self):
        """Resume streaming after showing recognition result"""
        print(f"[{self.id}] Resuming stream after result display")
        self.recognition_state = 'streaming'
        self.recognition_result = None
        self.recognition_person_name = ''
        self.recognition_message = ''
        self.send_resume_stream()

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
                    "has_hq_capture": c.id in hq_captures,
                    "recognition_state": getattr(c, 'recognition_state', 'streaming'),
                    "recognition_person_name": getattr(c, 'recognition_person_name', ''),
                    "recognition_message": getattr(c, 'recognition_message', '')
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
    <title>Smart Attendance System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --bg-color: #1a1a2e;
            --card-bg: rgba(22, 33, 62, 0.9);
            --primary: #00d4ff;
            --success: #00ff88;
            --error: #ff3366;
            --text-main: #ffffff;
            --text-sub: #8892b0;
        }

        body {
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: var(--text-main);
            margin: 0;
            padding: 0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            transition: background 0.5s ease;
        }

        /* Dynamic Theme Classes */
        body.theme-success {
            background: linear-gradient(135deg, #051c10, #0c3a23, #022010);
        }
        body.theme-error {
            background: linear-gradient(135deg, #2a0a10, #4a1218, #200508);
        }

        .dashboard {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 30px;
            width: 90%;
            max-width: 1200px;
            height: 85vh;
        }

        .video-section {
            background: var(--card-bg);
            border-radius: 24px;
            overflow: hidden;
            position: relative;
            box-shadow: 0 20px 50px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.1);
            display: flex;
            flex-direction: column;
        }

        .header {
            padding: 20px 30px;
            background: rgba(0,0,0,0.2);
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }

        .header h1 {
            margin: 0;
            font-size: 1.5rem;
            font-weight: 600;
            letter-spacing: 1px;
            color: var(--primary);
        }
        
        .theme-success .header h1 { color: var(--success); }
        .theme-error .header h1 { color: var(--error); }

        .live-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9rem;
            color: var(--text-sub);
            font-weight: 600;
            text-transform: uppercase;
        }

        .dot {
            width: 10px;
            height: 10px;
            background-color: var(--error);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--error);
        }
        
        .active .dot {
            background-color: var(--success);
            box-shadow: 0 0 10px var(--success);
            animation: pulse 2s infinite;
        }

        .video-container {
            flex: 1;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #000;
            overflow: hidden;
        }

        #video-feed, #captured-image {
            width: 100%;
            height: 100%;
            object-fit: contain;
            position: absolute;
            top: 0;
            left: 0;
            transition: opacity 0.3s ease;
        }
        
        /* Visibility Toggles */
        .show-video #captured-image { opacity: 0; z-index: 1; }
        .show-video #video-feed { opacity: 1; z-index: 2; }
        
        .show-capture #video-feed { opacity: 0; z-index: 1; }
        .show-capture #captured-image { opacity: 1; z-index: 2; }

        .status-panel {
            background: var(--card-bg);
            border-radius: 24px;
            padding: 40px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 20px 50px rgba(0,0,0,0.5);
            transition: all 0.3s ease;
        }
        
        .theme-success .status-panel { border-color: rgba(0, 255, 136, 0.3); background: rgba(0, 50, 25, 0.8); }
        .theme-error .status-panel { border-color: rgba(255, 51, 102, 0.3); background: rgba(50, 10, 20, 0.8); }

        /* Status Animations */
        .loader {
            position: relative;
            width: 120px;
            height: 120px;
            margin-bottom: 30px;
        }

        .ring {
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            border: 4px solid transparent;
            border-top-color: var(--primary);
            animation: spin 1s linear infinite;
        }
        
        .ring:nth-child(2) {
            width: 80%;
            height: 80%;
            top: 10%;
            left: 10%;
            border-top-color: #ff00ff;
            animation: spin 1.5s linear infinite reverse;
        }

        /* Success Icon */
        .icon-box {
            font-size: 5rem;
            margin-bottom: 20px;
            display: none;
        }
        .success-icon { color: var(--success); }
        .error-icon { color: var(--error); }

        .status-text {
            font-size: 1.2rem;
            color: var(--text-sub);
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .result-text {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-main);
            margin: 0;
            text-shadow: 0 0 20px rgba(0,0,0,0.5);
        }

        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }

        /* State specific visibility */
        .state-streaming .loader { display: none; }
        .state-streaming .icon-box { display: none; }
        .state-streaming .result-text { display: none; }
        
        .state-processing .loader { display: block; }
        .state-processing .icon-box { display: none; }
        
        .state-success .loader { display: none; }
        .state-success .icon-box.success-icon { display: block; filter: drop-shadow(0 0 20px var(--success)); }
        
        .state-error .loader { display: none; }
        .state-error .icon-box.error-icon { display: block; filter: drop-shadow(0 0 20px var(--error)); }
        
        /* Guide text overlay */
        .guide-overlay {
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.6);
            padding: 10px 20px;
            border-radius: 20px;
            color: #fff;
            font-size: 1.1rem;
            z-index: 10;
            border: 1px solid rgba(255,255,255,0.2);
            pointer-events: none;
        }

    </style>
</head>
<body class="theme-default">
    <div class="dashboard">
        <!-- Video Section -->
        <div class="video-section">
            <div class="header">
                <h1>SMART ATTENDANCE</h1>
                <div class="live-indicator active">
                    <div class="dot"></div>
                    <span id="conn-status">LIVE SYSTEM</span>
                </div>
            </div>
            
            <div class="video-container show-video" id="video-wrapper">
                <img id="video-feed" src="" alt="Video Feed">
                <img id="captured-image" src="" alt="Captured Frame">
                <div class="guide-overlay" id="guide-msg">Ready for attendance</div>
            </div>
        </div>

        <!-- Status Panel -->
        <div class="status-panel state-streaming" id="status-panel">
            <!-- Processing Animation -->
            <div class="loader">
                <div class="ring"></div>
                <div class="ring"></div>
            </div>

            <!-- Success Icon -->
            <div class="icon-box success-icon">✓</div>
            
            <!-- Error Icon -->
            <div class="icon-box error-icon">✗</div>

            <div class="status-text" id="status-label">SYSTEM READY</div>
            <h2 class="result-text" id="result-label">Waiting...</h2>
        </div>
    </div>

    <script>
        const videoWrapper = document.getElementById('video-wrapper');
        const videoFeed = document.getElementById('video-feed');
        const capturedImage = document.getElementById('captured-image');
        const statusPanel = document.getElementById('status-panel');
        const statusLabel = document.getElementById('status-label');
        const resultLabel = document.getElementById('result-label');
        const guideMsg = document.getElementById('guide-msg');
        
        let currentDeviceId = null;
        let activeStreamId = null;
        let lastState = 'streaming';

        async function updateDashboard() {
            try {
                const res = await fetch('/devices');
                const data = await res.json();
                
                if (data.devices.length > 0) {
                    const device = data.devices[0];
                    currentDeviceId = device.id;
                    
                    // Update video feed URL only if device changed
                    if (activeStreamId !== device.id) {
                        console.log("Switching stream to:", device.id);
                        videoFeed.src = '/video_feed/' + encodeURIComponent(device.id);
                        activeStreamId = device.id;
                    }
                    
                    const state = device.recognition_state;
                    
                    // Handle State Changes
                    updateUIState(state, device);
                    
                } else {
                    statusLabel.textContent = "OFFLINE";
                    resultLabel.textContent = "No Device";
                }
            } catch (e) {
                console.error("Dashboard update error:", e);
            }
        }

        function updateUIState(state, device) {
            // Update classes
            statusPanel.className = 'status-panel state-' + state;
            
            // Theme reset/set
            if (state === 'success') {
                document.body.className = 'theme-success';
            } else if (state === 'error' || state === 'timeout') {
                document.body.className = 'theme-error';
            } else {
                document.body.className = 'theme-default';
            }

            // UI Content Update
            if (state === 'streaming') {
                videoWrapper.className = 'video-container show-video';
                statusLabel.textContent = "READY";
                resultLabel.textContent = "Look at Camera";
                guideMsg.style.display = 'block';
                guideMsg.textContent = "Position face in oval";
                
            } else if (state === 'processing') {
                videoWrapper.className = 'video-container show-capture';
                // Update capture image
                capturedImage.src = '/hq_frame/' + device.id + '?' + Date.now(); // Force refresh
                
                statusLabel.textContent = "PROCESSING";
                resultLabel.textContent = "Identifying...";
                guideMsg.style.display = 'none';
                
            } else if (state === 'success') {
                videoWrapper.className = 'video-container show-capture';
                statusLabel.textContent = "TOTAL SUCCESS";
                resultLabel.textContent = device.recognition_person_name;
                guideMsg.style.display = 'none';
                
            } else if (state === 'error' || state === 'timeout') {
                videoWrapper.className = 'video-container show-capture';
                statusLabel.textContent = "ACCESS DENIED";
                resultLabel.textContent = device.recognition_message || "Not Recognized";
                guideMsg.style.display = 'none';
            }
            
            lastState = state;
        }

        // Start polling
        setInterval(updateDashboard, 500);
        updateDashboard();
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


async def init_backend_client():
    """Initialize backend client on startup."""
    global backend_client
    backend_client = get_client(BACKEND_URL)
    
    # Check backend health
    health = await backend_client.health_check()
    if health.get('status') == 'online':
        print(f'✓ Backend connected: {BACKEND_URL}')
        print(f'  - Liveness model: {"✓" if health.get("liveness_model") else "✗"}')
        print(f'  - Face recognition: {"✓" if health.get("face_recognition_model") else "✗"}')
        print(f'  - Known faces: {health.get("known_faces_count", 0)}')
    else:
        print(f'✗ Backend not available: {health.get("error", "Unknown error")}')
        if not BACKEND_ENABLED:
            print('  (BACKEND_ENABLED is False - recognition disabled)')


if __name__ == "__main__":
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(3000, address="0.0.0.0")
    
    print('=' * 60)
    print('*** Face Detection + Recognition Pipeline Server ***')
    print('=' * 60)
    print(f'\nFace Detection: {"ENABLED" if FACE_DETECTION_ENABLED else "DISABLED"}')
    print(f'Backend: {BACKEND_URL}')
    print(f'Cooldown: {COOLDOWN_SECONDS}s between captures')
    print('\nPipeline Flow:')
    print('  1. ESP32-CAM -> QVGA Stream -> Face Detection (Haar Cascade)')
    print('  2. Face Detected -> CAPTURE_HQ -> VGA Frame')
    print('  3. HQ Frame -> Backend API:')
    print('     a. Liveness Check (MiniFASNetV2)')
    print('     b. Face Recognition (InsightFace buffalo_l)')
    print('     c. Cosine Similarity (512D embeddings)')
    print('  4. Recognition Result -> Back to ESP32-CAM')
    print('\nDashboard: http://localhost:3000/view')
    print('=' * 60)
    
    # Initialize backend client
    tornado.ioloop.IOLoop.current().run_sync(init_backend_client)
    
    tornado.ioloop.IOLoop.current().start()
