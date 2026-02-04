# 🎓 Smart Distributed Attendance System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://python.org)
[![ESP32-CAM](https://img.shields.io/badge/ESP32--CAM-Supported-orange.svg)](https://www.espressif.com/)

A **real-time face recognition attendance system** built with ESP32-CAM, Python, and deep learning. The system uses a distributed 3-layer architecture for scalable, secure, and efficient attendance tracking in educational institutions or workplaces.

![Smart Attendance System](https://img.shields.io/badge/Status-Active-brightgreen)

---

## ✨ Features

- **🔐 Face Recognition** - AI-powered face detection and recognition using InsightFace/ArcFace
- **👁️ Liveness Detection** - Anti-spoofing protection against photos/videos using MiniVision model
- **📡 Real-time Streaming** - Live video feed from ESP32-CAM with WebSocket communication
- **📊 Attendance Logging** - SQLite-based attendance tracking with entry/exit management
- **🚫 Duplicate Prevention** - 60-second cooldown to prevent duplicate scans
- **⚠️ Anomaly Detection** - Flags night access, weekend access, and excessive scans
- **🖥️ Modern Web UI** - Glassmorphism design with real-time status updates
- **📱 Responsive Dashboard** - Works on desktop and mobile browsers

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SMART ATTENDANCE SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │  ESP32-CAM   │───▶│    Local     │───▶│    Cloud     │     │
│   │  (Embedded)  │    │   Gateway    │    │   Backend    │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                  │
│   • Camera capture     • Face detection    • Face recognition   │
│   • JPEG streaming     • YuNet detector    • Liveness check     │
│   • WiFi connection    • Stream serving    • Attendance DB      │
│   • HQ frame capture   • Backend proxy     • API endpoints      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
smart_attendence_system_esp32cam/
├── src/
│   ├── esp32cam_embedded_layer/     # ESP32-CAM Arduino firmware
│   │   └── esp32cam_ws_stream/
│   │       └── esp32cam_ws_stream.ino
│   │
│   ├── local_gateway_layer/         # Python local gateway server
│   │   ├── websockets_stream.py     # Main gateway with UI
│   │   ├── backend_client.py        # Cloud backend HTTP client
│   │   └── models/                  # YuNet face detection model
│   │
│   └── cloud_backend_layer/         # Python cloud backend
│       ├── main.py                  # FastAPI server
│       ├── database/                # SQLite attendance system
│       │   ├── models.py            # SQLAlchemy ORM models
│       │   ├── db_manager.py        # Database operations
│       │   └── attendance_service.py # Business logic
│       ├── models/                  # ML models (liveness, face rec)
│       └── embeddings/              # Stored face embeddings
│
├── README.md
└── LICENSE
```

---

## 🚀 Quick Start

### Prerequisites

- **Hardware**: ESP32-CAM module with OV2640 camera
- **Software**: Python 3.8+, Arduino IDE, PlatformIO (optional)
- **Network**: WiFi network accessible to all components

### 1. Flash ESP32-CAM

```bash
# Open Arduino IDE
# File > Open > src/esp32cam_embedded_layer/esp32cam_ws_stream/esp32cam_ws_stream.ino

# Configure WiFi credentials in the code:
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* websocket_server = "YOUR_GATEWAY_IP";

# Upload to ESP32-CAM (GPIO0 to GND during upload)
```

### 2. Start Cloud Backend

```bash
cd src/cloud_backend_layer

# Install dependencies
pip install -r requirements.txt

# Run the backend server
python main.py
# Server starts at http://localhost:8000
```

### 3. Start Local Gateway

```bash
cd src/local_gateway_layer

# Install dependencies
pip install tornado opencv-python numpy aiohttp

# Run the gateway server
python websockets_stream.py
# Gateway starts at http://localhost:3000
```

### 4. Access the Dashboard

Open your browser and navigate to:

```
http://localhost:3000/view
```

---

## 📝 Register New Faces

```bash
cd src/cloud_backend_layer

# Register a new person
python register_faces.py --name "John Doe" --id "emp001" --image path/to/face.jpg

# Or use the API directly
curl -X POST http://localhost:8000/register \
  -F "person_id=emp001" \
  -F "person_name=John Doe" \
  -F "image=@face.jpg"
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check & system status |
| `/recognize` | POST | Face recognition with attendance logging |
| `/register` | POST | Register new face embedding |
| `/faces` | GET | List registered faces |
| `/attendance/today` | GET | Today's attendance logs |
| `/attendance/daily-report` | GET | Daily attendance summary |
| `/attendance/stats` | GET | Database statistics |

---

## ⚙️ Configuration

### Attendance Settings (in `database/db_manager.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `cooldown_seconds` | 60 | Minimum seconds between scans |
| `work_start_hour` | 6 | Working hours start (6 AM) |
| `work_end_hour` | 22 | Working hours end (10 PM) |
| `lunch_start_hour` | 12 | Lunch period start |
| `lunch_end_hour` | 14 | Lunch period end |
| `min_hours_full_day` | 6.0 | Hours for full day attendance |
| `weekend_allowed` | False | Allow weekend attendance |

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Embedded | ESP32-CAM, Arduino C++ |
| Gateway | Python, Tornado, OpenCV, YuNet |
| Backend | Python, FastAPI, SQLAlchemy, SQLite |
| Face Recognition | InsightFace, ArcFace, ONNX Runtime |
| Liveness Detection | MiniVision Anti-Spoofing Model |
| Frontend | HTML5, CSS3 (Glassmorphism), JavaScript |

---

## 📊 Database Schema

```sql
-- Persons table
CREATE TABLE persons (
    person_id TEXT PRIMARY KEY,
    person_name TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    registered_at TIMESTAMP,
    last_seen TIMESTAMP
);

-- Attendance logs
CREATE TABLE attendance_logs (
    id INTEGER PRIMARY KEY,
    person_id TEXT,
    timestamp TIMESTAMP,
    event_type TEXT,  -- ENTRY, EXIT, DUPLICATE
    status TEXT,      -- SUCCESS, REJECTED, ERROR
    confidence FLOAT,
    is_anomaly BOOLEAN
);

-- Daily attendance summary
CREATE TABLE daily_attendance (
    id INTEGER PRIMARY KEY,
    person_id TEXT,
    attendance_date DATE,
    first_entry TIMESTAMP,
    last_exit TIMESTAMP,
    total_hours FLOAT,
    status TEXT  -- PRESENT, HALF_DAY, ABSENT
);
```

---

## 🔒 Security Features

- **Liveness Detection**: Prevents photo/video spoofing attacks
- **Confidence Thresholds**: Configurable similarity thresholds
- **Anomaly Flagging**: Detects suspicious access patterns
- **Image Logging**: Captures face images for anomalous attempts only
- **Cooldown Period**: Prevents rapid duplicate scans

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| ESP32-CAM not connecting | Check WiFi credentials and gateway IP |
| Face not detected | Ensure good lighting and face within oval guide |
| Low confidence scores | Re-register face with multiple angles |
| Database errors | Delete `database/attendance.db` and restart |
| Backend timeout | Check if cloud backend is running on port 8000 |

---

## 📈 Future Enhancements

- [ ] Multi-camera support
- [ ] Mobile app for attendance viewing
- [ ] Email/SMS notifications
- [ ] Export reports to Excel/PDF
- [ ] Cloud deployment (AWS/GCP)
- [ ] Admin dashboard for management

---

## 👥 Authors

**Swarnodip Nag** ❤️

Department of Computer Applications  
Calcutta University, Kolkata, India

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 Citations & References

If you use this project in your research or work, please cite:

```bibtex
@software{smart_distributed_attendance_2026,
  title = {Smart Distributed Attendance System},
  author = {Swarnodip Nag},
  year = {2026},
  institution = {Calcutta University},
  url = {https://github.com/Swarno-Coder/smart-attendence-system-esp32cam}
}
```

### Acknowledgements

This project uses the following open-source libraries and models:

- **[InsightFace](https://github.com/deepinsight/insightface)** - Face recognition library
- **[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)** - Fast face detection
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern Python web framework
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **[SQLAlchemy](https://www.sqlalchemy.org/)** - Python SQL toolkit
- **[Tornado](https://www.tornadoweb.org/)** - Python web framework for WebSockets

---

<p align="center">
  Made with ❤️ by Swarnodip Nag
</p>
