# Smart Attendance System - Face Recognition Backend

## Overview

This backend implements a two-stage face verification pipeline:

1. **Liveness Detection** - MiniFASNetV2 for anti-spoofing
2. **Face Recognition** - InsightFace (buffalo_l) with 512D embeddings

## Architecture

```
ESP32-CAM → Local Gateway → Backend API
    ↑           (websockets_stream.py)     (FastAPI)
    |                ↓                         ↓
    └────────── Recognition Result ←── Liveness + Recognition
```

## Setup

### 1. Install Dependencies

```bash
cd src/cloud_backend_layer
pip install -r requirements.txt
```

### 2. Download Models

#### MiniFASNetV2 (Liveness Detection)

Download `2.7_80x80_MiniFASNetV2.pth` from [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) and place in:
```
src/cloud_backend_layer/models/2.7_80x80_MiniFASNetV2.pth
```

#### InsightFace (Face Recognition)

The buffalo_l model will be downloaded automatically on first run. Alternatively, download from [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo).

### 3. Start the Backend

```bash
cd src/cloud_backend_layer
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start the Local Gateway

```bash
cd src/local_gateway_layer
python websockets_stream.py
```

## API Endpoints

### `POST /recognize`

Main recognition endpoint. Receives JPEG image, returns:

```json
{
  "success": true,
  "liveness": {
    "is_live": true,
    "confidence": 0.95,
    "message": "Face verified as live"
  },
  "face_detected": true,
  "face_match": {
    "person_id": "emp001",
    "person_name": "John Doe",
    "similarity": 0.85,
    "matched": true
  },
  "processing_time_ms": 150.5,
  "timestamp": "2024-01-29T20:00:00",
  "message": "Face recognized: John Doe"
}
```

### `POST /register`

Register a new face:
- Query params: `person_id`, `person_name`
- Body: multipart form with `image` file

### `GET /faces`

List all registered faces.

### `DELETE /faces/{person_id}`

Remove a registered face.

## Directory Structure

```
cloud_backend_layer/
├── main.py                 # FastAPI app
├── requirements.txt        # Python dependencies
├── liveness/
│   ├── __init__.py
│   └── mini_fasnet.py      # MiniFASNetV2 model
├── models/                 # Model weights (gitignored)
│   └── 2.7_80x80_MiniFASNetV2.pth
└── embeddings/             # Registered face embeddings
    └── *.npz
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_URL` | `http://localhost:8000` | Backend API URL |

Adjust thresholds in `main.py`:

```python
SIMILARITY_THRESHOLD = 0.45  # Face matching threshold
LIVENESS_THRESHOLD = 0.5     # Liveness detection threshold
```

## Pipeline Flow

1. ESP32-CAM captures QVGA stream → Local Gateway
2. Haar cascade detects face → Trigger HQ capture
3. ESP32-CAM sends VGA frame → Local Gateway
4. Gateway sends to Backend `/recognize`:
   - **Step 1**: Detect face using InsightFace
   - **Step 2**: Liveness check using MiniFASNetV2
   - **Step 3**: If live, extract 512D embedding
   - **Step 4**: Find best match via cosine similarity
5. Result sent back to Gateway → ESP32-CAM
