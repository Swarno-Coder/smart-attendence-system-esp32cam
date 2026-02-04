"""
Face Recognition Backend with Liveness Detection
=================================================
Flow:
1. Receive HQ image from local gateway
2. Liveness check using MiniFASNetV2
3. If genuine, recognize face using InsightFace (buffalo_l)
4. Find cosine similarity with known face embeddings (512D)
5. Return response to local gateway
"""

import os
import io
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import attendance database module
# Add current directory to path to ensure database module is found
import sys
_current_dir = Path(__file__).parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

try:
    from database import get_db_manager, AttendanceService
    DATABASE_AVAILABLE = True
except ImportError as e:
    DATABASE_AVAILABLE = False
    print(f"Warning: Database module not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============== Configuration ==============
MODELS_DIR = Path(__file__).parent / "models"
EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"
LIVENESS_MODEL_PATH = MODELS_DIR / "2.7_80x80_MiniFASNetV2.pth"
INSIGHTFACE_MODEL_NAME = "buffalo_l"
SIMILARITY_THRESHOLD = 0.45  # Cosine similarity threshold for face matching
LIVENESS_THRESHOLD = 0.5    # Threshold for liveness detection
# ==========================================


def get_onnx_providers() -> list:
    """
    Get available ONNX Runtime execution providers.
    Only includes CUDAExecutionProvider if CUDA is actually available and working.
    This prevents error messages when CUDA DLLs are not installed.
    """
    providers = []
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    
    if cuda_available:
        try:
            # Test if ONNX Runtime can actually use CUDA
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                # Double-check by trying to create a session option with CUDA
                try:
                    sess_options = ort.SessionOptions()
                    # If we get here, CUDA provider is available
                    providers.append('CUDAExecutionProvider')
                    logger.info("CUDA is available - using GPU acceleration")
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"CUDA availability check failed: {e}")
    
    # Always include CPU as fallback
    providers.append('CPUExecutionProvider')
    
    if len(providers) == 1:
        logger.info("Using CPU-only execution (CUDA not available)")
    
    return providers

app = FastAPI(
    title="Smart Attendance Face Recognition API",
    description="Face recognition backend with liveness detection using MiniFASNetV2 and InsightFace",
    version="1.0.0"
)

# CORS middleware for local gateway communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Response Models ==============
class LivenessResult(BaseModel):
    is_live: bool
    confidence: float
    message: str


class FaceMatch(BaseModel):
    person_id: str
    person_name: str
    similarity: float
    matched: bool


class AttendanceInfo(BaseModel):
    """Attendance logging result."""
    event_type: str = Field(..., description="ENTRY, EXIT, DUPLICATE, or INVALID")
    status: str = Field(..., description="SUCCESS, REJECTED, ALREADY_LOGGED, or ERROR")
    message: str = Field(..., description="Human-readable status message")
    daily_summary: Optional[Dict[str, Any]] = Field(None, description="Today's attendance summary")
    cooldown_remaining_seconds: Optional[int] = Field(None, description="Seconds until next scan allowed")
    is_anomaly: bool = Field(False, description="Whether this was flagged as anomalous")
    anomaly_reason: Optional[str] = Field(None, description="Reason for anomaly flag")


class RecognitionResponse(BaseModel):
    success: bool
    liveness: LivenessResult
    face_detected: bool
    face_match: Optional[FaceMatch] = None
    attendance: Optional[AttendanceInfo] = None
    processing_time_ms: float
    timestamp: str
    message: str


# ============== Global Model Instances ==============
liveness_detector = None
face_analyzer = None
known_embeddings: Dict[str, Dict[str, Any]] = {}
attendance_service: Optional['AttendanceService'] = None  # SQLite attendance tracking


def load_liveness_model():
    """
    Load MiniFASNetV2 for liveness detection.
    The model uses Silent Face Anti-Spoofing approach.
    """
    global liveness_detector
    
    try:
        import torch
        from liveness.mini_fasnet import LivenessPredictor
        
        if not LIVENESS_MODEL_PATH.exists():
            logger.warning(f"Liveness model not found at {LIVENESS_MODEL_PATH}")
            logger.info("Please download 2.7_80x80_MiniFASNetV2.pth and place it in the models directory")
            return False
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading liveness model on {device}")
        
        liveness_detector = LivenessPredictor(
            model_path=str(LIVENESS_MODEL_PATH),
            device=device
        )
        
        logger.info("Liveness model loaded successfully")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import liveness modules: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to load liveness model: {e}")
        return False


def load_face_recognition_model():
    """
    Load InsightFace model (buffalo_l) for face recognition.
    Buffalo_l produces 512-dimensional face embeddings.
    """
    global face_analyzer
    
    try:
        from insightface.app import FaceAnalysis
        
        # Get available execution providers (CUDA if available, otherwise CPU only)
        providers = get_onnx_providers()
        
        logger.info(f"Loading InsightFace model: {INSIGHTFACE_MODEL_NAME}")
        logger.info(f"Using execution providers: {providers}")
        
        face_analyzer = FaceAnalysis(
            name=INSIGHTFACE_MODEL_NAME,
            root=str(MODELS_DIR),
            providers=providers
        )
        
        # ctx_id: -1 for CPU, 0 for GPU
        ctx_id = 0 if 'CUDAExecutionProvider' in providers else -1
        # Lower threshold for better recall, smaller size for speed (VGA is 640x480)
        face_analyzer.prepare(ctx_id=ctx_id, det_size=(320, 320), det_thresh=0.3)
        
        logger.info("InsightFace model loaded successfully")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import insightface: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to load InsightFace model: {e}")
        return False


def load_known_embeddings():
    """
    Load known face embeddings from disk.
    Each person's embedding is stored as a .npz file containing:
    - embedding: 512D numpy array
    - person_id: unique identifier
    - person_name: display name
    """
    global known_embeddings
    
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    embedding_files = list(EMBEDDINGS_DIR.glob("*.npz"))
    logger.info(f"Found {len(embedding_files)} known face embeddings")
    
    for emb_file in embedding_files:
        try:
            data = np.load(emb_file, allow_pickle=True)
            
            # NPZ files need to access arrays by key, not .get()
            person_id = str(data['person_id']) if 'person_id' in data.files else emb_file.stem
            person_name = str(data['person_name']) if 'person_name' in data.files else emb_file.stem
            embedding = data['embedding']
            
            # Handle scalar arrays (when saved as np.savez with strings)
            if isinstance(person_id, np.ndarray):
                person_id = str(person_id.item()) if person_id.ndim == 0 else str(person_id)
            if isinstance(person_name, np.ndarray):
                person_name = str(person_name.item()) if person_name.ndim == 0 else str(person_name)
            
            if embedding.shape != (512,):
                logger.warning(f"Invalid embedding shape for {person_id}: {embedding.shape}")
                continue
            
            known_embeddings[person_id] = {
                'name': person_name,
                'embedding': embedding / np.linalg.norm(embedding)  # Normalize
            }
            logger.info(f"Loaded embedding for: {person_name} ({person_id})")
            
        except Exception as e:
            logger.error(f"Failed to load embedding {emb_file}: {e}")
    
    return len(known_embeddings)


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two normalized embeddings."""
    # Embeddings should already be normalized
    return float(np.dot(embedding1, embedding2))


def find_best_match(query_embedding: np.ndarray) -> Optional[FaceMatch]:
    """
    Find the best matching face from known embeddings.
    Returns None if no match exceeds the similarity threshold.
    """
    if not known_embeddings:
        return None
    
    # Normalize query embedding
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    
    best_match = None
    best_similarity = -1.0
    
    for person_id, data in known_embeddings.items():
        similarity = cosine_similarity(query_norm, data['embedding'])
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = {
                'person_id': person_id,
                'person_name': data['name'],
                'similarity': similarity
            }
    
    if best_match and best_similarity >= SIMILARITY_THRESHOLD:
        return FaceMatch(
            person_id=best_match['person_id'],
            person_name=best_match['person_name'],
            similarity=best_similarity,
            matched=True
        )
    elif best_match:
        return FaceMatch(
            person_id=best_match['person_id'],
            person_name=best_match['person_name'],
            similarity=best_similarity,
            matched=False
        )
    
    return None


def check_liveness(image: np.ndarray, face_bbox: tuple) -> LivenessResult:
    """
    Perform liveness detection on the detected face region.
    Uses MiniFASNetV2 to detect spoofing attacks (photos, screens, masks).
    """
    global liveness_detector
    
    if liveness_detector is None:
        return LivenessResult(
            is_live=True,  # Assume live if model not available
            confidence=0.0,
            message="Liveness model not loaded - assuming live"
        )
    
    try:
        # Extract face region with some margin
        x1, y1, x2, y2 = face_bbox
        h, w = image.shape[:2]
        
        # Add margin around face
        margin = int(max(x2 - x1, y2 - y1) * 0.3)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        face_region = image[y1:y2, x1:x2]
        
        # Predict liveness
        is_live, confidence = liveness_detector.predict(face_region)
        
        if is_live:
            return LivenessResult(
                is_live=True,
                confidence=confidence,
                message="Face verified as live"
            )
        else:
            return LivenessResult(
                is_live=False,
                confidence=confidence,
                message="Spoofing attack detected - face is not live"
            )
            
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return LivenessResult(
            is_live=False,
            confidence=0.0,
            message=f"Liveness check error: {str(e)}"
        )


def detect_and_recognize_face(image: np.ndarray) -> tuple:
    """
    Detect face and extract 512D embedding using InsightFace.
    Returns (face_detected, face_bbox, embedding, error_message)
    """
    global face_analyzer
    
    if face_analyzer is None:
        return False, None, None, "Face analyzer not loaded"
    
    try:
        # Detect faces
        faces = face_analyzer.get(image)
        
        if not faces:
            return False, None, None, "No face detected"
        
        # Get the largest face (most likely the main subject)
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        # Extract bounding box
        bbox = largest_face.bbox.astype(int)
        face_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
        
        # Get 512D embedding
        embedding = largest_face.embedding
        
        if embedding is None or len(embedding) != 512:
            return False, face_bbox, None, "Failed to extract face embedding"
        
        return True, face_bbox, embedding, None
        
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return False, None, None, str(e)


@app.on_event("startup")
async def startup_event():
    """Initialize models and database on startup."""
    global attendance_service
    
    logger.info("=" * 60)
    logger.info("Starting Face Recognition Backend")
    logger.info("=" * 60)
    
    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load models
    liveness_loaded = load_liveness_model()
    face_rec_loaded = load_face_recognition_model()
    num_embeddings = load_known_embeddings()
    
    # Initialize attendance database
    db_initialized = False
    if DATABASE_AVAILABLE:
        try:
            db_manager = get_db_manager()
            attendance_service = AttendanceService(db_manager)
            
            # Sync persons from existing embeddings
            synced = db_manager.sync_persons_from_embeddings(EMBEDDINGS_DIR)
            
            db_stats = db_manager.get_stats()
            db_initialized = True
            logger.info(f"Database: ✓ Initialized ({db_stats['total_persons']} persons, {db_stats['total_logs']} logs)")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            attendance_service = None
    else:
        logger.warning("Database: ✗ Module not available - attendance tracking disabled")
    
    logger.info("-" * 60)
    logger.info(f"Liveness Model: {'✓ Loaded' if liveness_loaded else '✗ Not Available'}")
    logger.info(f"Face Recognition: {'✓ Loaded' if face_rec_loaded else '✗ Not Available'}")
    logger.info(f"Known Faces: {num_embeddings}")
    logger.info(f"Attendance DB: {'✓ Ready' if db_initialized else '✗ Disabled'}")
    logger.info("=" * 60)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Smart Attendance Face Recognition API",
        "liveness_model": liveness_detector is not None,
        "face_recognition_model": face_analyzer is not None,
        "known_faces_count": len(known_embeddings),
        "attendance_database": attendance_service is not None
    }


@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_face(image: UploadFile = File(...)):
    """
    Main face recognition endpoint.
    
    Flow:
    1. Receive image from ESP32-CAM via local gateway
    2. Detect face using InsightFace
    3. Perform liveness check using MiniFASNetV2
    4. If live, extract 512D embedding and find match
    5. Return recognition result
    """
    start_time = datetime.now()
    
    try:
        # Read and decode image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        logger.info(f"Received image: {img.shape[1]}x{img.shape[0]}")
        
        # Step 1: Detect face and get embedding
        face_detected, face_bbox, embedding, error = detect_and_recognize_face(img)
        
        if not face_detected:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return RecognitionResponse(
                success=False,
                liveness=LivenessResult(is_live=False, confidence=0.0, message="N/A"),
                face_detected=False,
                face_match=None,
                processing_time_ms=processing_time,
                timestamp=datetime.now().isoformat(),
                message=error or "No face detected in image"
            )
        
        logger.info(f"Face detected at: {face_bbox}")
        
        # Step 2: Liveness check
        liveness_result = check_liveness(img, face_bbox)
        
        if not liveness_result.is_live:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return RecognitionResponse(
                success=False,
                liveness=liveness_result,
                face_detected=True,
                face_match=None,
                processing_time_ms=processing_time,
                timestamp=datetime.now().isoformat(),
                message="Liveness check failed - spoofing detected"
            )
        
        logger.info(f"Liveness verified: {liveness_result.confidence:.2f}")
        
        # Step 3: Find matching face
        face_match = find_best_match(embedding)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        if face_match and face_match.matched:
            logger.info(f"Face matched: {face_match.person_name} ({face_match.similarity:.2f})")
            
            # Step 4: Log attendance if database available
            attendance_info = None
            if attendance_service:
                try:
                    # Encode image for potential storage (failed/anomaly cases)
                    _, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    face_image_bytes = img_encoded.tobytes()
                    
                    attendance_result = attendance_service.log_attendance(
                        person_id=face_match.person_id,
                        person_name=face_match.person_name,
                        confidence=face_match.similarity,
                        liveness_score=liveness_result.confidence,
                        device_id="esp32cam",
                        face_image=face_image_bytes
                    )
                    
                    attendance_info = AttendanceInfo(
                        event_type=attendance_result.event_type,
                        status=attendance_result.status,
                        message=attendance_result.message,
                        daily_summary=attendance_result.daily_summary,
                        cooldown_remaining_seconds=attendance_result.cooldown_remaining,
                        is_anomaly=attendance_result.is_anomaly,
                        anomaly_reason=attendance_result.anomaly_reason
                    )
                    logger.info(f"Attendance logged: {attendance_result.event_type} - {attendance_result.message}")
                except Exception as e:
                    logger.error(f"Attendance logging failed: {e}")
                    attendance_info = AttendanceInfo(
                        event_type="ERROR",
                        status="ERROR",
                        message=f"Attendance logging error: {str(e)}"
                    )
            
            return RecognitionResponse(
                success=True,
                liveness=liveness_result,
                face_detected=True,
                face_match=face_match,
                attendance=attendance_info,
                processing_time_ms=processing_time,
                timestamp=datetime.now().isoformat(),
                message=f"Face recognized: {face_match.person_name}"
            )
        else:
            # Log failed recognition attempt
            if attendance_service:
                try:
                    _, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    attendance_service.log_failed_recognition(
                        reason="Face not recognized - no matching person found",
                        confidence=face_match.similarity if face_match else None,
                        liveness_score=liveness_result.confidence,
                        device_id="esp32cam",
                        face_image=img_encoded.tobytes()
                    )
                except Exception as e:
                    logger.error(f"Failed to log unrecognized attempt: {e}")
            
            return RecognitionResponse(
                success=False,
                liveness=liveness_result,
                face_detected=True,
                face_match=face_match,  # May contain best match below threshold
                processing_time_ms=processing_time,
                timestamp=datetime.now().isoformat(),
                message="Face not recognized - no matching person found"
            )
        
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return RecognitionResponse(
            success=False,
            liveness=LivenessResult(is_live=False, confidence=0.0, message="Error"),
            face_detected=False,
            face_match=None,
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat(),
            message=f"Error processing image: {str(e)}"
        )


@app.post("/register")
async def register_face(
    person_id: str = Query(..., description="Unique identifier for the person"),
    person_name: str = Query(..., description="Display name for the person"),
    image: UploadFile = File(...)
):
    """
    Register a new face for recognition.
    Extracts 512D embedding and saves to disk.
    """
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Detect face and get embedding
        face_detected, face_bbox, embedding, error = detect_and_recognize_face(img)
        
        if not face_detected:
            raise HTTPException(status_code=400, detail=error or "No face detected")
        
        # Normalize and save embedding
        embedding_norm = embedding / np.linalg.norm(embedding)
        
        embedding_path = EMBEDDINGS_DIR / f"{person_id}.npz"
        np.savez(
            embedding_path,
            embedding=embedding_norm,
            person_id=person_id,
            person_name=person_name
        )
        
        # Update in-memory cache
        known_embeddings[person_id] = {
            'name': person_name,
            'embedding': embedding_norm
        }
        
        # Add person to attendance database
        if attendance_service:
            try:
                attendance_service.ensure_person_exists(person_id, person_name)
            except Exception as e:
                logger.warning(f"Failed to add person to attendance DB: {e}")
        
        logger.info(f"Registered face for: {person_name} ({person_id})")
        
        return {
            "success": True,
            "message": f"Face registered for {person_name}",
            "person_id": person_id,
            "embedding_path": str(embedding_path)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/faces")
async def list_registered_faces():
    """List all registered faces."""
    faces = []
    for person_id, data in known_embeddings.items():
        faces.append({
            "person_id": person_id,
            "person_name": data['name']
        })
    return {"faces": faces, "count": len(faces)}


@app.delete("/faces/{person_id}")
async def delete_face(person_id: str):
    """Delete a registered face."""
    if person_id not in known_embeddings:
        raise HTTPException(status_code=404, detail="Person not found")
    
    # Remove from disk
    embedding_path = EMBEDDINGS_DIR / f"{person_id}.npz"
    if embedding_path.exists():
        embedding_path.unlink()
    
    # Remove from memory
    del known_embeddings[person_id]
    
    logger.info(f"Deleted face: {person_id}")
    return {"success": True, "message": f"Face deleted: {person_id}"}


# ============== Attendance Endpoints ==============

@app.get("/attendance/today")
async def get_today_attendance(person_id: Optional[str] = Query(None, description="Filter by person ID")):
    """Get today's attendance logs."""
    if not attendance_service:
        raise HTTPException(status_code=503, detail="Attendance database not available")
    
    try:
        logs = attendance_service.get_today_logs(person_id=person_id)
        return {
            "success": True,
            "date": datetime.now().date().isoformat(),
            "logs": logs,
            "count": len(logs)
        }
    except Exception as e:
        logger.error(f"Error getting today's attendance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/attendance/daily-report")
async def get_daily_report(date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format")):
    """Get daily attendance report with summary."""
    if not attendance_service:
        raise HTTPException(status_code=503, detail="Attendance database not available")
    
    try:
        from datetime import date as date_type
        report_date = None
        if date:
            report_date = date_type.fromisoformat(date)
        
        report = attendance_service.get_daily_report(report_date)
        return {"success": True, "report": report}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Error getting daily report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/attendance/person/{person_id}")
async def get_person_attendance(person_id: str):
    """Get today's attendance summary for a specific person."""
    if not attendance_service:
        raise HTTPException(status_code=503, detail="Attendance database not available")
    
    try:
        summary = attendance_service.get_person_today_summary(person_id)
        if not summary:
            return {
                "success": True,
                "person_id": person_id,
                "message": "No attendance records for today",
                "summary": None
            }
        return {"success": True, "person_id": person_id, "summary": summary}
    except Exception as e:
        logger.error(f"Error getting person attendance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/attendance/stats")
async def get_attendance_stats():
    """Get attendance database statistics."""
    if not attendance_service:
        return {
            "success": False,
            "database_available": False,
            "message": "Attendance database not available"
        }
    
    try:
        stats = attendance_service.db.get_stats()
        return {"success": True, "database_available": True, "stats": stats}
    except Exception as e:
        logger.error(f"Error getting attendance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
