"""
Backend Client for Face Recognition API
========================================
Client module to connect local gateway to cloud backend.
"""

import aiohttp
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class FaceRecognitionClient:
    """
    Async client for communicating with Face Recognition Backend.
    """
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    async def close(self):
        """Close the client session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check backend health status."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.backend_url}/") as response:
                if response.status == 200:
                    return await response.json()
                return {"status": "error", "code": response.status}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "offline", "error": str(e)}
    
    async def recognize_face(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Send image to backend for face recognition.
        
        Args:
            image_bytes: JPEG encoded image bytes
            
        Returns:
            Recognition result from backend
        """
        async with self._lock:  # Prevent concurrent requests
            try:
                session = await self._get_session()
                
                # Prepare multipart form data
                data = aiohttp.FormData()
                data.add_field(
                    'image',
                    image_bytes,
                    filename='capture.jpg',
                    content_type='image/jpeg'
                )
                
                async with session.post(
                    f"{self.backend_url}/recognize",
                    data=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Recognition result: {result.get('message', 'Unknown')}")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Recognition failed: {response.status} - {error_text}")
                        return {
                            "success": False,
                            "message": f"Backend error: {response.status}",
                            "error": error_text
                        }
                        
            except asyncio.TimeoutError:
                logger.error("Recognition request timed out")
                return {
                    "success": False,
                    "message": "Request timed out",
                    "error": "Timeout"
                }
            except aiohttp.ClientError as e:
                logger.error(f"Connection error: {e}")
                return {
                    "success": False,
                    "message": "Connection failed",
                    "error": str(e)
                }
            except Exception as e:
                logger.error(f"Recognition error: {e}")
                return {
                    "success": False,
                    "message": "Unknown error",
                    "error": str(e)
                }
    
    async def register_face(self, person_id: str, person_name: str, image_bytes: bytes) -> Dict[str, Any]:
        """
        Register a new face for recognition.
        
        Args:
            person_id: Unique identifier
            person_name: Display name
            image_bytes: JPEG encoded image bytes
            
        Returns:
            Registration result
        """
        try:
            session = await self._get_session()
            
            data = aiohttp.FormData()
            data.add_field('image', image_bytes, filename='register.jpg', content_type='image/jpeg')
            
            async with session.post(
                f"{self.backend_url}/register",
                data=data,
                params={"person_id": person_id, "person_name": person_name}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {"success": False, "message": error_text}
                    
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return {"success": False, "message": str(e)}
    
    async def list_faces(self) -> Dict[str, Any]:
        """Get list of registered faces."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.backend_url}/faces") as response:
                if response.status == 200:
                    return await response.json()
                return {"faces": [], "count": 0}
        except Exception as e:
            logger.error(f"List faces error: {e}")
            return {"faces": [], "count": 0, "error": str(e)}


# Global client instance
_client: Optional[FaceRecognitionClient] = None


def get_client(backend_url: str = "http://localhost:8000") -> FaceRecognitionClient:
    """Get or create global client instance."""
    global _client
    if _client is None:
        _client = FaceRecognitionClient(backend_url)
    return _client


async def close_client():
    """Close global client instance."""
    global _client
    if _client:
        await _client.close()
        _client = None
