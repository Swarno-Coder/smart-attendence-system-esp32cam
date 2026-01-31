"""
Test script for Face Recognition Backend API
=============================================
Run this to verify the backend is working correctly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "local_gateway_layer"))

from backend_client import FaceRecognitionClient


async def test_backend():
    """Test the face recognition backend."""
    print("=" * 60)
    print("Face Recognition Backend Test")
    print("=" * 60)
    
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
    client = FaceRecognitionClient(backend_url)
    
    try:
        # Test 1: Health Check
        print("\n[1] Testing health check...")
        health = await client.health_check()
        
        if health.get('status') == 'online':
            print(f"    [OK] Backend is online")
            print(f"    - Liveness model: {'[OK]' if health.get('liveness_model') else '[X]'}")
            print(f"    - Face recognition: {'[OK]' if health.get('face_recognition_model') else '[X]'}")
            print(f"    - Known faces: {health.get('known_faces_count', 0)}")
        else:
            print(f"    [X] Backend is offline: {health.get('error', 'Unknown error')}")
            return False
        
        # Test 2: List registered faces
        print("\n[2] Listing registered faces...")
        faces = await client.list_faces()
        print(f"    Found {faces.get('count', 0)} registered faces")
        for face in faces.get('faces', []):
            print(f"    - {face['person_name']} ({face['person_id']})")
        
        # Test 3: Recognition with test image (if available)
        test_image_path = Path(__file__).parent / "test_images" / "test_face.jpg"
        if test_image_path.exists():
            print(f"\n[3] Testing recognition with {test_image_path.name}...")
            with open(test_image_path, 'rb') as f:
                image_bytes = f.read()
            
            result = await client.recognize_face(image_bytes)
            print(f"    Success: {result.get('success')}")
            print(f"    Message: {result.get('message')}")
            
            if result.get('liveness'):
                liveness = result['liveness']
                print(f"    Liveness: {'Live' if liveness.get('is_live') else 'Spoof'} ({liveness.get('confidence', 0):.2f})")
            
            if result.get('face_match'):
                match = result['face_match']
                print(f"    Match: {match.get('person_name')} ({match.get('similarity', 0):.2f})")
        else:
            print(f"\n[3] Skipping recognition test (no test image at {test_image_path})")
            print("    Create 'test_images/test_face.jpg' to test recognition")
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n[X] Test failed: {e}")
        return False
    finally:
        await client.close()


async def test_register_face(image_path: str, person_id: str, person_name: str):
    """Register a face from an image file."""
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
    client = FaceRecognitionClient(backend_url)
    
    try:
        print(f"Registering face for {person_name}...")
        
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        result = await client.register_face(person_id, person_name, image_bytes)
        
        if result.get('success'):
            print(f"[OK] Successfully registered: {person_name}")
            print(f"  Embedding saved to: {result.get('embedding_path')}")
        else:
            print(f"[X] Registration failed: {result.get('message')}")
            
    finally:
        await client.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Face Recognition Backend")
    parser.add_argument("--register", type=str, help="Register a face from image path")
    parser.add_argument("--id", type=str, help="Person ID for registration")
    parser.add_argument("--name", type=str, help="Person name for registration")
    
    args = parser.parse_args()
    
    if args.register:
        if not args.id or not args.name:
            print("Error: --id and --name required for registration")
            sys.exit(1)
        asyncio.run(test_register_face(args.register, args.id, args.name))
    else:
        success = asyncio.run(test_backend())
        sys.exit(0 if success else 1)
