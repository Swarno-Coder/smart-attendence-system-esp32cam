"""
Face Registration Utility
=========================
Register faces from a directory of images.
Each image should be named as: person_id_name.jpg
Example: emp001_John_Doe.jpg
"""

import asyncio
import sys
import os
from pathlib import Path
import cv2
import numpy as np
import argparse


async def register_face_via_api(backend_url: str, image_bytes: bytes, person_id: str, person_name: str):
    """Register a face via the backend API."""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('image', image_bytes, filename='face.jpg', content_type='image/jpeg')
        
        async with session.post(
            f"{backend_url}/register",
            data=data,
            params={"person_id": person_id, "person_name": person_name}
        ) as response:
            return await response.json()


async def register_from_directory(images_dir: str, backend_url: str = "http://localhost:8000"):
    """
    Register all faces from a directory.
    
    Expected filename format: person_id_firstname_lastname.jpg
    Example: emp001_John_Doe.jpg -> id=emp001, name="John Doe"
    """
    images_path = Path(images_dir)
    
    if not images_path.exists():
        print(f"Error: Directory not found: {images_dir}")
        return
    
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png")) + list(images_path.glob("*.jpeg"))
    print(f"Found {len(image_files)} images in {images_dir}")
    
    success_count = 0
    fail_count = 0
    
    for img_path in image_files:
        try:
            # Parse filename: person_id_firstname_lastname.jpg
            stem = img_path.stem
            parts = stem.split("_")
            
            if len(parts) < 2:
                print(f"Skipping {img_path.name} - invalid filename format")
                continue
            
            person_id = parts[0]
            person_name = " ".join(parts[1:]).title()
            
            # Read image
            with open(img_path, 'rb') as f:
                image_bytes = f.read()
            
            print(f"Registering: {person_name} ({person_id})...", end=" ")
            
            result = await register_face_via_api(backend_url, image_bytes, person_id, person_name)
            
            if result.get('success'):
                print("✓")
                success_count += 1
            else:
                print(f"✗ {result.get('message', 'Unknown error')}")
                fail_count += 1
                
        except Exception as e:
            print(f"✗ Error: {e}")
            fail_count += 1
    
    print(f"\n{'=' * 40}")
    print(f"Registration complete:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")


async def register_single_webcam(backend_url: str, person_id: str, person_name: str):
    """
    Capture a face from webcam and register it.
    """
    print(f"Opening webcam to capture face for {person_name}...")
    print("Press SPACE to capture, ESC to cancel")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    captured = False
    frame_to_register = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display frame
            display = frame.copy()
            cv2.putText(display, f"Registering: {person_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "SPACE: Capture | ESC: Cancel", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Face Registration", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                print("Cancelled")
                break
            elif key == 32:  # SPACE
                frame_to_register = frame
                captured = True
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    if captured and frame_to_register is not None:
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', frame_to_register)
        image_bytes = buffer.tobytes()
        
        print("Registering face...")
        result = await register_face_via_api(backend_url, image_bytes, person_id, person_name)
        
        if result.get('success'):
            print(f"✓ Successfully registered: {person_name}")
        else:
            print(f"✗ Registration failed: {result.get('message')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Registration Utility")
    parser.add_argument("--dir", type=str, help="Directory containing face images")
    parser.add_argument("--webcam", action="store_true", help="Capture from webcam")
    parser.add_argument("--id", type=str, help="Person ID")
    parser.add_argument("--name", type=str, help="Person name")
    parser.add_argument("--backend", type=str, default="http://localhost:8000", help="Backend URL")
    
    args = parser.parse_args()
    
    if args.dir:
        asyncio.run(register_from_directory(args.dir, args.backend))
    elif args.webcam:
        if not args.id or not args.name:
            print("Error: --id and --name required for webcam capture")
            sys.exit(1)
        asyncio.run(register_single_webcam(args.backend, args.id, args.name))
    else:
        print("Usage:")
        print("  Register from directory: python register_faces.py --dir ./faces/")
        print("  Register from webcam:    python register_faces.py --webcam --id emp001 --name 'John Doe'")
