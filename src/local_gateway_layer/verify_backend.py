
import asyncio
import os
import sys
from pathlib import Path

# Add current directory to path to import backend_client
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

from backend_client import FaceRecognitionClient

TEST_IMAGES_DIR = current_dir.parent / "cloud_backend_layer" / "TEST"

async def main():
    print(f"Testing images from: {TEST_IMAGES_DIR}")
    
    if not TEST_IMAGES_DIR.exists():
        print(f"Error: Directory not found: {TEST_IMAGES_DIR}")
        return

    client = FaceRecognitionClient()
    
    # Health check
    print("\nChecking backend health...")
    health = await client.health_check()
    print(f"Backend Status: {health}")
    
    if health.get('status') != 'online':
        print("Backend is offline. Please start main.py first.")
        await client.close()
        return

    # Process images
    files = list(TEST_IMAGES_DIR.glob("*.jpg"))
    print(f"\nFound {len(files)} test images.")
    
    for img_path in files:
        print(f"\nProcessing: {img_path.name}")
        try:
            with open(img_path, "rb") as f:
                image_bytes = f.read()
            
            result = await client.recognize_face(image_bytes)
            
            # Formatted Output
            liveness = result.get('liveness', {})
            recognition = result.get('recognition', {})
            
            print(f"  > Liveness: {'PASS' if liveness.get('is_live') else 'FAIL'}")
            print(f"    Confidence: {liveness.get('confidence', 0):.2f}")
            if not liveness.get('is_live'):
                 print(f"    Reason: {liveness.get('message')}")
                 
            success = result.get('success')
            if success:
                print(f"  > Recognition: SUCCESS")
                match = result.get('face_match', {})
                print(f"    Person: {match.get('person_name')} ({match.get('person_id')})")
                print(f"    Similarity: {match.get('similarity', 0):.2f}")
            else:
                 print(f"  > Recognition: FAIL")
                 print(f"    Message: {result.get('message')}")
                 if result.get('face_match'):
                      match = result.get('face_match')
                      print(f"    Best Match (Low Conf): {match.get('name')} ({match.get('similarity', 0):.2f})")
                      
        except Exception as e:
            print(f"  > Error: {e}")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
