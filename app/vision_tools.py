import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
import os

# Initialize InsightFace (will download models on first run)
# We use a small model for speed in this demo
app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def get_blur_score(image_path: str) -> float:
    """
    Calculates the blur score of an image using the Variance of Laplacian method.
    Higher score means sharper image.
    """
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(score)
    except Exception as e:
        print(f"Error calculating blur: {e}")
        return 0.0

def get_exposure_score(image_path: str) -> float:
    """
    Calculates the average brightness of an image.
    0 is pure black, 255 is pure white.
    """
    try:
        image = cv2.imread(image_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness = hsv[:,:,2].mean()
        return float(brightness)
    except Exception as e:
        print(f"Error calculating exposure: {e}")
        return 0.0

def analyze_faces(image_path: str) -> dict:
    """
    Detects faces and provides clarity/attribute info using InsightFace.
    """
    try:
        image = cv2.imread(image_path)
        faces = app.get(image)
        
        result = {
            "count": len(faces),
            "faces": []
        }
        
        for face in faces:
            # We can use embedding or clarity heuristics here
            # InsightFace provides det_score (detection confidence)
            result["faces"].append({
                "confidence": float(face.det_score),
                "bbox": face.bbox.tolist(),
                "age": int(face.age) if hasattr(face, 'age') else None,
                "gender": int(face.gender) if hasattr(face, 'gender') else None
            })
            
        return result
    except Exception as e:
        print(f"Error analyzing faces: {e}")
        return {"count": 0, "faces": []}

def get_image_hash(image_path: str) -> str:
    """
    Generates a simple dHash for duplicate detection.
    """
    try:
        img = Image.open(image_path).convert('L').resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(img)
        diff = pixels[:, 1:] > pixels[:, :-1]
        return "".join(["1" if b else "0" for b in diff.flatten()])
    except Exception as e:
        print(f"Error generating hash: {e}")
        return ""
