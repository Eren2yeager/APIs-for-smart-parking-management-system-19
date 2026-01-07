from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv
import requests
from pathlib import Path

load_dotenv()


class LicensePlateDetector:
    def __init__(self):
        # Auto-download license plate detection model if not exists
        model_path = os.getenv("LICENSE_PLATE_MODEL", "license_plate_detector.pt")
        
        # Check if model exists, if not download from Hugging Face
        if not os.path.exists(model_path):
            print(f"License plate model not found at {model_path}")
            print("Downloading pre-trained model from Hugging Face...")
            self._download_model(model_path)
        
        # Load the model (downloads automatically on first run)
        self.model = YOLO(model_path)
        self.confidence_threshold = float(os.getenv("PLATE_DETECTION_CONFIDENCE", "0.3"))
    
    def _download_model(self, save_path):
        """Download pre-trained license plate detection model from Hugging Face"""
        try:
            # Hugging Face model URL
            model_url = "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt"
            
            print(f"Downloading from: {model_url}")
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            # Save the model
            total_size = int(response.headers.get('content-length', 0))
            with open(save_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        downloaded += len(chunk)
                        f.write(chunk)
                        # Show progress
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownload progress: {progress:.1f}%", end='')
            
            print(f"\n✓ Model downloaded successfully to {save_path}")
        
        except Exception as e:
            print(f"\n✗ Failed to download model: {str(e)}")
            print("Please download manually from: https://huggingface.co/Koushim/yolov8-license-plate-detection")
            raise
    
    def detect_plates(self, image_bytes):
        """Detect license plates in image and return bounding boxes"""
        try:
            # Convert bytes to image
            image = Image.open(BytesIO(image_bytes))
            image_np = np.array(image)
            
            # Run detection
            results = self.model(image_np)
            
            plates = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = float(box.conf[0])
                    
                    # Filter by confidence threshold
                    if confidence < self.confidence_threshold:
                        continue
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    plates.append({
                        "confidence": round(confidence, 2),
                        "bbox": {
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2)
                        }
                    })
            
            return {
                "success": True,
                "plate_count": len(plates),
                "plates": plates,
                "image_shape": image_np.shape
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def crop_plates(self, image_bytes, bboxes):
        """Crop license plate regions from image"""
        try:
            image = Image.open(BytesIO(image_bytes))
            image_np = np.array(image)
            
            cropped_plates = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                
                # Add more padding (10% on each side for better OCR)
                height, width = image_np.shape[:2]
                padding_x = int((x2 - x1) * 0.1)
                padding_y = int((y2 - y1) * 0.1)
                
                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(width, x2 + padding_x)
                y2 = min(height, y2 + padding_y)
                
                # Crop the region
                cropped = image_np[y1:y2, x1:x2]
                
                # Ensure minimum size for OCR
                crop_height, crop_width = cropped.shape[:2]
                if crop_height < 30 or crop_width < 80:
                    # Resize to minimum size
                    scale = max(30 / crop_height, 80 / crop_width)
                    new_width = int(crop_width * scale)
                    new_height = int(crop_height * scale)
                    cropped = cv2.resize(cropped, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
                cropped_plates.append(cropped)
            
            return cropped_plates
        
        except Exception as e:
            print(f"Cropping error: {str(e)}")
            return []
