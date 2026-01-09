import os
from dotenv import load_dotenv
import sys
import warnings

# Load environment variables FIRST
load_dotenv()


# Suppress the connectivity check message by redirecting stderr temporarily
import io
_original_stderr = sys.stderr

from paddleocr import PaddleOCR
import numpy as np
from io import BytesIO
from PIL import Image
import re
import cv2
import logging

# Restore stderr
sys.stderr = _original_stderr

# Suppress PaddleOCR verbose logging
logging.getLogger('ppocr').setLevel(logging.ERROR)
logging.getLogger('paddlex').setLevel(logging.ERROR)


class PlateReader:
    def __init__(self):
        print("Loading PaddleOCR...")
        # Temporarily suppress output during initialization
        _stderr = sys.stderr
        sys.stderr = io.StringIO()
        
        try:
            self.reader = PaddleOCR(lang='en' , use_angle_cls=False)
        finally:
            sys.stderr = _stderr
        
        print("PaddleOCR ready!")
        self.min_confidence = float(os.getenv("DETECTION_CONFIDENCE", "0.5"))
        self.debug = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    def read_from_cropped(self, cropped_image_np):
        """Read text from a cropped license plate image (numpy array)"""
        try:
            # Use original image directly - PaddleOCR handles preprocessing
            results = self.reader.ocr(cropped_image_np)
            
            if not results or not results[0]:
                return None
            
            result_dict = results[0]
            
            if 'rec_texts' not in result_dict or not result_dict['rec_texts']:
                return None
            
            texts = result_dict['rec_texts']
            scores = result_dict.get('rec_scores', [1.0] * len(texts))
            
            # Collect valid results (at least 3 alphanumeric characters)
            all_texts = []
            for text, confidence in zip(texts, scores):
                cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(cleaned) >= 3:
                    all_texts.append({
                        "text": cleaned,
                        "raw": text,
                        "conf": confidence
                    })
            
            if not all_texts:
                return None
            
            # Get best result by confidence
            best = max(all_texts, key=lambda x: x["conf"])
            
            # If multiple results, combine them
            if len(all_texts) > 1:
                combined_text = ''.join([t["text"] for t in all_texts])
                combined_raw = ' '.join([t["raw"] for t in all_texts])
                avg_conf = sum([t["conf"] for t in all_texts]) / len(all_texts)
                
                if len(combined_text) > len(best["text"]) and avg_conf > 0.3:
                    best = {
                        "text": combined_text,
                        "raw": combined_raw,
                        "conf": avg_conf
                    }


            return {
                "text": best["text"],
                "raw_text": best["raw"],
                "confidence": round(best["conf"], 2)
            }
        
        except Exception as e:
            if self.debug:
                print(f"OCR Error: {e}")
            return None
