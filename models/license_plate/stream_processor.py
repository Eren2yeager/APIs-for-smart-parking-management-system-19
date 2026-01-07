"""
Real-time license plate recognition stream processor
Handles frame skipping, deduplication, and session state management
"""

import time
import os
from dotenv import load_dotenv
from .pipeline import PlateRecognitionPipeline

load_dotenv()


class PlateStreamProcessor:
    """
    Processes video stream frames for license plate recognition
    Features:
    - Frame skipping for performance
    - Deduplication (ignore same plate for N seconds)
    - Session state management
    """
    
    def __init__(self, skip_frames=None, dedup_window=None):
        """
        Initialize stream processor
        
        Args:
            skip_frames: Process every Nth frame (default from env or 5)
            dedup_window: Ignore duplicate plates for N seconds (default from env or 10)
        """
        self.pipeline = PlateRecognitionPipeline()
        
        # Configuration
        self.skip_frames = skip_frames or int(os.getenv("GATE_FRAME_SKIP", "5"))
        self.dedup_window = dedup_window or int(os.getenv("GATE_DEDUP_WINDOW", "10"))
        
        # State
        self.frame_count = 0
        self.processed_count = 0
        self.seen_plates = {}  # {plate_number: last_seen_timestamp}
        
        # Removed verbose initialization log
    
    def should_process_frame(self):
        """Determine if current frame should be processed"""
        return self.frame_count % self.skip_frames == 0
    
    def is_duplicate(self, plate_number):
        """
        Check if plate was recently seen
        
        Args:
            plate_number: License plate text
            
        Returns:
            bool: True if duplicate (seen within dedup_window)
        """
        current_time = time.time()
        
        if plate_number in self.seen_plates:
            last_seen = self.seen_plates[plate_number]
            time_diff = current_time - last_seen
            
            if time_diff < self.dedup_window:
                return True
        
        # Update last seen time
        self.seen_plates[plate_number] = current_time
        return False
    
    def cleanup_old_plates(self):
        """Remove plates older than dedup_window from memory"""
        current_time = time.time()
        plates_to_remove = []
        
        for plate_number, last_seen in self.seen_plates.items():
            if current_time - last_seen > self.dedup_window * 2:
                plates_to_remove.append(plate_number)
        
        for plate_number in plates_to_remove:
            del self.seen_plates[plate_number]
    
    def process_frame(self, frame_bytes):
        """
        Process a single frame from video stream
        
        Args:
            frame_bytes: Image data as bytes
            
        Returns:
            dict: Processing result with plates and metadata, or None if frame skipped
        """
        self.frame_count += 1
        start_time = time.time()
        
        # Frame skipping
        if not self.should_process_frame():
            return None
        
        # Process with pipeline
        result = self.pipeline.process(frame_bytes)
        
        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "frame_number": self.frame_count,
                "timestamp": time.time()
            }
        
        # Mark new vs duplicate plates
        plates_with_status = []
        for plate in result.get("plates", []):
            plate_number = plate["plate_number"]
            is_new = not self.is_duplicate(plate_number)
            
            plates_with_status.append({
                "plate_number": plate_number,
                "raw_text": plate.get("raw_text", ""),
                "confidence": plate["ocr_confidence"],
                "detection_confidence": plate["detection_confidence"],
                "bbox": plate["bbox"],
                "is_new": is_new
            })
        
        # Cleanup old entries periodically
        if self.frame_count % 100 == 0:
            self.cleanup_old_plates()
        
        self.processed_count += 1
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "type": "plate_detection",
            "timestamp": time.time(),
            "frame_number": self.frame_count,
            "processed_frame_number": self.processed_count,
            "plates": plates_with_status,
            "plates_detected": len(plates_with_status),
            "new_plates": sum(1 for p in plates_with_status if p["is_new"]),
            "processing_time_ms": processing_time
        }
    
    def reset_state(self):
        """Reset processor state (for new session)"""
        self.frame_count = 0
        self.processed_count = 0
        self.seen_plates.clear()
        # State reset (silent)
    
    def get_stats(self):
        """Get processor statistics"""
        return {
            "total_frames": self.frame_count,
            "processed_frames": self.processed_count,
            "skip_rate": f"1/{self.skip_frames}",
            "tracked_plates": len(self.seen_plates)
        }
