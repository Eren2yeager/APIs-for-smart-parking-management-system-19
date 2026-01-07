"""
Real-time parking lot capacity monitoring stream processor
Handles frame skipping, capacity alerts, and state tracking
"""

import time
import os
from dotenv import load_dotenv
from .parking_detector import ParkingSlotDetector

load_dotenv()


class ParkingStreamProcessor:
    """
    Processes video stream frames for parking lot capacity monitoring
    Features:
    - Frame skipping (parking status changes slowly)
    - Capacity alerts (near full)
    - State change detection
    """
    
    def __init__(self, skip_frames=None, capacity_threshold=None, max_capacity=None):
        """
        Initialize parking stream processor
        
        Args:
            skip_frames: Process every Nth frame (default from env or 10)
            capacity_threshold: Alert threshold 0-1 (default from env or 0.9)
            max_capacity: Maximum allowed vehicles (default from env or None)
        """
        self.detector = ParkingSlotDetector()
        
        # Configuration
        self.skip_frames = skip_frames or int(os.getenv("LOT_FRAME_SKIP", "10"))
        self.capacity_threshold = capacity_threshold or float(os.getenv("LOT_CAPACITY_ALERT", "0.9"))
        self.max_capacity = max_capacity or (int(os.getenv("LOT_MAX_CAPACITY")) if os.getenv("LOT_MAX_CAPACITY") else None)
        
        # State
        self.frame_count = 0
        self.processed_count = 0
        self.last_occupancy = None
        self.last_alert_time = 0
        self.alert_cooldown = 30  # seconds between alerts
        
        # Removed verbose initialization log
    
    def should_process_frame(self):
        """Determine if current frame should be processed"""
        return self.frame_count % self.skip_frames == 0
    
    def should_alert(self, occupancy_rate):
        """
        Check if capacity alert should be triggered
        
        Args:
            occupancy_rate: Current occupancy rate (0-1)
            
        Returns:
            bool: True if alert should be sent
        """
        current_time = time.time()
        
        # Check if above threshold
        if occupancy_rate < self.capacity_threshold:
            return False
        
        # Check cooldown (don't spam alerts)
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False
        
        self.last_alert_time = current_time
        return True
    
    def detect_state_change(self, current_occupancy):
        """
        Detect significant changes in parking occupancy
        
        Args:
            current_occupancy: Current number of occupied slots
            
        Returns:
            dict: Change information or None
        """
        if self.last_occupancy is None:
            self.last_occupancy = current_occupancy
            return None
        
        change = current_occupancy - self.last_occupancy
        
        if abs(change) >= 1:  # At least 1 slot changed
            change_info = {
                "previous": self.last_occupancy,
                "current": current_occupancy,
                "change": change,
                "direction": "increased" if change > 0 else "decreased"
            }
            self.last_occupancy = current_occupancy
            return change_info
        
        return None
    
    def process_frame(self, frame_bytes):
        """
        Process a single frame from video stream
        
        Args:
            frame_bytes: Image data as bytes
            
        Returns:
            dict: Processing result with slot statuses and metadata, or None if frame skipped
        """
        self.frame_count += 1
        start_time = time.time()
        
        # Frame skipping
        if not self.should_process_frame():
            return None
        
        # Detect parking slots
        result = self.detector.detect_slots(frame_bytes)
        
        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error", "Unknown error"),
                "frame_number": self.frame_count,
                "timestamp": time.time()
            }
        
        # Extract data
        total_slots = result["total_slots"]
        occupied = result["occupied"]
        empty = result["empty"]
        occupancy_rate = result["occupancy_rate"]
        
        # Check for alerts
        alert = self.should_alert(occupancy_rate)
        
        # Detect state changes
        state_change = self.detect_state_change(occupied)
        
        # Check capacity limit
        over_capacity = False
        if self.max_capacity and occupied > self.max_capacity:
            over_capacity = True
        
        self.processed_count += 1
        processing_time = int((time.time() - start_time) * 1000)
        
        response = {
            "success": True,
            "type": "capacity_update",
            "timestamp": time.time(),
            "frame_number": self.frame_count,
            "processed_frame_number": self.processed_count,
            "total_slots": total_slots,
            "occupied": occupied,
            "empty": empty,
            "occupancy_rate": occupancy_rate,
            "alert": alert,
            "over_capacity": over_capacity,
            "slots": result["slots"],
            "processing_time_ms": processing_time
        }
        
        # Add state change info if detected
        if state_change:
            response["state_change"] = state_change
        
        return response
    
    def reset_state(self):
        """Reset processor state (for new session)"""
        self.frame_count = 0
        self.processed_count = 0
        self.last_occupancy = None
        self.last_alert_time = 0
        # State reset (silent)
    
    def get_stats(self):
        """Get processor statistics"""
        return {
            "total_frames": self.frame_count,
            "processed_frames": self.processed_count,
            "skip_rate": f"1/{self.skip_frames}",
            "current_occupancy": self.last_occupancy
        }
