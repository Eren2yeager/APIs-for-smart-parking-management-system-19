import os
import sys
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from websocket.connection_manager import ConnectionManager
from models.vehicle_detector import VehicleDetector
from models.license_plate import PlateRecognitionPipeline
from models.license_plate.stream_processor import PlateStreamProcessor
from models.parking_space_detector import ParkingSlotDetector
from models.parking_space_detector.stream_processor import ParkingStreamProcessor
from utils.frame_utils import base64_to_bytes
import uvicorn
import json

app = FastAPI(title="Smart Parking API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers (lazy loading to avoid double init with reload=True)
manager = ConnectionManager()
vehicle_detector = None
plate_pipeline = None
parking_detector = None


def get_vehicle_detector():
    global vehicle_detector
    if vehicle_detector is None:
        vehicle_detector = VehicleDetector()
    return vehicle_detector


def get_plate_pipeline():
    global plate_pipeline
    if plate_pipeline is None:
        plate_pipeline = PlateRecognitionPipeline()
    return plate_pipeline


def get_parking_detector():
    global parking_detector
    if parking_detector is None:
        parking_detector = ParkingSlotDetector()
    return parking_detector



@app.get("/")
async def root():
    return {"message": "Smart Parking API is running"}


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "smart-parking-backend"}


@app.post("/api/detect-vehicle")
async def detect_vehicle(file: UploadFile = File(...)):
    """Detect vehicles in uploaded image"""
    contents = await file.read()
    detector = get_vehicle_detector()
    result = detector.detect(contents)
    return result


@app.post("/api/recognize-plate")
async def recognize_plate(file: UploadFile = File(...)):
    """Recognize license plate using two-stage pipeline"""
    contents = await file.read()
    pipeline = get_plate_pipeline()
    result = pipeline.process(contents)
    return result


@app.post("/api/detect-parking-slots")
async def detect_parking_slots(file: UploadFile = File(...)):
    """
    Detect parking slot occupancy in parking lot image
    Returns: empty slots, occupied slots, and occupancy rate with bounding boxes
    """
    contents = await file.read()
    detector = get_parking_detector()
    result = detector.detect_slots(contents)
    return result



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"Message received: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/gate-monitor")
async def gate_monitor_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time license plate recognition at entry/exit gates
    
    Client sends frames, server responds with plate detections
    """
    await manager.connect(websocket)
    processor = PlateStreamProcessor()
    
    try:
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "Gate monitor ready",
            "config": {
                "frame_skip": processor.skip_frames,
                "dedup_window": processor.dedup_window
            }
        })
        
        print(f"✓ Gate Monitor connected | Frame skip: {processor.skip_frames}")
        
        while True:
            # Receive frame data
            data = await websocket.receive()
            
            # Handle binary data (raw image bytes)
            if "bytes" in data:
                frame_bytes = data["bytes"]
            
            # Handle text data (JSON with base64 image)
            elif "text" in data:
                try:
                    message = json.loads(data["text"])
                    
                    # Handle control messages
                    if message.get("type") == "reset":
                        processor.reset_state()
                        await websocket.send_json({
                            "type": "reset_ack",
                            "message": "State reset successful"
                        })
                        print("⟳ Gate Monitor state reset")
                        continue
                    
                    if message.get("type") == "stats":
                        stats = processor.get_stats()
                        await websocket.send_json({
                            "type": "stats",
                            "data": stats
                        })
                        continue
                    
                    # Extract frame data
                    if "data" in message:
                        base64_data = message["data"]
                        frame_bytes = base64_to_bytes(base64_data)
                    else:
                        continue
                        
                except json.JSONDecodeError as e:
                    print(f"✗ Gate Monitor: Invalid JSON")
                    await websocket.send_json({
                        "type": "error",
                        "error": "Invalid JSON format"
                    })
                    continue
                except Exception as e:
                    print(f"✗ Gate Monitor: Frame decode error - {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Failed to decode frame: {str(e)}"
                    })
                    continue
            else:
                continue
            
            # Process frame
            result = processor.process_frame(frame_bytes)
            
            # Send result (None if frame was skipped)
            if result:
                plates_count = result.get('plates_detected', 0)
                new_count = result.get('new_plates', 0)
                processing_time = result.get('processing_time_ms', 0)
                
                if plates_count > 0:
                    plates_str = ", ".join([p['plate_number'] for p in result.get('plates', [])])
                    print(f"🚗 Detected {plates_count} plate(s): {plates_str} | {processing_time}ms | New: {new_count}")
                else:
                    print(f"○ No plates detected | {processing_time}ms")
                
                await websocket.send_json(result)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        stats = processor.get_stats()
        print(f"✗ Gate Monitor disconnected | Processed: {stats['processed_frames']}/{stats['total_frames']} frames")
    except Exception as e:
        print(f"✗ Gate Monitor error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass
        manager.disconnect(websocket)


@app.websocket("/ws/lot-monitor")
async def lot_monitor_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time parking lot capacity monitoring
    
    Client sends frames, server responds with slot occupancy status
    """
    await manager.connect(websocket)
    processor = ParkingStreamProcessor()
    
    try:
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "Lot monitor ready",
            "config": {
                "frame_skip": processor.skip_frames,
                "capacity_threshold": processor.capacity_threshold,
                "max_capacity": processor.max_capacity
            }
        })
        
        print(f"✓ Lot Monitor connected | Frame skip: {processor.skip_frames} | Alert threshold: {int(processor.capacity_threshold*100)}%")
        
        while True:
            # Receive frame data
            data = await websocket.receive()
            
            # Handle binary data (raw image bytes)
            if "bytes" in data:
                frame_bytes = data["bytes"]
            
            # Handle text data (JSON with base64 image)
            elif "text" in data:
                try:
                    message = json.loads(data["text"])
                    
                    # Handle control messages
                    if message.get("type") == "reset":
                        processor.reset_state()
                        await websocket.send_json({
                            "type": "reset_ack",
                            "message": "State reset successful"
                        })
                        print("⟳ Lot Monitor state reset")
                        continue
                    
                    if message.get("type") == "stats":
                        stats = processor.get_stats()
                        await websocket.send_json({
                            "type": "stats",
                            "data": stats
                        })
                        continue
                    
                    # Extract frame data
                    if "data" in message:
                        base64_data = message["data"]
                        frame_bytes = base64_to_bytes(base64_data)
                    else:
                        continue
                        
                except json.JSONDecodeError as e:
                    print(f"✗ Lot Monitor: Invalid JSON")
                    await websocket.send_json({
                        "type": "error",
                        "error": "Invalid JSON format"
                    })
                    continue
                except Exception as e:
                    print(f"✗ Lot Monitor: Frame decode error - {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Failed to decode frame: {str(e)}"
                    })
                    continue
            else:
                continue
            
            # Process frame
            result = processor.process_frame(frame_bytes)
            
            # Send result (None if frame was skipped)
            if result:
                total = result.get('total_slots', 0)
                occupied = result.get('occupied', 0)
                occupancy = result.get('occupancy_rate', 0)
                alert = result.get('alert', False)
                processing_time = result.get('processing_time_ms', 0)
                
                alert_icon = "⚠️" if alert else "✓"
                print(f"{alert_icon} Parking: {occupied}/{total} slots ({int(occupancy*100)}%) | {processing_time}ms")
                
                await websocket.send_json(result)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        stats = processor.get_stats()
        print(f"✗ Lot Monitor disconnected | Processed: {stats['processed_frames']}/{stats['total_frames']} frames")
    except Exception as e:
        print(f"✗ Lot Monitor error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass
        manager.disconnect(websocket)


if __name__ == "__main__":
    print("=" * 60)
    print("🚗 Smart Parking Management System")
    print("=" * 60)
    print("\n📦 AI Models (lazy loading):")
    print("  • YOLOv8 - Vehicle Detection")
    print("  • YOLOv8 - License Plate Detection")
    print("  • PaddleOCR - Text Recognition")
    print("  • Roboflow - Parking Slot Detection")
    print("\n🌐 WebSocket Endpoints:")
    print("  • /ws/gate-monitor - License Plate Recognition")
    print("  • /ws/lot-monitor - Parking Capacity Monitoring")
    print("\n⚙️  Configuration:")
    print(f"  • Gate frame skip: {os.getenv('GATE_FRAME_SKIP', '1')}")
    print(f"  • Lot frame skip: {os.getenv('LOT_FRAME_SKIP', '1')}")
    print(f"  • Capacity alert: {int(float(os.getenv('LOT_CAPACITY_ALERT', '0.9'))*100)}%")
    print("=" * 60)
    print()
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run("main:app", host=host, port=port, reload=False)
