import os
import sys
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from websocket.connection_manager import ConnectionManager
from models.vehicle_detector import VehicleDetector
from models.license_plate import PlateRecognitionPipeline
import uvicorn

app = FastAPI(title="Smart Parking API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")],
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


if __name__ == "__main__":
    # This is now handled by run.py
    # Use: python run.py
    
    print("🚗 Smart Parking API Starting...")
    print("\n📦 Models will load on first request")
    print("  - Vehicle Detector: YOLOv8")
    print("  - License Plate Detector: YOLOv8")
    print("  - OCR: PaddleOCR\n")
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    # Use reload=False to avoid double loading
    # Set reload=True only during development if needed
    reload_mode = os.getenv("DEV_MODE", "false").lower() == "true"
    
    uvicorn.run("main:app", host=host, port=port, reload=reload_mode)

