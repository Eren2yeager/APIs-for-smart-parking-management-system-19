# Smart Parking - Python Backend

AI/ML processing and WebSocket server for real-time parking management.

## Features
- Vehicle detection using YOLOv8
- **Two-stage license plate recognition:**
  - Stage 1: YOLO detects plate regions (auto-downloads model on first run)
  - Stage 2: PaddleOCR reads text from detected plates (faster & more accurate than EasyOCR)
- WebSocket server for real-time updates
- Camera feed processing

## Setup

1. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python main.py
```

**First run:** The license plate detection model (~6MB) will auto-download from Hugging Face. This only happens once.

Server will start at `http://localhost:8000`

## API Endpoints

- `GET /` - Root endpoint
- `GET /api/health` - Health check
- `POST /api/detect-vehicle` - Detect vehicles in image
- `POST /api/recognize-plate` - Two-stage plate recognition (detect → crop → OCR)
- `WS /ws` - WebSocket connection for real-time updates

## Testing

Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

Or use the test script:
```bash
python test_plate_detection.py path/to/image.jpg
```

**Batch test cropped plates:**
```bash
python test_all_crops.py
```

**Test OCR directly on an image:**
```bash
python test_ocr_directly.py debug_crops/plate_image.jpg
```

## Documentation

- **[PLATE_RECOGNITION_GUIDE.md](PLATE_RECOGNITION_GUIDE.md)** - Complete guide on the two-stage pipeline
- **[models/README.md](models/README.md)** - Model documentation
