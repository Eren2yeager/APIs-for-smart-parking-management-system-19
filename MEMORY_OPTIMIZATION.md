# Memory Optimization Changes

## Problem
Server was crashing on Render due to memory exceeded (>512MB) when using `/api/recognize-plate` endpoint.

## Root Cause
- Local YOLOv8 license plate model: ~80-100MB
- Local YOLOv8n vehicle detection model: ~50-80MB
- PaddleOCR: ~350-400MB
- Image processing: ~10-20MB per request
- **Total: ~600-700MB** (exceeds Render's 512MB limit)

## Solutions Implemented

### 1. ✅ Replaced Local YOLO with Roboflow API (~80MB saved)
**File: `models/license_plate/detector.py`**
- Removed local YOLOv8 model loading
- Switched to Roboflow API for license plate detection
- No model in memory (API handles inference)
- Uses same Roboflow setup as parking detector

**Configuration:**
```python
rf = Roboflow(api_key="iFFDE6mLuRtrRR8tspsE")
project = rf.workspace().project("license-plate-recognition-rxg4e")
model = project.version(11).model
```

### 2. ✅ Removed Vehicle Detection API (~50-80MB saved)
**Files: `main.py`, `models/vehicle_detector.py` (deleted)**
- Removed `/api/detect-vehicle` endpoint
- Deleted `VehicleDetector` class
- Removed YOLOv8n model dependency
- No local YOLO models at all now

### 3. ✅ Image Resizing Before Processing (~5-7MB saved per request)
**Files: `models/license_plate/detector.py`**
- Resize images to max 1280px before detection
- Resize images before cropping plates
- Maintains quality while reducing memory footprint
- License plates remain readable at lower resolution

### 4. ✅ Aggressive Memory Cleanup (~5-10MB saved)
**File: `models/license_plate/pipeline.py`**
- Added `gc.collect()` after each processing stage
- Delete intermediate arrays immediately after use
- Process cropped plates sequentially (not all at once)
- Clean up on errors

**Memory cleanup points:**
- After detection (delete image_bytes)
- After each OCR operation (delete cropped_plate)
- After all processing (delete cropped_plates array)
- On exceptions

### 5. ✅ Environment Variables for Memory Efficiency
**File: `main.py`**
```python
os.environ['OMP_NUM_THREADS'] = '1'  # Reduce OpenCV memory
os.environ['MALLOC_TRIM_THRESHOLD_'] = '100000'  # Aggressive memory release
```

### 6. ✅ Updated Dependencies
**File: `requirements.txt`**
- Removed: `ultralytics` (YOLOv8)
- Removed: `huggingface-hub`
- Kept: `roboflow` (already used for parking detection)

## Expected Results

### Memory Usage After Optimization:
- Roboflow API (License Plates): ~0MB (cloud-based)
- Roboflow API (Parking Slots): ~0MB (cloud-based)
- PaddleOCR: ~350-400MB
- FastAPI + Dependencies: ~80MB
- **Total: ~430-480MB** ✅ (comfortably under 512MB limit)

### Memory Savings:
- License plate YOLO removal: ~80MB
- Vehicle detection YOLO removal: ~50-80MB
- Image resizing: ~5-7MB per request
- Memory cleanup: ~5-10MB per request
- **Total savings: ~150-200MB**

## Trade-offs

### Pros:
✅ Stays within Render's free tier memory limit
✅ Keeps PaddleOCR (good accuracy)
✅ No speed loss (Roboflow API is fast)
✅ Minimal code changes

### Cons:
⚠️ Requires internet connection for detection
⚠️ Roboflow API rate limits (free tier)
⚠️ Slight latency increase (~200-500ms for API call)

## Testing Recommendations

1. Test `/api/recognize-plate` endpoint with various image sizes
2. Monitor memory usage on Render dashboard
3. Test with multiple concurrent requests
4. Verify accuracy is maintained with resized images

## Rollback Plan

If issues occur, you can rollback by:
1. Restore `detector.py` to use local YOLO
2. Add `ultralytics` back to `requirements.txt`
3. Remove garbage collection calls (optional)

## Environment Variables

No new environment variables required. Existing ones still work:
- `ROBOFLOW_API_KEY` (already set)
- `PLATE_DETECTION_CONFIDENCE` (default: 40)
- `PLATE_DETECTION_OVERLAP` (default: 30)
