#!/usr/bin/env python3
"""
Quick performance test for PaddleOCR optimizations
"""
import os
import sys
import time
import multiprocessing
from dotenv import load_dotenv

load_dotenv()

def check_environment():
    """Check environment variables"""
    print("=" * 60)
    print("🔍 Environment Configuration Check")
    print("=" * 60)
    
    cpu_count = multiprocessing.cpu_count()
    print(f"\n💻 System Info:")
    print(f"  • Total CPU Cores: {cpu_count}")
    
    env_vars = [
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS', 
        'OPENBLAS_NUM_THREADS',
        'CPU_NUM',
        'FLAGS_fraction_of_gpu_memory_to_use',
        'FLAGS_eager_delete_tensor_gb',
        'FLAGS_fast_eager_deletion_mode'
    ]
    
    print(f"\n⚙️  Performance Environment Variables:")
    for var in env_vars:
        value = os.getenv(var, 'Not Set')
        print(f"  • {var}: {value}")
    
    print()

def test_paddleocr():
    """Test PaddleOCR initialization and performance"""
    print("=" * 60)
    print("🚀 PaddleOCR Performance Test")
    print("=" * 60)
    
    try:
        print("\n📦 Loading PaddleOCR...")
        start = time.time()
        
        from models.license_plate.reader import PlateReader
        
        reader = PlateReader()
        load_time = time.time() - start
        
        print(f"✓ PaddleOCR loaded in {load_time:.2f}s")
        print(f"  • Engine: {reader.reader_type}")
        print(f"  • Min Confidence: {reader.min_confidence}")
        
        # Check if test images exist
        test_dir = "test_imgaes"  # Note: typo in original folder name
        if os.path.exists(test_dir):
            test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            if test_images:
                print(f"\n🖼️  Found {len(test_images)} test images")
                print("  Testing OCR performance...")
                
                import cv2
                times = []
                
                for img_file in test_images[:3]:  # Test first 3 images
                    img_path = os.path.join(test_dir, img_file)
                    img = cv2.imread(img_path)
                    
                    if img is not None:
                        start = time.time()
                        result = reader.read_from_cropped(img)
                        elapsed = (time.time() - start) * 1000
                        times.append(elapsed)
                        
                        if result:
                            print(f"  • {img_file}: {result['text']} ({elapsed:.0f}ms, conf: {result['confidence']})")
                        else:
                            print(f"  • {img_file}: No text detected ({elapsed:.0f}ms)")
                
                if times:
                    avg_time = sum(times) / len(times)
                    print(f"\n📊 Average OCR Time: {avg_time:.0f}ms")
                    
                    if avg_time < 150:
                        print("  ✓ Excellent performance!")
                    elif avg_time < 300:
                        print("  ✓ Good performance")
                    else:
                        print("  ⚠️  Consider enabling GPU or adjusting thread count")
            else:
                print(f"\n⚠️  No test images found in {test_dir}")
        else:
            print(f"\n⚠️  Test directory '{test_dir}' not found")
            print("  Place test images in this directory to test OCR performance")
        
        print("\n✓ PaddleOCR is ready and optimized!")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    check_environment()
    test_paddleocr()
    
    print("\n" + "=" * 60)
    print("📝 Summary")
    print("=" * 60)
    print("If you see 'PaddleOCR GPU mode enabled', GPU acceleration is active.")
    print("If you see 'PaddleOCR CPU mode optimized', multi-threading is active.")
    print("\nTo monitor real-time performance:")
    print("  • Windows: Task Manager > Performance tab")
    print("  • Linux: htop or top command")
    print("  • Mac: Activity Monitor")
    print("\nExpected CPU usage during OCR: 50-75%")
    print("=" * 60)

if __name__ == "__main__":
    # Set environment variables before importing
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    num_threads = max(2, int(cpu_count * 0.6))
    
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
    os.environ['CPU_NUM'] = str(num_threads)
    os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.5'
    os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'
    os.environ['FLAGS_fast_eager_deletion_mode'] = 'true'
    
    main()
