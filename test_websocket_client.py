"""
Test WebSocket client for gate and lot monitoring
Tests both endpoints with sample images
"""

import asyncio
import websockets
import json
import base64
import sys
import os


async def test_gate_monitor(image_path):
    """Test gate monitor WebSocket endpoint"""
    uri = "ws://localhost:8000/ws/gate-monitor"
    
    # Read image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    print(f"\n🚪 Testing Gate Monitor with: {image_path}")
    print("=" * 60)
    
    async with websockets.connect(uri) as websocket:
        # Receive connection message
        response = await websocket.recv()
        print(f"Connected: {json.loads(response)}")
        
        # Send frames (simulate stream)
        for i in range(10):
            message = {
                "type": "frame",
                "data": image_base64,
                "timestamp": asyncio.get_event_loop().time(),
                "gate_id": "entrance_1"
            }
            
            await websocket.send(json.dumps(message))
            print(f"\n📤 Sent frame {i+1}")
            
            # Receive response (may be None if frame skipped)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                result = json.loads(response)
                
                if result.get("type") == "plate_detection":
                    print(f"✅ Frame {result['frame_number']} processed:")
                    print(f"   Plates detected: {result['plates_detected']}")
                    print(f"   New plates: {result['new_plates']}")
                    print(f"   Processing time: {result['processing_time_ms']}ms")
                    
                    for plate in result.get('plates', []):
                        status = "🆕 NEW" if plate['is_new'] else "🔄 SEEN"
                        print(f"   {status} {plate['plate_number']} (conf: {plate['confidence']})")
                        print(f"      BBox: {plate['bbox']}")
            
            except asyncio.TimeoutError:
                print(f"⏭️  Frame {i+1} skipped (no response)")
            
            await asyncio.sleep(0.1)  # Simulate frame rate
        
        # Get stats
        try:
            await websocket.send(json.dumps({"type": "stats"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            stats = json.loads(response)
            print(f"\n📊 Final Stats: {stats.get('data', stats)}")
        except Exception as e:
            print(f"\n⚠️  Could not get stats: {e}")


async def test_lot_monitor(image_path):
    """Test lot monitor WebSocket endpoint"""
    uri = "ws://localhost:8000/ws/lot-monitor"
    
    # Read image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    print(f"\n🅿️  Testing Lot Monitor with: {image_path}")
    print("=" * 60)
    
    async with websockets.connect(uri) as websocket:
        # Receive connection message
        response = await websocket.recv()
        print(f"Connected: {json.loads(response)}")
        
        # Send frames (simulate stream) - need more frames due to skip rate
        for i in range(15):
            message = {
                "type": "frame",
                "data": image_base64,
                "timestamp": asyncio.get_event_loop().time(),
                "lot_id": "parking_lot_A"
            }
            
            await websocket.send(json.dumps(message))
            print(f"\n📤 Sent frame {i+1}")
            
            # Receive response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                result = json.loads(response)
                
                if result.get("type") == "capacity_update":
                    print(f"✅ Frame {result['frame_number']} processed:")
                    print(f"   Total slots: {result['total_slots']}")
                    print(f"   Occupied: {result['occupied']}")
                    print(f"   Empty: {result['empty']}")
                    print(f"   Occupancy rate: {result['occupancy_rate']*100:.1f}%")
                    print(f"   Alert: {'🚨 YES' if result['alert'] else '✅ NO'}")
                    print(f"   Processing time: {result['processing_time_ms']}ms")
                    
                    if result.get('state_change'):
                        change = result['state_change']
                        print(f"   📈 State change: {change['direction']} by {abs(change['change'])}")
            
            except asyncio.TimeoutError:
                print(f"⏭️  Frame {i+1} skipped (no response)")
            
            await asyncio.sleep(0.2)  # Simulate frame rate
        
        # Get stats
        try:
            await websocket.send(json.dumps({"type": "stats"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            stats = json.loads(response)
            print(f"\n📊 Final Stats: {stats.get('data', stats)}")
        except Exception as e:
            print(f"\n⚠️  Could not get stats: {e}")


async def main():
    """Main test function"""
    if len(sys.argv) < 3:
        print("Usage: python test_websocket_client.py <gate|lot> <image_path>")
        print("\nExamples:")
        print("  python test_websocket_client.py gate debug_crops/plate_20260107_131153_0_original.jpg")
        print("  python test_websocket_client.py lot parking_lot_image.jpg")
        sys.exit(1)
    
    endpoint = sys.argv[1]
    image_path = sys.argv[2]
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    try:
        if endpoint == "gate":
            await test_gate_monitor(image_path)
        elif endpoint == "lot":
            await test_lot_monitor(image_path)
        else:
            print(f"❌ Error: Unknown endpoint '{endpoint}'. Use 'gate' or 'lot'")
            sys.exit(1)
    
    except websockets.exceptions.WebSocketException as e:
        print(f"\n❌ WebSocket Error: {e}")
        print("Make sure the server is running: python main.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
