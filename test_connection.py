#!/usr/bin/env python3
"""
Simple test script to verify the WebSocket connection is working
"""
import asyncio
import websockets
import base64
import cv2
import numpy as np
import requests

BACKEND_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"

async def test_websocket_connection():
    """Test WebSocket connection to a stream"""
    try:
        # First, get active streams
        print("Fetching active streams...")
        resp = requests.get(f"{BACKEND_URL}/active_streams")
        if not resp.ok:
            print(f"Failed to get active streams: {resp.status_code}")
            return
        
        streams = resp.json()
        if not streams:
            print("No active streams found. Add a stream first.")
            return
        
        print(f"Found {len(streams)} active streams:")
        for stream in streams:
            print(f"  - {stream['name']} (ID: {stream['stream_id']})")
        
        # Test connection to first stream
        stream_id = streams[0]['stream_id']
        print(f"\nTesting WebSocket connection to stream: {stream_id}")
        
        async with websockets.connect(f"{WS_URL}/{stream_id}") as ws:
            print("WebSocket connected successfully!")
            
            # Try to receive a few frames
            for i in range(3):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    frame_data = base64.b64decode(msg)
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        print(f"Received frame {i+1}: {frame.shape}")
                    else:
                        print(f"Failed to decode frame {i+1}")
                        
                except asyncio.TimeoutError:
                    print(f"Timeout waiting for frame {i+1}")
                    break
                except Exception as e:
                    print(f"Error receiving frame {i+1}: {e}")
                    break
            
            print("WebSocket test completed!")
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    print("Testing WebSocket connection...")
    asyncio.run(test_websocket_connection())