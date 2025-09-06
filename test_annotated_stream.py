#!/usr/bin/env python3
"""
Test script to verify annotated video stream is working
"""
import asyncio
import websockets
import base64
import cv2
import numpy as np
import requests
from datetime import datetime

BACKEND_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"

async def test_annotated_stream():
    """Test annotated WebSocket stream"""
    try:
        # Get active streams
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
        print(f"\nTesting annotated WebSocket stream: {stream_id}")
        
        async with websockets.connect(f"{WS_URL}/{stream_id}") as ws:
            print("WebSocket connected successfully!")
            
            # Try to receive a few annotated frames
            for i in range(5):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    frame_data = base64.b64decode(msg)
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        print(f"Received annotated frame {i+1}: {frame.shape}")
                        
                        # Save a sample frame to verify annotations
                        if i == 0:
                            cv2.imwrite(f"sample_annotated_frame_{stream_id}.jpg", frame)
                            print(f"Saved sample annotated frame as 'sample_annotated_frame_{stream_id}.jpg'")
                        
                        # Check if frame has annotations (look for colored rectangles)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray, 50, 150)
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # Look for rectangular shapes (annotations)
                        rect_count = 0
                        for contour in contours:
                            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                            if len(approx) == 4:  # Rectangle
                                rect_count += 1
                        
                        print(f"  - Detected {rect_count} rectangular annotations")
                        
                    else:
                        print(f"Failed to decode frame {i+1}")
                        
                except asyncio.TimeoutError:
                    print(f"Timeout waiting for frame {i+1}")
                    break
                except Exception as e:
                    print(f"Error receiving frame {i+1}: {e}")
                    break
            
            print("Annotated stream test completed!")
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    print("Testing annotated video stream...")
    asyncio.run(test_annotated_stream())