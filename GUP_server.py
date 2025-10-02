import threading, time, cv2, base64, tempfile, os, json, boto3, uuid, queue
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header , Body
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from pathlib import Path

app = FastAPI()

# Configuration
S3_BUCKET = os.getenv("S3_BUCKET", "violence-detector-bucket")
API_KEY = os.getenv("API_KEY", "default-secret-key")
S3_LOGS_KEY = "logs/violence_detection_log.json"

# AWS clients
s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "eu-west-2"))

# YOLO model
model = YOLO("violence_detection_v4.pt")

# Global state
STREAMS = {}

ID2CLASS = {0: "violence", 1: "nonviolence", 2: "fall"}

def get_class_name(cls_id: int) -> str:
    return ID2CLASS.get(cls_id, f"class_{cls_id}")

def verify_api_key(api_key: str = Header(...)):
    """Verify API key for protected endpoints"""
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def log_incident(stream_name, confidence, clip_path=None, snapshot_key=None):
    """Log detection incident to S3"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": ts, 
        "stream": stream_name, 
        "confidence": confidence,
        "clip": clip_path,
        "snapshot": snapshot_key
    }
    
    # Load existing logs
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_LOGS_KEY)
        logs = json.loads(obj["Body"].read().decode("utf-8"))
    except:
        logs = []
    
    # Add new entry and save
    logs.append(entry)
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=S3_LOGS_KEY,
        Body=json.dumps(logs, indent=2).encode("utf-8"),
        ContentType="application/json"
    )

def detection_loop(stream_id):
    """Main detection loop running on GPU"""
    stream = STREAMS[stream_id]
    
    # Handle file paths
    video_source = stream["url"]
    if stream.get("is_demo", False) and not video_source.startswith(('http://', 'https://', 'rtsp://')):
        video_source = os.path.abspath(video_source)
    
    # Configure video capture
    if str(video_source).startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
            "rtsp_transport;tcp|max_delay;0|stimeout;5000000"
        cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {video_source}")
        STREAMS[stream_id]["running"] = False
        return
    
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_skip = 3
    frame_count = 0
    
    print(f"GPU: Detection loop started for stream {stream_id}")
    
    while stream.get("running", False):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        
        frame_count += 1
        
        # Prepare display frame
        display_frame = cv2.resize(frame, (640, 360))
        
        # Skip YOLO processing for some frames
        if frame_count % frame_skip != 0:
            continue
        
        # YOLO inference on smaller frame
        small_frame = cv2.resize(frame, (320, 320))
        
        try:
            results = model(small_frame)[0]
            confidence = max([float(det.conf[0].item()) for det in results.boxes]) if results.boxes else 0.0
            detected_classes = []

            if results.boxes is not None:
                for box in results.boxes:
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    cls_name = get_class_name(cls_id)
                    detected_classes.append(cls_name)

                    if conf >= 0.3:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        scale_x = frame.shape[1] / 320
                        scale_y = frame.shape[0] / 320
                        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                        color = (0, 255, 0) if cls_name == "nonviolence" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls_name.upper()} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception as e:
            print(f"YOLO inference error: {e}")
            confidence = 0.0
            detected_classes = []
        
        # Create annotated display frame
        display_frame = frame.copy()
        
        # Check if actual violence or fall is detected (not just high confidence nonviolence)
        actual_violence_detected = confidence >= stream["threshold"] and any(c in ["violence", "fall"] for c in detected_classes)
        
        if actual_violence_detected:
            cv2.rectangle(display_frame, (10, 10), (400, 80), (0, 0, 255), -1)
            cv2.putText(display_frame, f"VIOLENCE DETECTED! {confidence:.2f}", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        else:
            cv2.rectangle(display_frame, (10, 10), (300, 60), (0, 255, 0), -1)
            cv2.putText(display_frame, f"SAFE {confidence:.2f}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display_frame, timestamp, (10, display_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Resize for streaming
        display_frame = cv2.resize(display_frame, (640, 360))
        
        # Handle clip recording - use the same logic as display
        if actual_violence_detected:
            # Save snapshot
            snapshot_path = f"{stream_id}_{int(time.time())}.jpg"
            cv2.imwrite(snapshot_path, frame)
            s3_key_snapshot = f"snapshots/{Path(snapshot_path).name}"
            
            try:
                s3.upload_file(snapshot_path, S3_BUCKET, s3_key_snapshot,
                              ExtraArgs={"ContentType": "image/jpeg"})
                log_incident(stream["name"], confidence, snapshot_key=s3_key_snapshot)
            except Exception as e:
                print(f"Snapshot upload failed: {e}")
            finally:
                try:
                    os.remove(snapshot_path)
                except:
                    pass
    
    # Cleanup
    cap.release()
    print(f"GPU: Detection loop ended for stream {stream_id}")


@app.get("/video/{stream_id}")
def stream_video(stream_id: str):
    """MJPEG stream endpoint with real-time detection"""
    def generate():
        if stream_id not in STREAMS or not STREAMS[stream_id].get("running", False):
            error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Stream not available", (50, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return
        
        stream = STREAMS[stream_id]
        video_source = stream["url"]
        
        # Handle file paths
        if stream.get("is_demo", False) and not video_source.startswith(('http://', 'https://', 'rtsp://')):
            video_source = os.path.abspath(video_source)
        
        # Configure video capture
        if str(video_source).startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
                "rtsp_transport;tcp|max_delay;0|stimeout;5000000"
            cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            error_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, f"Could not open video source", (50, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            cap.release()
            return
        
        frame_count = 0
        frame_skip = 2  # Process every 2nd frame for better performance
        
        while STREAMS[stream_id].get("running", False):
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            
            frame_count += 1
            
            # Resize frame for display
            display_frame = cv2.resize(frame, (640, 360))
            
            # Initialize defaults
            confidence = 0.0
            detected_classes = []
            
            # Skip YOLO processing for some frames to improve streaming performance
            if frame_count % frame_skip == 0:
                try:
                    # YOLO inference on smaller frame
                    small_frame = cv2.resize(frame, (320, 320))
                    results = model(small_frame)[0]
                    confidence = max([float(det.conf[0].item()) for det in results.boxes]) if results.boxes else 0.0
                    detected_classes = []
                    
                    if results.boxes is not None:
                        for box in results.boxes:
                            conf = float(box.conf[0].item())
                            cls_id = int(box.cls[0].item())
                            cls_name = get_class_name(cls_id)
                            detected_classes.append(cls_name)  # Add to detected classes list
                            
                            if conf >= 0.3:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                # Scale back to display size
                                scale_x = display_frame.shape[1] / 320
                                scale_y = display_frame.shape[0] / 320
                                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                                
                                color = (0, 255, 0) if cls_name == "nonviolence" else (0, 0, 255)
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                label = f"{cls_name.upper()} {conf:.2f}"
                                cv2.putText(display_frame, label, (x1, y1 - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Check for actual violence/fall detection (not just high confidence nonviolence)
                    actual_violence_detected = confidence >= stream["threshold"] and any(c in ["violence", "fall"] for c in detected_classes)
                    
                    if actual_violence_detected:
                        cv2.rectangle(display_frame, (10, 10), (400, 80), (0, 0, 255), -1)
                        cv2.putText(display_frame, f"VIOLENCE DETECTED! {confidence:.2f}", 
                                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                    else:
                        cv2.rectangle(display_frame, (10, 10), (300, 60), (0, 255, 0), -1)
                        cv2.putText(display_frame, f"SAFE {confidence:.2f}", 
                                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                    
                except Exception as e:
                    print(f"Detection error in streaming: {e}")
                    confidence = 0.0
                    detected_classes = []
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, timestamp, (10, display_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Encode and yield frame
            _, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        cap.release()
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/start_stream")
async def gpu_start_stream(data: dict = Body(...), api_key: str = Header(...)):
    """Start stream processing on GPU (called by CPU service)"""
    verify_api_key(api_key)
    
    
    # Extract only the fields we actually use
    stream_id = data["stream_id"]
    STREAMS[stream_id] = {
        "stream_id": stream_id,
        "name": data.get("name", f"stream_{stream_id}"),
        "url": data["url"],
        "threshold": data.get("threshold", 0.5),
        "phone": data.get("phone", ""),
        "is_demo": data.get("is_demo", False),
        "running": True ,
        "created_at": data.get("created_at","")
    }
    
    # Start detection loop in background thread
    thread = threading.Thread(target=detection_loop, args=(stream_id,), daemon=True)
    thread.start()
    
    return {"status": "started", "stream_id": stream_id}


@app.post("/stop_stream")
async def gpu_stop_stream(data: dict = Body(...), api_key: str = Header(...)):
    """Stop stream processing on GPU (called by CPU service)"""
    verify_api_key(api_key)
    
    stream_id = data["stream_id"]
    if stream_id in STREAMS:
        STREAMS[stream_id]["running"] = False
        return {"status": "stopped", "stream_id": stream_id}
    else:
        raise HTTPException(status_code=404, detail="Stream not found")

@app.post("/delete_stream")
async def gpu_delete_stream(data: dict = Body(...), api_key: str = Header(...)):
    """Delete stream from GPU (called by CPU service)"""
    verify_api_key(api_key)
    
    stream_id = data["stream_id"]
    if stream_id in STREAMS:
        STREAMS[stream_id]["running"] = False
        # Cleanup resources
        del STREAMS[stream_id]
    
    return {"status": "deleted", "stream_id": stream_id}

@app.get("/health")
async def health_check():
    """GPU service health check"""
    return {"status": "healthy", "service": "gpu_worker", "active_streams": len(STREAMS)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)