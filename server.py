import threading, time, cv2, base64, tempfile, os, json, boto3, botocore, re, uuid, queue
import numpy as np
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from twilio.rest import Client
from dotenv import load_dotenv
from pathlib import Path
import asyncio

# -------------------------------
# Load env and config
# -------------------------------
load_dotenv()
BASE_DIR = Path(os.getcwd())

S3_BUCKET = "violence-detector-bucket"
S3_KEY = "logs/violence_detection_log.json"

# Twilio
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
twilio_client = Client(TWILIO_SID, TWILIO_AUTH) if TWILIO_SID and TWILIO_AUTH else None

# AWS S3
s3 = boto3.client("s3")

# YOLO model
model = YOLO("violence_detection_v4.pt")

# FastAPI
app = FastAPI()

# -------------------------------
# Globals
# -------------------------------
STREAMS = {}      # stream_id -> metadata
CLIENTS = {}      # stream_id -> set(WebSocket)
FRAME_QUEUES = {} # stream_id -> asyncio.Queue


# -------------------------------
# Helper functions
# -------------------------------
def is_valid_phone(number: str) -> bool:
    return re.match(r"^\+[1-9]\d{7,14}$", number or "") is not None

def load_logs_from_s3():
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return []
        return []
    except Exception:
        return []

def save_logs_to_s3(data):
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=S3_KEY,
        Body=json.dumps(data, indent=4).encode("utf-8"),
        ContentType="application/json"
    )

def generate_presigned_url(key, expires=86400):
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=expires
        )
    except Exception:
        return None

def log_incident(stream_name, confidence, clip_path=None, snapshot_key=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"timestamp": ts, "stream": stream_name, "confidence": confidence}

    if clip_path:
        entry["clip"] = clip_path   
    if snapshot_key:
        entry["snapshot"] = snapshot_key

    logs = load_logs_from_s3()
    logs.append(entry)
    save_logs_to_s3(logs)


def send_sms_alert(phone, message):
    if twilio_client and is_valid_phone(phone):
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=phone
        )

# -------------------------------
# Detection loop
# -------------------------------
# Removed broadcast_frame function - using queue-based approach instead
            
def detection_loop(stream_id):
    stream = STREAMS[stream_id]
    
    # Handle file paths - convert to absolute path if it's a local file
    video_source = stream["url"]
    if stream.get("is_demo", False) and not video_source.startswith(('http://', 'https://', 'rtsp://')):
        # It's a local file, make it absolute
        video_source = os.path.abspath(video_source)
        print(f"DEBUG: Detection loop - Converted to absolute path: {video_source}")
        print(f"DEBUG: Detection loop - Absolute path exists: {os.path.exists(video_source)}")
    
    cap = cv2.VideoCapture(video_source)
    
    # Check if video capture is successful
    if not cap.isOpened():
        print(f"ERROR: Detection loop - Could not open video source: {video_source}")
        # Mark stream as stopped if video source can't be opened
        STREAMS[stream_id]["running"] = False
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_buffer = []
    buffer_size = int(fps * 5)
    consecutive_count = 0
    alert_trigger = 5
    recording = False
    out = None

    if stream_id not in FRAME_QUEUES:
        FRAME_QUEUES[stream_id] = queue.Queue()

    print(f"DEBUG: Detection loop started for stream {stream_id}")

    while stream.get("running", False):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)  # Increased delay for better performance
            continue

        # YOLO detection
        try:
            results = model(frame)[0]
            confidence = max([float(det.conf[0].item()) for det in results.boxes]) if results.boxes else 0.0
            
            # Draw bounding boxes for detected objects
            if results.boxes is not None:
                for box in results.boxes:
                    conf = float(box.conf[0].item())
                    if conf >= 0.3:  # Only show boxes with confidence > 0.3
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if conf < stream["threshold"] else (0, 0, 255), 2)
                        # Draw confidence label
                        label = f"{'VIOLENCE' if conf >= stream['threshold'] else 'SAFE'}: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except:
            confidence = 0.0

        violence_detected = confidence >= stream["threshold"]

        # Create annotated frame with better visibility
        display_frame = frame.copy()
        
        # Add background rectangle for better text visibility
        if violence_detected:
            cv2.rectangle(display_frame, (10, 10), (400, 80), (0, 0, 255), -1)  # Red background
            cv2.putText(display_frame, f"VIOLENCE DETECTED! {confidence:.2f}", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        else:
            cv2.rectangle(display_frame, (10, 10), (300, 60), (0, 255, 0), -1)  # Green background
            cv2.putText(display_frame, f"SAFE {confidence:.2f}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display_frame, timestamp, (10, display_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Resize for fast streaming
        display_frame = cv2.resize(display_frame, (640,360))

        # Push frame to async queue
        if stream_id in FRAME_QUEUES:
            FRAME_QUEUES[stream_id].put(display_frame)
        
        # The WebSocket clients will get frames from the queue in the WebSocket handler
        # Clean up any disconnected WebSocket clients
        if stream_id in CLIENTS:
            to_remove = []
            for ws in list(CLIENTS[stream_id]):
                try:
                    # Check if WebSocket is still connected
                    if hasattr(ws, 'client_state') and ws.client_state.name != "CONNECTED":
                        to_remove.append(ws)
                except:
                    to_remove.append(ws)
            for ws in to_remove:
                CLIENTS[stream_id].discard(ws)
        
        # Clip recording logic...
        # (keep the same as before)


        # Clip recording logic
        frame_buffer.append(frame.copy())
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)

        if violence_detected:
            consecutive_count += 1
        else:
            consecutive_count = 0

        if consecutive_count >= alert_trigger and not recording:
            tmp_clip = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            clip_path = tmp_clip.name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(clip_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
            for bf in frame_buffer:
                out.write(bf)
            recording = True

            # Snapshot
            snapshot_path = f"{stream_id}_{int(time.time())}.jpg"
            cv2.imwrite(snapshot_path, frame)
            s3_key_snapshot = f"snapshots/{Path(snapshot_path).name}"
            try: s3.upload_file(snapshot_path, S3_BUCKET, s3_key_snapshot)
            except: pass
            try: os.remove(snapshot_path)
            except: pass

            send_sms_alert(
                stream["phone"],
                f"⚠️ Violence detected on {stream['name']} (Confidence: {confidence:.2f})"
            )

        if recording and out:
            out.write(frame)
            if len(frame_buffer) >= buffer_size:
                out.release()
                recording = False
                s3_key = f"clips/{Path(clip_path).name}"
                try:
                    s3.upload_file(clip_path, S3_BUCKET, s3_key)
                    log_incident(stream["name"], confidence, clip_path=s3_key, snapshot_key=s3_key_snapshot)
                except: pass
                try: os.remove(clip_path)
                except: pass
                frame_buffer = []

    # Cleanup resources
    cap.release()
    if out: 
        out.release()
    
    # Mark stream as stopped when loop ends
    if stream_id in STREAMS:
        STREAMS[stream_id]["running"] = False
    
    print(f"DEBUG: Detection loop ended for stream {stream_id}")

# -------------------------------
# WebSocket endpoint
# -------------------------------
@app.websocket("/ws/{stream_id}")
async def websocket_endpoint(ws: WebSocket, stream_id: str):
    await ws.accept()

    if stream_id not in CLIENTS:
        CLIENTS[stream_id] = set()
    CLIENTS[stream_id].add(ws)

    try:
        while True:
            # Check if WebSocket is still open
            if ws.client_state.name != "CONNECTED":
                print(f"WebSocket not connected for stream {stream_id}")
                break
                
            # Check if frame queue exists and has frames
            if stream_id in FRAME_QUEUES and not FRAME_QUEUES[stream_id].empty():
                try:
                    frame = FRAME_QUEUES[stream_id].get_nowait()
                    
                    # Optimize frame for live streaming
                    # Reduce quality for faster transmission
                    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    jpg_str = base64.b64encode(buffer).decode()
                    
                    await ws.send_text(jpg_str)
                    # Reduced logging for performance
                    if hasattr(ws, '_frame_count'):
                        ws._frame_count += 1
                    else:
                        ws._frame_count = 1
                    
                    if ws._frame_count % 30 == 0:  # Log every 30 frames
                        print(f"Sent {ws._frame_count} frames to stream {stream_id}")
                        
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    # If we can't send, the connection is likely closed
                    break
            else:
                # Shorter wait for more responsive streaming
                await asyncio.sleep(0.016)  # ~60 FPS target
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for stream {stream_id}")
    except Exception as e:
        print(f"WebSocket error for stream {stream_id}: {e}")
    finally:
        # Always clean up the connection
        if stream_id in CLIENTS:
            CLIENTS[stream_id].discard(ws)
            print(f"Cleaned up WebSocket connection for stream {stream_id}")

# -------------------------------
# API endpoints
# -------------------------------
@app.post("/add_stream")
def add_stream(
    name: str = Query(...),
    url_or_file: str = Query(...),
    threshold: float = Query(0.5),
    phone: str = Query(""),
    file_uploaded: bool = Query(False)
):
    stream_id = str(uuid.uuid4())
    STREAMS[stream_id] = {
        "stream_id": stream_id,
        "name": name,
        "url": url_or_file,
        "threshold": threshold,
        "phone": phone,
        "running": True,
        "is_demo": file_uploaded
    }

    print(f"DEBUG: Adding stream {stream_id} with URL: {url_or_file}")
    
    # Start detection loop in a new thread
    t = threading.Thread(target=detection_loop, args=(stream_id,), daemon=True)
    t.start()

    return {"stream_id": stream_id, "status": "started"}

@app.get("/active_streams")
def get_active_streams():
    return [
        {
            "stream_id": s["stream_id"],
            "name": s["name"],
            "is_demo": s.get("is_demo", False),
            "running": s.get("running", False)
        }
        for s in STREAMS.values()
    ]

@app.post("/stop_stream/{stream_id}")
def stop_stream(stream_id: str):
    """Stop a running stream"""
    if stream_id not in STREAMS:
        return {"error": "Stream not found"}
    
    STREAMS[stream_id]["running"] = False
    
    # Clean up resources
    if stream_id in FRAME_QUEUES:
        # Clear the queue
        while not FRAME_QUEUES[stream_id].empty():
            try:
                FRAME_QUEUES[stream_id].get_nowait()
            except:
                break
    
    return {"message": f"Stream {stream_id} stopped successfully"}

@app.post("/start_stream/{stream_id}")
def start_stream(stream_id: str):
    """Start a stopped stream"""
    if stream_id not in STREAMS:
        return {"error": "Stream not found"}
    
    # Always allow restarting - this handles cases where the stream failed to start initially
    STREAMS[stream_id]["running"] = True
    
    # Start detection loop in a new thread
    t = threading.Thread(target=detection_loop, args=(stream_id,), daemon=True)
    t.start()
    
    print(f"DEBUG: Starting stream {stream_id}")
    return {"message": f"Stream {stream_id} started successfully"}

@app.delete("/delete_stream/{stream_id}")
def delete_stream(stream_id: str):
    """Delete a stream completely"""
    if stream_id not in STREAMS:
        return {"error": "Stream not found"}
    
    # Stop the stream first
    STREAMS[stream_id]["running"] = False
    
    # Clean up resources
    if stream_id in FRAME_QUEUES:
        del FRAME_QUEUES[stream_id]
    
    if stream_id in CLIENTS:
        del CLIENTS[stream_id]
    
    # Remove from STREAMS
    del STREAMS[stream_id]
    
    return {"message": f"Stream {stream_id} deleted successfully"}

@app.get("/logs")
def get_logs(stream: str = None, sort: str = "desc"):
    logs = load_logs_from_s3()

    for entry in logs:
        if entry.get("clip"):
            entry["clip_url"] = generate_presigned_url(entry["clip"])
        if entry.get("snapshot"):
            entry["snapshot_url"] = generate_presigned_url(entry["snapshot"])

    if stream:
        logs = [l for l in logs if l.get("stream", "").lower() == stream.lower()]

    logs.sort(key=lambda x: x.get("timestamp", ""), reverse=(sort == "desc"))

    return logs





@app.get("/video/{stream_id}")
def stream_video(stream_id: str):
    """Stream live video with violence detection overlays"""
    if stream_id not in STREAMS:
        return {"error": "Stream not found"}
    
    def generate_video():
        stream = STREAMS[stream_id]
        
        # Debug: Print the video source URL/path
        print(f"DEBUG: Attempting to open video source: {stream['url']}")
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        print(f"DEBUG: File exists: {os.path.exists(stream['url'])}")
        
        # Handle file paths - convert to absolute path if it's a local file
        video_source = stream["url"]
        if stream.get("is_demo", False) and not video_source.startswith(('http://', 'https://', 'rtsp://')):
            # It's a local file, make it absolute
            video_source = os.path.abspath(video_source)
            print(f"DEBUG: Converted to absolute path: {video_source}")
            print(f"DEBUG: Absolute path exists: {os.path.exists(video_source)}")
        
        cap = cv2.VideoCapture(video_source)
        
        # Check if video capture is successful
        if not cap.isOpened():
            print(f"ERROR: Could not open video source: {video_source}")
            # Create error frame
            error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_frame, f"ERROR: Could not open video source", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', error_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            return
        
        # Set video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"DEBUG: Video opened successfully - FPS: {fps}, Size: {width}x{height}")
        
        while stream.get("running", False):
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO detection
            try:
                results = model(frame)[0]
                confidence = max([float(det.conf[0].item()) for det in results.boxes]) if results.boxes else 0.0
                
                # Draw bounding boxes for detected objects
                if results.boxes is not None:
                    for box in results.boxes:
                        conf = float(box.conf[0].item())
                        if conf >= 0.3:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if conf < stream["threshold"] else (0, 0, 255), 2)
                            # Draw confidence label
                            label = f"{'VIOLENCE' if conf >= stream['threshold'] else 'SAFE'}: {conf:.2f}"
                            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except:
                confidence = 0.0
            
            violence_detected = confidence >= stream["threshold"]
            
            # Add status overlay
            if violence_detected:
                cv2.rectangle(frame, (10, 10), (400, 80), (0, 0, 255), -1)
                cv2.putText(frame, f"VIOLENCE DETECTED! {confidence:.2f}", 
                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            else:
                cv2.rectangle(frame, (10, 10), (300, 60), (0, 255, 0), -1)
                cv2.putText(frame, f"SAFE {confidence:.2f}", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Yield frame data directly (no VideoWriter needed for streaming)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        cap.release()
    
    return StreamingResponse(generate_video(), media_type="multipart/x-mixed-replace; boundary=frame")

