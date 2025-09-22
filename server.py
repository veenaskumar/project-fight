# server.py

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
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True) # Ensure upload directory exists

S3_BUCKET = "violence-detector-bucket"
S3_KEY = "logs/violence_detection_log.json"

# Twilio
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
twilio_client = Client(TWILIO_SID, TWILIO_AUTH) if TWILIO_SID and TWILIO_AUTH else None

# AWS S3
s3 = boto3.client(
    "s3",
    region_name="eu-west-2",
    config=botocore.client.Config(signature_version="s3v4")
)

ID2CLASS = {0: "violence", 1: "nonviolence", 2: "fall"}
def get_class_name(cls_id: int) -> str:
    return ID2CLASS.get(cls_id, f"class_{cls_id}")

# YOLO model
model = YOLO("violence_detection_v4.pt")

# FastAPI
app = FastAPI()

# -------------------------------
# Globals
# -------------------------------
STREAMS = {}      # stream_id -> metadata
CLIENTS = {}      # stream_id -> set(WebSocket)
FRAME_QUEUES = {} # stream_id -> queue.Queue (thread-safe, for annotated frames)
ALERT_COOLDOWN_SECONDS = 60 # Cooldown period in seconds to prevent alert spam

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
    except Exception as e:
        print(f"Presigned URL generation failed for {key}: {e}", flush=True)
        return None


def log_incident(stream_name, confidence, clip_path=None, snapshot_key=None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"timestamp": ts, "stream": stream_name, "confidence": confidence}

    if clip_path:
        if clip_path.startswith("http"):
            clip_path = clip_path.split(".amazonaws.com/")[-1].split("?")[0]
        entry["clip"] = clip_path

    if snapshot_key:
        if snapshot_key.startswith("http"):
            snapshot_key = snapshot_key.split(".amazonaws.com/")[-1].split("?")[0]
        entry["snapshot"] = snapshot_key

    logs = load_logs_from_s3()
    logs.append(entry)
    save_logs_to_s3(logs)


def send_sms_alert(phone, message):
    if twilio_client and is_valid_phone(phone):
        try:
            twilio_client.messages.create(
                body=message,
                from_=TWILIO_PHONE_NUMBER,
                to=phone
            )
            print(f"SMS alert sent to {phone}")
        except Exception as e:
            print(f"Failed to send SMS to {phone}: {e}")

# -------------------------------
# Detection loop
# -------------------------------
def detection_loop(stream_id):
    stream = STREAMS[stream_id]
    
    video_source = stream["url"]
    if stream.get("is_demo", False) and not video_source.startswith(('http://', 'https://', 'rtsp://')):
        video_source = str(UPLOAD_DIR / Path(video_source).name)
        print(f"DEBUG: Using absolute path for local file: {video_source}")
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {video_source}")
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
        FRAME_QUEUES[stream_id] = queue.Queue(maxsize=30) # Prevent queue from growing indefinitely

    print(f"DEBUG: Detection loop started for stream {stream_id}")

    while stream.get("running", False):
        ret, frame = cap.read()
        if not ret:
            print(f"DEBUG: End of video stream {stream_id}. Stopping loop.")
            break

        # YOLO detection
        try:
            results = model(frame, verbose=False)[0]
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
                        color = (0, 0, 255) if "violence" in cls_name or "fall" in cls_name else (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls_name.upper()} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception as e:
            print(f"Error during model inference: {e}")
            confidence = 0.0
            detected_classes = []

        is_violence_or_fall = "violence" in detected_classes or "fall" in detected_classes
        violence_detected = is_violence_or_fall and confidence >= stream["threshold"]

        # Create annotated frame
        display_frame = frame.copy()
        
        if violence_detected:
            cv2.rectangle(display_frame, (10, 10), (400, 80), (0, 0, 255), -1)
            cv2.putText(display_frame, f"ALERT! {confidence:.2f}", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        else:
            cv2.rectangle(display_frame, (10, 10), (300, 60), (0, 255, 0), -1)
            cv2.putText(display_frame, f"SAFE {confidence:.2f}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display_frame, timestamp, (10, display_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Push frame to queue for WebSocket and MJPEG streamers
        if stream_id in FRAME_QUEUES:
            if FRAME_QUEUES[stream_id].full():
                FRAME_QUEUES[stream_id].get_nowait() # Discard oldest frame if queue is full
            FRAME_QUEUES[stream_id].put(display_frame)
        
        # Clip recording and alert logic
        frame_buffer.append(frame.copy())
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)

        if violence_detected:
            consecutive_count += 1
        else:
            consecutive_count = 0

        # ✨ MODIFICATION: Alert and Cooldown Logic
        time_since_last_alert = time.time() - stream.get("last_alert_time", 0)
        
        if consecutive_count >= alert_trigger and not recording and time_since_last_alert > ALERT_COOLDOWN_SECONDS:
            print(f"Triggering alert for stream {stream_id}")
            stream["last_alert_time"] = time.time() # Update last alert time
            
            tmp_clip = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            clip_path = tmp_clip.name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(clip_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
            for bf in frame_buffer:
                out.write(bf)
            recording = True

            snapshot_path = f"{stream_id}_{int(time.time())}.jpg"
            cv2.imwrite(snapshot_path, frame)
            s3_key_snapshot = f"snapshots/{Path(snapshot_path).name}"
            try:
                s3.upload_file(snapshot_path, S3_BUCKET, s3_key_snapshot, ExtraArgs={"ContentType": "image/jpeg"})
            finally:
                if os.path.exists(snapshot_path): os.remove(snapshot_path)

            send_sms_alert(
                stream["phone"],
                f"⚠️ Alert on {stream['name']} (Confidence: {confidence:.2f}). Check dashboard for details."
            )
            
            # This part runs only ONCE per event due to the cooldown
            if out:
                out.release()
                recording = False
                s3_key_clip = f"clips/{Path(clip_path).name}"
                try:
                    s3.upload_file(clip_path, S3_BUCKET, s3_key_clip, ExtraArgs={"ContentType": "video/mp4"})
                    log_incident(stream["name"], confidence, clip_path=s3_key_clip, snapshot_key=s3_key_snapshot)
                    print(f"Successfully uploaded clip {s3_key_clip}")
                except Exception as e:
                    print(f"Failed to upload clip {clip_path}: {e}")
                finally:
                    if os.path.exists(clip_path): os.remove(clip_path)
                frame_buffer = [] # Clear buffer after saving

    # Cleanup
    cap.release()
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
            if ws.client_state.name != "CONNECTED": break
            
            if stream_id in FRAME_QUEUES and not FRAME_QUEUES[stream_id].empty():
                try:
                    frame = FRAME_QUEUES[stream_id].get_nowait()
                    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    jpg_str = base64.b64encode(buffer).decode()
                    await ws.send_text(jpg_str)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                except Exception:
                    break
            else:
                await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for stream {stream_id}")
    finally:
        if stream_id in CLIENTS:
            CLIENTS[stream_id].discard(ws)

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
        "is_demo": file_uploaded,
        "last_alert_time": 0 # Initialize alert timestamp
    }
    
    t = threading.Thread(target=detection_loop, args=(stream_id,), daemon=True)
    t.start()
    return {"stream_id": stream_id, "status": "started"}

@app.get("/active_streams")
def get_active_streams():
    # Return a copy to avoid race conditions if dict is modified during iteration
    streams_copy = list(STREAMS.values())
    return [
        {
            "stream_id": s["stream_id"],
            "name": s["name"],
            "is_demo": s.get("is_demo", False),
            "running": s.get("running", False),
            "phone": s.get("phone", ""),
            "threshold": s.get("threshold", 0.5)
        }
        for s in streams_copy
    ]

@app.post("/stop_stream/{stream_id}")
def stop_stream(stream_id: str):
    if stream_id not in STREAMS:
        return {"error": "Stream not found"}
    STREAMS[stream_id]["running"] = False
    return {"message": f"Stream {stream_id} stopping..."}

@app.post("/start_stream/{stream_id}")
def start_stream(stream_id: str):
    if stream_id not in STREAMS:
        return {"error": "Stream not found"}
    if STREAMS[stream_id].get("running", False):
        return {"message": "Stream is already running"}
        
    STREAMS[stream_id]["running"] = True
    t = threading.Thread(target=detection_loop, args=(stream_id,), daemon=True)
    t.start()
    return {"message": f"Stream {stream_id} started"}

@app.delete("/delete_stream/{stream_id}")
def delete_stream(stream_id: str):
    if stream_id not in STREAMS:
        return {"error": "Stream not found"}
    
    STREAMS[stream_id]["running"] = False
    time.sleep(1) # Give the loop a moment to stop
    
    if stream_id in FRAME_QUEUES: del FRAME_QUEUES[stream_id]
    if stream_id in CLIENTS: del CLIENTS[stream_id]
    if stream_id in STREAMS: del STREAMS[stream_id]
    
    return {"message": f"Stream {stream_id} deleted"}

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
    def generate_video():
        if stream_id not in STREAMS:
            print(f"Stream {stream_id} not found for MJPEG stream.")
            return

        while STREAMS.get(stream_id) and STREAMS[stream_id].get("running", False):
            try:
                frame = FRAME_QUEUES[stream_id].get(timeout=10) # Wait up to 10s for a new frame
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except queue.Empty:
                print(f"No frame received for stream {stream_id} in 10 seconds. Ending MJPEG stream.")
                break # End stream if no frame for 10s
            except Exception:
                # Stream was likely deleted
                break
    
    return StreamingResponse(generate_video(), media_type="multipart/x-mixed-replace; boundary=frame")