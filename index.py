# app.py
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from datetime import datetime, timedelta
import json
import threading
import queue
from pathlib import Path
from twilio.rest import Client
from dotenv import load_dotenv
import re
import boto3
import uuid
from collections import deque
from streamlit_autorefresh import st_autorefresh

# -------------------------------
# Config & Paths
# -------------------------------
BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "logo.png"
CSS_PATH = BASE_DIR / "styles.css"

# S3 config (set env or edit here)
S3_BUCKET = os.getenv("S3_BUCKET", "violence-detector-bucket")
LOGS_KEY = "logs/violence_detection_log.json"
CLIPS_PREFIX = "clips/"

# UI page
st.set_page_config(page_title="NightShield", page_icon=str(LOGO_PATH), layout="wide")
if CSS_PATH.exists():
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), use_container_width=True)
    header_col1, header_col2 = st.columns([1, 8])
    with header_col1:
        st.image(str(LOGO_PATH), width=80)
    with header_col2:
        st.markdown("<div class='ns-header'><h1>NightShield</h1></div>", unsafe_allow_html=True)

# -------------------------------
# Environment & clients
# -------------------------------
load_dotenv()
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_FROM_NUMBER")

twilio_client = None
if TWILIO_SID and TWILIO_AUTH:
    twilio_client = Client(TWILIO_SID, TWILIO_AUTH)

# boto3 S3 client
s3 = boto3.client("s3")

# -------------------------------
# Helpers
# -------------------------------
_logs_lock = threading.Lock()

def load_logs_from_s3():
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=LOGS_KEY)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        return []
    except Exception as e:
        print("load_logs_from_s3 error:", e)
        return []

def save_logs_to_s3(data):
    try:
        s3.put_object(Bucket=S3_BUCKET, Key=LOGS_KEY,
                      Body=json.dumps(data, indent=2).encode("utf-8"),
                      ContentType="application/json")
    except Exception as e:
        print("save_logs_to_s3 error:", e)

def append_log(entry):
    with _logs_lock:
        logs = load_logs_from_s3()
        logs.append(entry)
        save_logs_to_s3(logs)

def is_valid_phone(number: str) -> bool:
    pattern = re.compile(r"^\+[1-9]\d{7,14}$")
    return bool(pattern.match(number))

def presign_s3_url(key: str, expires_in: int = 86400):
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=expires_in
        )
    except Exception as e:
        print("presign error:", e)
        return None

# -------------------------------
# Model
# -------------------------------
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "violence_detection_v4.pt")
model = YOLO(MODEL_PATH)

# -------------------------------
# Session State init
# -------------------------------
if "streams" not in st.session_state:
    st.session_state["streams"] = {}
if "max_streams" not in st.session_state:
    st.session_state["max_streams"] = 4

# -------------------------------
# Worker
# -------------------------------
def upload_clip_to_s3(local_path, s3_key):
    try:
        s3.upload_file(Filename=local_path, Bucket=S3_BUCKET, Key=s3_key)
        return True
    except Exception as e:
        print("Upload clip error:", e)
        return False

def run_stream_worker(stream_id: str):
    info = st.session_state["streams"].get(stream_id)
    if not info:
        return

    source = info["source"]
    threshold = info["threshold"]
    phone = info.get("phone_number")
    stop_event = info["stop_event"]
    frame_q = info["frame_queue"]
    max_pre_seconds = info.get("pre_seconds", 5)
    post_seconds = info.get("post_seconds", 5)

    # Open capture
    if info["mode"] == "rtsp":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        frame_q.put(("__error__", f"Failed to open source: {source}"))
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    pre_buffer = deque(maxlen=int(max_pre_seconds * fps))

    recording = False
    video_writer = None
    post_counter = 0
    detection_count = 0
    alert_sent = False
    best_conf = 0.0

    while not stop_event.is_set() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            frame_q.put(("__error__", "Stream ended"))
            break

        pre_buffer.append(frame.copy())
        results = model(frame)

        detection_made = False
        best_conf = 0.0
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf >= threshold:
                        detection_made = True
                        best_conf = max(best_conf, conf)

        if detection_made:
            detection_count += 1
            post_counter = int(post_seconds * fps)
            if not recording:
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_file.close()
                clip_path = tmp_file.name
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(clip_path, fourcc, fps, (w, h))
                for f in pre_buffer:
                    video_writer.write(f)
                info["current_clip_path"] = clip_path
                recording = True
            if video_writer:
                video_writer.write(frame)

            if twilio_client and phone and not alert_sent and detection_count >= info.get("alert_frame_threshold", 3):
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                body = f"🚨 {info['name']} — Violence detected at {ts} (confidence {best_conf:.2f})"
                try:
                    twilio_client.messages.create(body=body, from_=TWILIO_PHONE_NUMBER, to=phone)
                    alert_sent = True
                except Exception as e:
                    print(f"[{stream_id}] Twilio error:", e)
        else:
            detection_count = 0
            if recording:
                if post_counter > 0:
                    if video_writer:
                        video_writer.write(frame)
                    post_counter -= 1
                else:
                    if video_writer:
                        video_writer.release()
                    recording = False
                    clip_local = info.pop("current_clip_path", None)
                    if clip_local and os.path.exists(clip_local):
                        clip_filename = f"{stream_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.mp4"
                        s3_key = CLIPS_PREFIX + clip_filename
                        uploaded = upload_clip_to_s3(clip_local, s3_key)
                        clip_url = presign_s3_url(s3_key) if uploaded else None
                        entry = {
                            "id": str(uuid.uuid4()),
                            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                            "stream_id": stream_id,
                            "stream_name": info.get("name"),
                            "event": "violence_detected",
                            "confidence": float(best_conf),
                            "clip_s3_key": s3_key if uploaded else None,
                            "clip_url": clip_url
                        }
                        append_log(entry)
                        try:
                            os.remove(clip_local)
                        except:
                            pass
                        alert_sent = False

        overlay = frame.copy()
        if detection_made:
            cv2.putText(overlay, f"Violence ({best_conf:.2f})", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        if not frame_q.full():
            frame_q.put((overlay, None))

    if video_writer:
        video_writer.release()
    cap.release()
    info["running"] = False
    frame_q.put((None, "__stopped__"))

# -------------------------------
# Stream helpers
# -------------------------------
def add_stream(name, source, mode, threshold, phone, pre_seconds=5, post_seconds=5):
    sid = f"stream_{len(st.session_state['streams'])+1}"
    if len(st.session_state["streams"]) >= st.session_state["max_streams"]:
        return None, "max_reached"
    frame_q = queue.Queue(maxsize=5)
    stop_event = threading.Event()
    info = {
        "id": sid,
        "name": name,
        "source": source,
        "mode": mode,
        "threshold": float(threshold),
        "phone_number": phone,
        "frame_queue": frame_q,
        "stop_event": stop_event,
        "running": False,
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "pre_seconds": pre_seconds,
        "post_seconds": post_seconds,
        "alert_frame_threshold": 3
    }
    st.session_state["streams"][sid] = info
    return sid, None

def start_stream(sid):
    info = st.session_state["streams"].get(sid)
    if not info or info["running"]:
        return
    t = threading.Thread(target=run_stream_worker, args=(sid,), daemon=True)
    info["thread"] = t
    info["running"] = True
    info["stop_event"].clear()
    t.start()

def stop_stream(sid):
    info = st.session_state["streams"].get(sid)
    if not info:
        return
    info["stop_event"].set()
    info["running"] = False

def delete_stream(sid):
    stop_stream(sid)
    st.session_state["streams"].pop(sid, None)

# -------------------------------
# UI Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["Stream Management", "Live Preview", "Detection Clips"])

# ===== Management =====
with tab1:
    st.header("Stream Management Dashboard")
    with st.form("add_stream_form"):
        name = st.text_input("Stream name", "New Stream")
        mode = st.selectbox("Mode", ["rtsp", "demo"])
        source = st.text_input("RTSP/HTTP URL or local file")
        if mode=="demo":
            up = st.file_uploader("Upload MP4 demo", type=["mp4"])
            if up:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(up.read())
                tmp.close()
                source = tmp.name
        threshold = st.slider("Detection threshold",0.0,1.0,0.5,0.01)
        phone = st.text_input("Alert phone number", "")
        submitted = st.form_submit_button("Add Stream")
        if submitted:
            if phone and not is_valid_phone(phone):
                st.error("Phone must be in E.164 format")
            else:
                sid,_ = add_stream(name,source,mode,threshold,phone)
                if sid:
                    st.success(f"Added {sid}")

    st.subheader("Active Streams")
    for sid,info in st.session_state["streams"].items():
        cols=st.columns([1,2,2,2])
        with cols[0]: st.write(sid)
        with cols[1]: st.write(info["name"])
        with cols[2]: st.write("✅ Running" if info["running"] else "⏸ Stopped")
        with cols[3]:
            if st.button("Start",key=f"start_{sid}"): start_stream(sid)
            if st.button("Stop",key=f"stop_{sid}"): stop_stream(sid)
            if st.button("Delete",key=f"del_{sid}"): delete_stream(sid)

# ===== Live Preview =====
with tab2:
    st.header("Live Preview")
    st_autorefresh(interval=1000,key="refresh")
    if not st.session_state["streams"]:
        st.info("No streams configured")
    else:
        cols = st.columns(2)
        for i,(sid,info) in enumerate(st.session_state["streams"].items()):
            with cols[i%2]:
                st.subheader(info["name"])
                st.caption(f"{sid} · Thr {info['threshold']}")
                placeholder = st.empty()
                try:
                    if not info["frame_queue"].empty():
                        frame,label = info["frame_queue"].get_nowait()
                        if frame is None and label=="__stopped__":
                            placeholder.info("Stopped")
                        elif frame=="__error__":
                            placeholder.error(label)
                        else:
                            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                            placeholder.image(rgb,use_column_width=True)
                            if label: st.warning(label)
                except queue.Empty:
                    placeholder.info("Waiting for frames...")

# ===== Clips =====
with tab3:
    st.header("Detection Clips")
    logs = load_logs_from_s3()
    if not logs: st.info("No clips yet")
    for ent in sorted(logs,key=lambda x:x["timestamp"],reverse=True):
        st.markdown("---")
        st.write(f"**{ent['stream_name']}** — {ent['timestamp']}")
        if ent.get("confidence"):
            st.write(f"Conf {ent['confidence']:.2f}")
        if ent.get("clip_s3_key"):
            url=presign_s3_url(ent["clip_s3_key"])
            if url: st.video(url)
