# index.py
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
import traceback
import shutil

# -------------------------------
# Basic config
# -------------------------------
BASE_DIR = Path(__file__).parent
CLIPS_LOCAL_DIR = BASE_DIR / "clips"
LOGS_LOCAL_PATH = BASE_DIR / "logs_local.json"

# Load .env
load_dotenv()

# S3 / TWILIO config from env
S3_BUCKET = os.getenv("S3_BUCKET", "violence-detector-bucket")
LOGS_KEY = os.getenv("LOGS_KEY", "logs/violence_detection_log.json")
CLIPS_PREFIX = os.getenv("CLIPS_PREFIX", "clips/")

TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM_NUMBER")

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "violence_detection_v4.pt")

# UI config
st.set_page_config(page_title="NightShield", layout="wide")

# Ensure local dirs
CLIPS_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Initialize clients (might be None)
# -------------------------------
# Twilio
twilio_client = None
if TWILIO_SID and TWILIO_AUTH:
    try:
        twilio_client = Client(TWILIO_SID, TWILIO_AUTH)
    except Exception as e:
        print("Twilio init error:", e)

# boto3 S3 client
s3 = boto3.client("s3")
AWS_AVAILABLE = True
try:
    # quick check — list buckets (lightweight)
    s3.list_buckets()
except Exception as e:
    AWS_AVAILABLE = False
    print("AWS not available or credentials missing:", e)

# -------------------------------
# Helpers: local logs & S3 fallback
# -------------------------------
_logs_lock = threading.Lock()

def load_logs_s3():
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=LOGS_KEY)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        return []
    except Exception as e:
        print("load_logs_from_s3 error:", e)
        return []

def save_logs_s3(data):
    try:
        s3.put_object(Bucket=S3_BUCKET, Key=LOGS_KEY,
                      Body=json.dumps(data, indent=2).encode("utf-8"),
                      ContentType="application/json")
        return True
    except Exception as e:
        print("save_logs_to_s3 error:", e)
        return False

def load_logs_local():
    if not LOGS_LOCAL_PATH.exists():
        return []
    try:
        with open(LOGS_LOCAL_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("load local logs error:", e)
        return []

def save_logs_local(data):
    try:
        with open(LOGS_LOCAL_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print("save local logs error:", e)
        return False

def append_log(entry):
    """Thread-safe append log; uses S3 if available else local file"""
    with _logs_lock:
        if AWS_AVAILABLE:
            data = load_logs_s3()
            data.append(entry)
            success = save_logs_s3(data)
            if not success:
                # fallback to local
                print("S3 save failed; falling back to local logs.")
                data_local = load_logs_local()
                data_local.append(entry)
                save_logs_local(data_local)
        else:
            data_local = load_logs_local()
            data_local.append(entry)
            save_logs_local(data_local)

def load_logs():
    if AWS_AVAILABLE:
        try:
            return load_logs_s3()
        except Exception:
            return load_logs_local()
    else:
        return load_logs_local()

def presign_s3_url(key: str, expires_in: int = 86400):
    if not AWS_AVAILABLE:
        return None
    try:
        return s3.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=expires_in)
    except Exception as e:
        print("presign error:", e)
        return None

# -------------------------------
# Clips upload (S3 or local)
# -------------------------------
def upload_clip(local_path, stream_id):
    """Upload clip; return dict with keys: uploaded(bool), s3_key or local_path"""
    if AWS_AVAILABLE:
        key = f"{CLIPS_PREFIX}{stream_id}/{Path(local_path).name}"
        try:
            s3.upload_file(Filename=local_path, Bucket=S3_BUCKET, Key=key)
            return {"uploaded": True, "s3_key": key}
        except Exception as e:
            print("S3 upload failed:", e)
            # still keep local copy
            dest = CLIPS_LOCAL_DIR / f"{stream_id}_{Path(local_path).name}"
            try:
                shutil.move(local_path, dest)
                return {"uploaded": False, "local_path": str(dest)}
            except Exception:
                return {"uploaded": False, "local_path": local_path}
    else:
        # keep local copy (move)
        dest = CLIPS_LOCAL_DIR / f"{stream_id}_{Path(local_path).name}"
        try:
            shutil.move(local_path, dest)
            return {"uploaded": False, "local_path": str(dest)}
        except Exception as e:
            print("Local move failed:", e)
            return {"uploaded": False, "local_path": local_path}

# -------------------------------
# Phone validation
# -------------------------------
def is_valid_phone(number: str) -> bool:
    if not number:
        return False
    pattern = re.compile(r"^\+[1-9]\d{7,14}$")
    return bool(pattern.match(number))

# -------------------------------
# Load YOLO model
# -------------------------------
try:
    model = YOLO(YOLO_MODEL_PATH)
    model_names = getattr(model, "names", None) or {}
except Exception as e:
    st.error(f"Failed loading YOLO model: {e}")
    model = None
    model_names = {}

# -------------------------------
# Session-state init
# -------------------------------
if "streams" not in st.session_state:
    st.session_state["streams"] = {}  # stream_id -> metadata
if "max_streams" not in st.session_state:
    st.session_state["max_streams"] = 4

# -------------------------------
# Worker: per-stream
# -------------------------------
def run_stream_worker(stream_id: str):
    """
    Worker reads frames, runs model, records clips, uploads, and appends logs.
    Stores frames into frame_queue as (overlay_frame, label_str)
    """
    try:
        info = st.session_state["streams"].get(stream_id)
        if not info:
            return
    except Exception:
        return

    source = info["source"]
    mode = info["mode"]
    threshold = float(info.get("threshold", 0.5))
    phone = info.get("phone_number")
    stop_event = info["stop_event"]
    frame_q = info["frame_queue"]
    pre_seconds = int(info.get("pre_seconds", 5))
    post_seconds = int(info.get("post_seconds", 5))
    alert_frame_threshold = int(info.get("alert_frame_threshold", 3))

    # Open capture
    if mode == "rtsp":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        try:
            frame_q.put((None, "__error__", f"Failed to open source: {source}"))
        except Exception:
            pass
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    pre_buffer = deque(maxlen=max(1, int(pre_seconds * fps)))

    recording = False
    video_writer = None
    post_counter = 0
    detection_count = 0
    alert_sent = False
    best_conf_violence = 0.0
    best_conf_fall = 0.0

    while not stop_event.is_set() and cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                # Try briefly to reconnect for RTSP, else break
                frame_q.put((None, "__error__", "Stream ended or cannot read frame"))
                break

            pre_buffer.append(frame.copy())

            # run model if available
            detection_made = False
            labels_detected = []  # collect label names and confidences
            if model:
                try:
                    results = model(frame)
                except Exception as e:
                    print(f"[{stream_id}] model error: {e}")
                    results = None
            else:
                results = None

            if results:
                for r in results:
                    # r.boxes may exist
                    if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            try:
                                conf = float(box.conf[0])
                            except Exception:
                                conf = 0.0
                            # try to get class index/name
                            cls_idx = None
                            cls_name = None
                            try:
                                # ultralytics boxes may have .cls
                                cls_idx = int(box.cls[0]) if hasattr(box, "cls") else None
                            except Exception:
                                cls_idx = None
                            if cls_idx is not None and model_names:
                                cls_name = model_names.get(cls_idx, str(cls_idx))
                            # detect violence / fall by class name
                            if cls_name:
                                labels_detected.append((cls_name, conf))
                                if cls_name.lower() in ("violence", "fight", "fighting", "violent"):
                                    if conf >= threshold:
                                        detection_made = True
                                        best_conf_violence = max(best_conf_violence, conf)
                                if cls_name.lower() in ("fall", "fallen", "person_fall"):
                                    if conf >= threshold:
                                        detection_made = True
                                        best_conf_fall = max(best_conf_fall, conf)
                            else:
                                # If no names mapping, treat any high confidence box as a violence candidate
                                if conf >= threshold:
                                    detection_made = True
                                    best_conf_violence = max(best_conf_violence, conf)

            # If detection made -> start/update recording
            if detection_made:
                detection_count += 1
                post_counter = int(post_seconds * fps)
                if not recording:
                    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tmpf.close()
                    clip_path = tmpf.name
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(clip_path, fourcc, fps, (w, h))
                    # write pre-buffer into writer
                    for pf in pre_buffer:
                        video_writer.write(pf)
                    info["current_clip_path"] = clip_path
                    recording = True
                if video_writer:
                    video_writer.write(frame)

                # Send alert once per event threshold
                if (twilio_client is not None) and phone and (not alert_sent) and detection_count >= alert_frame_threshold:
                    # Choose event type & confidence to report
                    event_type = "violence_detected" if best_conf_violence >= best_conf_fall else "fall_detected"
                    conf_to_report = best_conf_violence if event_type == "violence_detected" else best_conf_fall
                    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                    body = f"🚨 {info.get('name')} — {event_type} at {ts} (conf {conf_to_report:.2f})"
                    try:
                        twilio_client.messages.create(body=body, from_=TWILIO_FROM, to=phone)
                        alert_sent = True
                        print(f"[{stream_id}] Alert sent to {phone}: {body}")
                    except Exception as e:
                        print(f"[{stream_id}] Twilio error: {e}")

            else:
                detection_count = 0
                if recording:
                    if post_counter > 0:
                        if video_writer:
                            video_writer.write(frame)
                        post_counter -= 1
                    else:
                        # finish clip
                        if video_writer:
                            video_writer.release()
                        recording = False
                        clip_local = info.pop("current_clip_path", None)
                        if clip_local and os.path.exists(clip_local):
                            upload_res = upload_clip(clip_local, stream_id)
                            clip_url = None
                            clip_s3_key = None
                            if upload_res.get("uploaded"):
                                clip_s3_key = upload_res.get("s3_key")
                                clip_url = presign_s3_url(clip_s3_key) if clip_s3_key else None
                            else:
                                clip_url = upload_res.get("local_path")
                                clip_s3_key = None

                            # choose event type to log (prefer violence if both)
                            event_type = "violence_detected" if best_conf_violence >= best_conf_fall else "fall_detected"
                            conf_val = float(best_conf_violence if event_type == "violence_detected" else best_conf_fall) if (best_conf_violence or best_conf_fall) else None

                            entry = {
                                "id": str(uuid.uuid4()),
                                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                                "stream_id": stream_id,
                                "stream_name": info.get("name") or stream_id,
                                "event": event_type,
                                "confidence": conf_val,
                                "clip_s3_key": clip_s3_key,
                                "clip_local_path": clip_url if not clip_s3_key else None
                            }
                            append_log(entry)
                            # remove local temp if S3 uploaded (if not needed)
                            try:
                                if upload_res.get("uploaded") and os.path.exists(clip_local):
                                    os.remove(clip_local)
                            except Exception:
                                pass
                            # reset alert flag so future events can trigger
                            alert_sent = False
                            best_conf_violence = 0.0
                            best_conf_fall = 0.0

            # create overlay for UI
            overlay = frame.copy()
            label_texts = []
            if best_conf_violence > 0:
                label_texts.append(f"Violence {best_conf_violence:.2f}")
            if best_conf_fall > 0:
                label_texts.append(f"Fall {best_conf_fall:.2f}")
            if label_texts:
                cv2.putText(overlay, " | ".join(label_texts), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # push to frame queue (keep simple tuple)
            try:
                if not frame_q.full():
                    frame_q.put((overlay, None))
            except Exception:
                pass

        except Exception as e:
            print(f"[{stream_id}] worker exception: {e}")
            traceback.print_exc()
            try:
                frame_q.put((None, "__error__", str(e)))
            except Exception:
                pass
            break

    try:
        if video_writer:
            video_writer.release()
    except Exception:
        pass
    try:
        cap.release()
    except Exception:
        pass

    # mark stopped
    try:
        info["running"] = False
        frame_q.put((None, "__stopped__"))
    except Exception:
        pass

# -------------------------------
# Stream management helpers
# -------------------------------
def add_stream(name, source, mode, threshold, phone, pre_seconds=5, post_seconds=5, alert_frame_threshold=3):
    sid = f"stream_{len(st.session_state['streams'])+1}"
    if len(st.session_state["streams"]) >= st.session_state["max_streams"]:
        return None, "max_reached"
    frame_q = queue.Queue(maxsize=6)
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
        "thread": None,
        "running": False,
        "pre_seconds": pre_seconds,
        "post_seconds": post_seconds,
        "alert_frame_threshold": alert_frame_threshold,
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }
    st.session_state["streams"][sid] = info
    return sid, None

def start_stream(sid):
    info = st.session_state["streams"].get(sid)
    if not info:
        return "not_found"
    if info.get("running"):
        return "already_running"
    t = threading.Thread(target=run_stream_worker, args=(sid,), daemon=True)
    info["thread"] = t
    info["running"] = True
    info["stop_event"].clear()
    t.start()
    return "started"

def stop_stream(sid):
    info = st.session_state["streams"].get(sid)
    if not info:
        return "not_found"
    info["stop_event"].set()
    info["running"] = False
    return "stopped"

def delete_stream(sid):
    stop_stream(sid)
    st.session_state["streams"].pop(sid, None)

# -------------------------------
# UI tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["Stream Management", "Live Preview", "Detection Clips"])

# Management tab
with tab1:
    st.header("Stream Management Dashboard")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Add / Edit Stream")
        with st.form("add_stream_form", clear_on_submit=False):
            name = st.text_input("Stream name (friendly)", value="New Stream")
            mode = st.selectbox("Mode", options=["rtsp", "demo"], format_func=lambda x: "RTSP/HTTP" if x=="rtsp" else "Demo (upload file)")
            source = st.text_input("RTSP/HTTP URL (or local file path when in demo)", value="")
            if mode == "demo":
                uploaded_demo = st.file_uploader("Upload MP4 for Demo (optional)", type=["mp4"], key="demo_upload")
                if uploaded_demo:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tmp.write(uploaded_demo.read())
                    tmp.close()
                    source = tmp.name
                    st.info("Uploaded demo clip saved locally.")
            threshold = st.slider("Detection confidence threshold", 0.0, 1.0, 0.5, 0.01)
            phone = st.text_input("Alert phone number (E.164, optional)", value="")
            pre_seconds = st.number_input("Pre-event seconds (clip)", min_value=0, max_value=10, value=5)
            post_seconds = st.number_input("Post-event seconds (clip)", min_value=0, max_value=10, value=5)
            submitted = st.form_submit_button("Add Stream")
            if submitted:
                if mode == "rtsp" and not source:
                    st.error("Please provide RTSP/HTTP URL.")
                else:
                    if phone and not is_valid_phone(phone):
                        st.error("Phone number must be in E.164 format (e.g., +911234567890).")
                    else:
                        sid, err = add_stream(name=name, source=source, mode=mode, threshold=threshold, phone=phone, pre_seconds=pre_seconds, post_seconds=post_seconds)
                        if sid:
                            st.success(f"Stream added: {sid}")
                        else:
                            if err == "max_reached":
                                st.error("Maximum streams reached.")
                            else:
                                st.error("Failed to add stream.")

    with c2:
        st.subheader("Active Streams")
        for sid, info in st.session_state["streams"].items():
            cols = st.columns([1, 3, 2, 2, 2])
            with cols[0]:
                st.write(f"**{sid}**")
            with cols[1]:
                st.write(f"{info['name']}")
            with cols[2]:
                status = "✅ Monitoring" if info.get("running") else "⏸ Not monitoring"
                st.write(status)
            with cols[3]:
                st.write(f"Threshold: {info['threshold']:.2f}")
                st.write(f"Phone: {info.get('phone_number') or '—'}")
            with cols[4]:
                if st.button("Start", key=f"start_{sid}"):
                    res = start_stream(sid)
                    if res == "started":
                        st.success(f"Started {sid}")
                    elif res == "already_running":
                        st.info("Already running")
                if st.button("Stop", key=f"stop_{sid}"):
                    stop_stream(sid)
                    st.warning(f"Stopped {sid}")
                if st.button("Delete", key=f"del_{sid}"):
                    delete_stream(sid)
                    st.error(f"Deleted {sid}")

# Live preview tab
with tab2:
    st.header("Live Preview")
    st_autorefresh(interval=1000, key="lp_refresh")
    active_streams = list(st.session_state["streams"].items())
    if not active_streams:
        st.info("No streams configured. Add from Stream Management.")
    else:
        cols = st.columns(2)
        for i, (sid, info) in enumerate(active_streams):
            with cols[i % 2]:
                st.subheader(f"{info.get('name')}  —  {sid}")
                st.caption(f"Mode: {info.get('mode')} · Threshold: {info.get('threshold'):.2f} · Phone: {info.get('phone_number') or '—'}")
                # quick inline controls
                col_a, col_b, col_c = st.columns([1,1,1])
                with col_a:
                    new_thr = st.number_input(f"Adjust threshold ({sid})", min_value=0.0, max_value=1.0, value=float(info['threshold']), step=0.01, key=f"thr_{sid}")
                    if new_thr != info['threshold']:
                        info['threshold'] = float(new_thr)
                        st.rerun()
                with col_b:
                    new_phone = st.text_input(f"Recipient ({sid})", value=info.get('phone_number',''), key=f"phone_{sid}")
                    if new_phone != info.get('phone_number'):
                        if new_phone and not is_valid_phone(new_phone):
                            st.error("Phone must be E.164")
                        else:
                            info['phone_number'] = new_phone
                with col_c:
                    if info.get("running"):
                        if st.button("Stop", key=f"card_stop_{sid}"):
                            stop_stream(sid)
                            st.rerun()
                    else:
                        if st.button("Start", key=f"card_start_{sid}"):
                            start_stream(sid)
                            st.rerun()

                # display latest frame
                placeholder = st.empty()
                try:
                    if not info["frame_queue"].empty():
                        item = info["frame_queue"].get_nowait()
                        if item:
                            frame_img, _label = item
                            if frame_img is None:
                                placeholder.info("No frame")
                            else:
                                rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                                placeholder.image(rgb, use_column_width=True)
                        else:
                            placeholder.info("Waiting for frames...")
                    else:
                        placeholder.info("Waiting for frames...")
                except queue.Empty:
                    placeholder.info("Waiting for frames...")
                except Exception as e:
                    placeholder.info(f"Frame error: {e}")

# Detection clips tab
with tab3:
    st.header("Detection Clips (last 24h)")
    logs = load_logs()
    # filter only last 24h
    now = datetime.utcnow()
    recent = []
    for entry in logs:
        try:
            t = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
            if (now - t).total_seconds() <= 86400:
                recent.append(entry)
        except Exception:
            recent.append(entry)

    if not recent:
        st.info("No clips in the last 24 hours.")
    else:
        for ent in sorted(recent, key=lambda x: x.get("timestamp",""), reverse=True):
            st.markdown("---")
            name = ent.get("stream_name") or ent.get("stream_id") or ent.get("source") or "Unknown"
            st.write(f"**{name}** — {ent.get('timestamp')}")
            st.write(f"Event: {ent.get('event')}")
            if ent.get("confidence") is not None:
                st.write(f"Confidence: {ent.get('confidence'):.2f}")
            if ent.get("clip_s3_key"):
                url = presign_s3_url(ent.get("clip_s3_key"))
                if url:
                    st.video(url)
                else:
                    st.write("Clip exists but presign failed.")
            elif ent.get("clip_local_path"):
                try:
                    st.video(ent.get("clip_local_path"))
                except Exception:
                    st.write(f"Local clip: {ent.get('clip_local_path')}")
