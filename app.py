import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from datetime import datetime
import json
import threading
import queue
from pathlib import Path
from twilio.rest import Client
from dotenv import load_dotenv
import re
import boto3

# -------------------------------
# Paths & Config
# -------------------------------
BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "logo.png"
CSS_PATH = BASE_DIR / "styles.css"
S3_BUCKET = "violence-detector-bucket"
S3_KEY = "logs/violence_detection_log.json"

st.set_page_config(page_title="NightShield", page_icon=str(LOGO_PATH), layout="wide")

with open(CSS_PATH) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.image(str(LOGO_PATH), use_container_width=True)
st.sidebar.markdown("## NightShield")

header_col1, header_col2 = st.columns([1, 8])
with header_col1:
    st.image(str(LOGO_PATH), width=80)
with header_col2:
    st.markdown("<div class='ns-header'><h1>NightShield</h1></div>", unsafe_allow_html=True)

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_FROM_NUMBER")

client = None
if TWILIO_SID and TWILIO_AUTH:
    client = Client(TWILIO_SID, TWILIO_AUTH)

# -------------------------------
# AWS S3 client
# -------------------------------
s3 = boto3.client("s3")

def load_logs_from_s3():
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except s3.exceptions.NoSuchKey:
        return []
    except Exception:
        return []

def save_logs_to_s3(data):
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=S3_KEY,
            Body=json.dumps(data, indent=4).encode("utf-8"),
            ContentType="application/json"
        )
    except Exception as e:
        print(f"S3 save error: {e}")

# -------------------------------
# Detection State (for SMS control)
# -------------------------------
DETECTION_COUNT = {"violence_detected": 0, "fall_detected": 0}
ALERT_SENT = {"violence_detected": False, "fall_detected": False}
NO_DETECTION_COUNT = {"violence_detected": 0, "fall_detected": 0}
FRAME_THRESHOLD = 10  # frames before triggering/resetting

# -------------------------------
# Auto-delete logs older than 24h
# -------------------------------
def auto_delete_old_logs():
    data = load_logs_from_s3()
    now = datetime.now()
    new_data = []
    for entry in data:
        try:
            entry_time = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
            if (now - entry_time).total_seconds() < 86400:  # 24h
                new_data.append(entry)
        except Exception:
            new_data.append(entry)
    if len(new_data) != len(data):
        save_logs_to_s3(new_data)

# -------------------------------
# Log incidents + Twilio alert
# -------------------------------
def log_incident(timestamp, source, event_type="violence_detected", confidence=None, alert_number=None):
    global DETECTION_COUNT, ALERT_SENT, NO_DETECTION_COUNT

    # Increment detection count
    DETECTION_COUNT[event_type] += 1
    NO_DETECTION_COUNT[event_type] = 0

    conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
    event_msg = (
        f"🚨 Violence detected at {timestamp} (source: {source}, confidence={conf_str})"
        if event_type == "violence_detected"
        else f"⚠️ Fall detected at {timestamp} (source: {source}, confidence={conf_str})"
    )

    log_entry = {
        "timestamp": timestamp,
        "source": source,
        "event": event_type,
    }
    if confidence is not None:
        log_entry["confidence"] = confidence

    data = load_logs_from_s3()
    data.append(log_entry)
    save_logs_to_s3(data)

    # 🔔 SMS only once per FRAME_THRESHOLD detections
    if client and alert_number and not ALERT_SENT[event_type] and DETECTION_COUNT[event_type] >= FRAME_THRESHOLD:
        try:
            client.messages.create(
                body=event_msg,
                from_=TWILIO_PHONE_NUMBER,
                to=alert_number
            )
            ALERT_SENT[event_type] = True
            print(f"✅ SMS sent: {event_msg}")
        except Exception as e:
            print(f"Twilio error: {e}")

# Run cleanup on startup
auto_delete_old_logs()

# -------------------------------
# Load YOLO model
# -------------------------------
model = YOLO("violence_detection_v4.pt")

# -------------------------------
# Phone validation helper
# -------------------------------
def is_valid_phone(number: str) -> bool:
    pattern = re.compile(r"^\+[1-9]\d{7,14}$")  # E.164 format
    return bool(pattern.match(number))

# -------------------------------
# UI Tabs
# -------------------------------
tab1, tab2 = st.tabs(["Detection", "Incidents"])

with tab1:
    st.subheader("🎥 Violence Detection with YOLOv8")
    st.write("Upload an image, video, or connect an RTSP stream for detection.")

    # 🔑 Phone number for alerts
    st.sidebar.subheader("Alert Settings")
    alert_number = st.sidebar.text_input("Enter phone number (E.164 format, e.g. +91xxxxxxxxxx)", key="alert_number")

    valid_number = is_valid_phone(alert_number) if alert_number else False
    if not valid_number:
        st.warning("⚠️ Please enter a valid phone number in E.164 format.")

    st.sidebar.subheader("RTSP Controls")
    rtsp_url = st.sidebar.text_input("RTSP stream URL", key="rtsp_url", value="")
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        0.0, 1.0, 0.5, 0.01, key="conf_slider",
    )

    # RTSP helpers
    rtsp_running = st.session_state.get("rtsp_running", False)

    def stop_rtsp():
        st.session_state["rtsp_running"] = False

    def run_rtsp_stream(rtsp_url, confidence_threshold, frame_queue, stop_event, alert_number):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            frame_queue.put((None, f"❌ Failed to open RTSP stream: {rtsp_url}"))
            return
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25

        while not stop_event.is_set() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                frame_queue.put((None, "⚠️ Failed to read frame. Stream ended."))
                break

            results = model(frame)
            detection_made = False

            for r in results:
                annotated = r.plot()
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        try:
                            conf = float(box.conf[0])
                        except Exception:
                            conf = None
                        if conf and conf >= confidence_threshold:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            log_incident(
                                timestamp,
                                f"RTSP frame {frame_idx}, approx {frame_idx / int(fps)}s",
                                confidence=conf,
                                event_type="violence_detected",
                                alert_number=alert_number
                            )
                            detection_made = True
                frame_queue.put((annotated, None))

            # No detection reset
            if not detection_made:
                for key in NO_DETECTION_COUNT:
                    NO_DETECTION_COUNT[key] += 1
                    if NO_DETECTION_COUNT[key] >= FRAME_THRESHOLD:
                        DETECTION_COUNT[key] = 0
                        ALERT_SENT[key] = False
                        NO_DETECTION_COUNT[key] = 0

            frame_idx += 1

        cap.release()
        frame_queue.put((None, None))

    # Start/Stop buttons
    col1, col2 = st.sidebar.columns(2)
    if "rtsp_stop_event" not in st.session_state:
        st.session_state["rtsp_stop_event"] = threading.Event()
    if "rtsp_frame_queue" not in st.session_state:
        st.session_state["rtsp_frame_queue"] = queue.Queue(maxsize=2)
    with col1:
        if st.button("Start RTSP Stream"):
            if not valid_number:
                st.error("⚠️ Enter valid phone number before detection.")
            elif not st.session_state.get("rtsp_running", False) and rtsp_url:
                st.session_state["rtsp_running"] = True
                st.session_state["rtsp_stop_event"].clear()
                st.session_state["rtsp_frame_queue"] = queue.Queue(maxsize=2)
                thread = threading.Thread(
                    target=run_rtsp_stream,
                    args=(rtsp_url, confidence_threshold, st.session_state["rtsp_frame_queue"], st.session_state["rtsp_stop_event"], alert_number),
                    daemon=True,
                )
                with st.spinner("Connecting to stream..."):
                    thread.start()
                st.session_state["rtsp_thread"] = thread
    with col2:
        if st.button("Stop RTSP Stream"):
            stop_rtsp()
            st.session_state["rtsp_stop_event"].set()

    # Display RTSP stream
    if st.session_state.get("rtsp_running", False):
        stframe = st.empty()
        while st.session_state.get("rtsp_running", False):
            try:
                annotated, error_msg = st.session_state["rtsp_frame_queue"].get(timeout=1)
            except queue.Empty:
                continue
            if error_msg:
                st.error(error_msg)
                st.session_state["rtsp_running"] = False
                break
            if annotated is not None:
                stframe.image(annotated, channels="BGR", use_container_width=True)
            else:
                st.session_state["rtsp_running"] = False
                break

    # File Upload
    uploaded_file = st.file_uploader(
        "Upload Image/Video",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        key="file_uploader_detection",
    )
    if uploaded_file and valid_number:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        if uploaded_file.type.startswith("image"):
            img = cv2.imread(tfile.name)
            results = model(img)
            detection_made = False
            for r in results:
                annotated = r.plot()
                st.image(annotated, channels="BGR")
                if r.boxes:
                    for box in r.boxes:
                        conf = float(box.conf[0]) if hasattr(box, "conf") else None
                        if conf and conf >= confidence_threshold:
                            log_incident(
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                uploaded_file.name,
                                confidence=conf,
                                event_type="violence_detected",
                                alert_number=alert_number
                            )
                            detection_made = True
            if not detection_made:
                for key in NO_DETECTION_COUNT:
                    NO_DETECTION_COUNT[key] += 1
                    if NO_DETECTION_COUNT[key] >= FRAME_THRESHOLD:
                        DETECTION_COUNT[key] = 0
                        ALERT_SENT[key] = False
                        NO_DETECTION_COUNT[key] = 0
        elif uploaded_file.type.startswith("video"):
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            frame_idx = 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                detection_made = False
                for r in results:
                    annotated = r.plot()
                    if r.boxes:
                        for box in r.boxes:
                            conf = float(box.conf[0]) if hasattr(box, "conf") else None
                            if conf and conf >= confidence_threshold:
                                log_incident(
                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    f"{uploaded_file.name} (frame {frame_idx}, ~{frame_idx/int(fps)}s)",
                                    confidence=conf,
                                    event_type="violence_detected",
                                    alert_number=alert_number
                                )
                                detection_made = True
                    st.image(annotated, channels="BGR", use_container_width=True)
                if not detection_made:
                    for key in NO_DETECTION_COUNT:
                        NO_DETECTION_COUNT[key] += 1
                        if NO_DETECTION_COUNT[key] >= FRAME_THRESHOLD:
                            DETECTION_COUNT[key] = 0
                            ALERT_SENT[key] = False
                            NO_DETECTION_COUNT[key] = 0
                frame_idx += 1
            cap.release()

with tab2:
    st.subheader("Incident Table (Last 24h)")
    incidents = load_logs_from_s3()
    incidents = sorted(incidents, key=lambda x: x["timestamp"], reverse=True)
    if incidents:
        st.dataframe(incidents)
    else:
        st.info("No incidents detected in the last 24 hours.")
