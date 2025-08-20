
import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from datetime import datetime
import json
import threading
import time

LOG_FILE = "violence_detection_log.json"

# Auto-delete logs older than 24 hours
def auto_delete_old_logs():
    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    now = datetime.now()
    new_data = []
    for entry in data:
        try:
            entry_time = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")
            if (now - entry_time).total_seconds() < 86400:
                new_data.append(entry)
        except Exception:
            new_data.append(entry)
    if len(new_data) != len(data):
        with open(LOG_FILE, "w") as f:
            json.dump(new_data, f, indent=4)

auto_delete_old_logs()

def log_violence(timestamp, source, confidence=None):
    log_entry = {
        "timestamp": timestamp,
        "source": source,
        "event": "violence_detected"
    }
    if confidence is not None:
        log_entry["confidence"] = confidence
    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(log_entry)
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Load model
model = YOLO("violence-detection-through-cctv_step3.pt")


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Detection", "Incidents"])

if page == "Detection":
    st.title("🎥 Violence Detection with YOLOv8")
    st.write("Upload an image or video and the model will detect violence.")

    # RTSP input and controls in sidebar
    st.sidebar.subheader("RTSP Controls")
    rtsp_url = st.sidebar.text_input("RTSP stream URL", key="rtsp_url")
    confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="conf_slider")

    rtsp_running = st.session_state.get("rtsp_running", False)
    rtsp_thread = st.session_state.get("rtsp_thread", None)

    def stop_rtsp():
        st.session_state["rtsp_running"] = False

    def start_rtsp():
        st.session_state["rtsp_running"] = True

    def run_rtsp_stream(rtsp_url, confidence_threshold):
        cap = cv2.VideoCapture(rtsp_url)
        stframe = st.empty()
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        if not cap.isOpened():
            st.error("Failed to open RTSP stream. Please check the URL.")
            return
        while st.session_state.get("rtsp_running", False) and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            for r in results:
                annotated = r.plot()
                # Log if violence detected above threshold
                if r.boxes and len(r.boxes) > 0:
                    for box in r.boxes:
                        conf = float(box.conf[0]) if hasattr(box, 'conf') else None
                        if conf is not None and conf >= confidence_threshold:
                            timestamp = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            log_violence(timestamp, f"RTSP ({rtsp_url}) frame {frame_idx}, approx {frame_idx/int(fps)}s", confidence=conf)
                stframe.image(annotated, channels="BGR", use_container_width=True)
            frame_idx += 1
        cap.release()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Start RTSP Stream"):
            if not st.session_state.get("rtsp_running", False) and rtsp_url:
                st.session_state["rtsp_running"] = True
                thread = threading.Thread(target=run_rtsp_stream, args=(rtsp_url, confidence_threshold), daemon=True)
                thread.start()
                st.session_state["rtsp_thread"] = thread
    with col2:
        if st.button("Stop RTSP Stream"):
            stop_rtsp()

    # Upload file
    uploaded_file = st.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"], key="file_uploader_detection")

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        if uploaded_file.type.startswith("image"):
            img = cv2.imread(tfile.name)
            results = model(img)
            for r in results:
                annotated = r.plot()
                st.image(annotated, channels="BGR")
                # Log if violence detected
                if r.boxes and len(r.boxes) > 0:
                    for box in r.boxes:
                        conf = float(box.conf[0]) if hasattr(box, 'conf') else None
                        if conf is not None and conf >= confidence_threshold:
                            log_violence(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), uploaded_file.name, confidence=conf)

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
                for r in results:
                    annotated = r.plot()
                    # Log if violence detected
                    if r.boxes and len(r.boxes) > 0:
                        for box in r.boxes:
                            conf = float(box.conf[0]) if hasattr(box, 'conf') else None
                            if conf is not None and conf >= confidence_threshold:
                                timestamp = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                log_violence(timestamp, f"{uploaded_file.name} (frame {frame_idx}, approx {frame_idx/int(fps)}s)", confidence=conf)
                    st.image(annotated, channels="BGR", use_container_width=True)
                frame_idx += 1
            cap.release()

elif page == "Incidents":
    st.title("Incident Table (Last 24h)")
    try:
        with open(LOG_FILE, "r") as f:
            incidents = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        incidents = []
    if incidents:
        st.dataframe(incidents)
    else:
        st.info("No incidents detected in the last 24 hours.")

