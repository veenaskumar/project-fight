import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
from datetime import datetime
import json
import threading
import queue

LOG_FILE = "violence_detection_log.json"

# -------------------------------
# Auto-delete logs older than 24h
# -------------------------------
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
            if (now - entry_time).total_seconds() < 86400:  # 24h
                new_data.append(entry)
        except Exception:
            new_data.append(entry)

    if len(new_data) != len(data):
        with open(LOG_FILE, "w") as f:
            json.dump(new_data, f, indent=4)


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


# Run cleanup on startup
auto_delete_old_logs()

# -------------------------------
# Load YOLO model
# -------------------------------
model = YOLO("final_fight_fall.pt")

# -------------------------------
# Sidebar navigation
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Detection", "Incidents"])

# ==============================================================
# Detection Page
# ==============================================================

if page == "Detection":
    st.title("🎥 Violence Detection with YOLOv8")
    st.write("Upload an image, video, or connect an RTSP stream for detection.")

    # RTSP input and controls in sidebar
    st.sidebar.subheader("RTSP Controls")
    rtsp_url = st.sidebar.text_input("RTSP stream URL", key="rtsp_url", value="rtsp://127.0.0.1:8554/mystream")
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="conf_slider"
    )

    # Session state management
    rtsp_running = st.session_state.get("rtsp_running", False)

    def stop_rtsp():
        st.session_state["rtsp_running"] = False

    def run_rtsp_stream(rtsp_url, confidence_threshold, frame_queue, stop_event):
    # Force TCP transport (important for stability with mediamtx)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            frame_queue.put((None, f"❌ Failed to open RTSP stream: {rtsp_url}"))
            return
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25  # fallback to 25 if not detected

        while not stop_event.is_set() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                frame_queue.put((None, "⚠️ Failed to read frame. Stream ended or connection dropped."))
                break

        # Run YOLO model inference
            results = model(frame)

            for r in results:
                annotated = r.plot()

            # Process detections
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        try:
                            conf = float(box.conf[0])
                        except Exception:
                            conf = None

                        if conf is not None and conf >= confidence_threshold:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            log_violence(
                                timestamp,
                                f"RTSP ({rtsp_url}) frame {frame_idx}, approx {frame_idx / int(fps)}s",
                                confidence=conf
                            )

            # Only send the latest annotated frame
                frame_queue.put((annotated, None))

            frame_idx += 1

        cap.release()
        frame_queue.put((None, None))  # Signal end

    # Sidebar buttons
    col1, col2 = st.sidebar.columns(2)
    if "rtsp_stop_event" not in st.session_state:
        st.session_state["rtsp_stop_event"] = threading.Event()
    if "rtsp_frame_queue" not in st.session_state:
        st.session_state["rtsp_frame_queue"] = queue.Queue(maxsize=2)
    with col1:
        if st.button("Start RTSP Stream"):
            if not st.session_state.get("rtsp_running", False) and rtsp_url:
                st.session_state["rtsp_running"] = True
                st.session_state["rtsp_stop_event"].clear()
                st.session_state["rtsp_frame_queue"] = queue.Queue(maxsize=2)
                thread = threading.Thread(
                    target=run_rtsp_stream,
                    args=(rtsp_url, confidence_threshold, st.session_state["rtsp_frame_queue"], st.session_state["rtsp_stop_event"]),
                    daemon=True
                )
                thread.start()
                st.session_state["rtsp_thread"] = thread
    with col2:
        if st.button("Stop RTSP Stream"):
            stop_rtsp()
            st.session_state["rtsp_stop_event"].set()

    # -----------------------------------------------------------
    # RTSP Stream Display (Main Thread)
    # -----------------------------------------------------------
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
                # End of stream
                st.session_state["rtsp_running"] = False
                break

    # -----------------------------------------------------------
    # File upload detection (unchanged)
    # -----------------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload Image/Video",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        key="file_uploader_detection"
    )

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        if uploaded_file.type.startswith("image"):
            img = cv2.imread(tfile.name)
            results = model(img)
            for r in results:
                annotated = r.plot()
                st.image(annotated, channels="BGR")
                if r.boxes and len(r.boxes) > 0:
                    for box in r.boxes:
                        conf = float(box.conf[0]) if hasattr(box, 'conf') else None
                        if conf is not None and conf >= confidence_threshold:
                            log_violence(
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                uploaded_file.name,
                                confidence=conf
                            )

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
                    if r.boxes and len(r.boxes) > 0:
                        for box in r.boxes:
                            conf = float(box.conf[0]) if hasattr(box, 'conf') else None
                            if conf is not None and conf >= confidence_threshold:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                log_violence(
                                    timestamp,
                                    f"{uploaded_file.name} (frame {frame_idx}, approx {frame_idx/int(fps)}s)",
                                    confidence=conf
                                )
                    st.image(annotated, channels="BGR", use_container_width=True)

                frame_idx += 1

            cap.release()

# ==============================================================
# Incidents Page
# ==============================================================

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
