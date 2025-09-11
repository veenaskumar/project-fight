# app.py
import streamlit as st
import requests, base64, cv2, numpy as np
from pathlib import Path
from io import BytesIO
import threading
import asyncio
import websockets
from datetime import datetime

BACKEND_URL = "http://18.170.163.99:8000"
WS_URL = "ws://18.170.163.99:8000/ws"

st.set_page_config(
    page_title="Violence Detection Dashboard", 
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a1a, #2d2d2d);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 10px;
    }
    
    .system-title {
        color: #ff4444;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .system-subtitle {
        color: #cccccc;
        font-size: 1.2em;
        text-align: center;
        margin: 5px 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #00ff00;
        margin-right: 5px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .sidebar-logo {
        text-align: center;
        margin-bottom: 20px;
    }
    .black > strong {
        color: black;
    }
    .metric-card {
        background: linear-gradient(135deg, #2d2d2d, #1a1a1a);
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff4444;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header with Logo
# -------------------------------
st.markdown("""
<div class="main-header">
    <div class="logo-container">
        <img src="data:image/png;base64,{}" width="120" style="margin-right: 20px;">
        <div>
            <h1 class="system-title">🔴 Violence Detection System</h1>
            <p class="system-subtitle">Real-time AI-powered Security Monitoring</p>
        </div>
    </div>
    <div style="text-align: center; margin-top: 15px;">
        <span class="status-indicator"></span>
        <span style="color: #cccccc; font-size: 1.1em;">System Online</span>
        <span style="color: #666; margin: 0 20px;">|</span>
        <span style="color: #cccccc; font-size: 1.1em;">Version 2.0</span>
        <span style="color: #666; margin: 0 20px;">|</span>
        <span style="color: #cccccc; font-size: 1.1em;">AI-Powered Detection</span>
    </div>
</div>
""".format(base64.b64encode(open("logo.png", "rb").read()).decode()), unsafe_allow_html=True)

st.markdown("---")

# -------------------------------
# Sidebar with Logo and Info
# -------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <img src="data:image/png;base64,{}" width="120" style="border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">
    </div>
    """.format(base64.b64encode(open("logo.png", "rb").read()).decode()), unsafe_allow_html=True)
    
    st.markdown("### 🔴 Violence Detection System")
    st.markdown("**Real-time AI Security Monitoring**")
    
    st.markdown("---")
    
    # System Status
    st.markdown("#### 📊 System Status")
    try:
        resp = requests.get(f"{BACKEND_URL}/active_streams")
        if resp.ok:
            active_streams = resp.json()
            st.success(f"🟢 Backend Online")
            st.info(f"📡 Active Streams: {len(active_streams)}")
        else:
            st.error("🔴 Backend Offline")
    except:
        st.error("🔴 Backend Offline")
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("#### 📈 Quick Stats")
    st.metric("Detection Model", "YOLO v8")
    st.metric("Alert System", "SMS + Email")
    st.metric("Storage", "AWS S3")
    
    st.markdown("---")
    
    # Navigation
    st.markdown("#### 🧭 Navigation")
    st.markdown("""
    - **Stream Dashboard**: Add new video streams
    - **Live Preview**: View real-time detection
    - **Detection Clips**: Review recorded incidents
    """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8em;">
        <p>© 2025 NightSheield AI</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Tabs
# -------------------------------
tabs = st.tabs(["Add Stream", "Manage Streams", "Live Preview", "Detection Clips"])

# -------------------------------
# 1️⃣ Add Stream Tab
# -------------------------------
with tabs[0]:
    st.header("➕ Add New Stream")
    
    # Create a centered form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 30px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        
        name = st.text_input("Stream Name", placeholder="e.g., Camera 1, Demo Video", help="Give your stream a descriptive name")
        threshold = st.slider("Violence Threshold", 0.1, 1.0, 0.5, help="Lower values = more sensitive detection")
        phone = st.text_input("Alert Phone (+countrycode)", placeholder="+1234567890", help="Phone number for SMS alerts when violence is detected")
        
        st.markdown("---")
        
        # Input method selection
        st.subheader("📹 Video Source")
        input_method = st.radio("Choose Video Source", ["RTSP Stream", "Upload File"], horizontal=True)
        
        if input_method == "RTSP Stream":
            st.info("🌐 **RTSP Stream**: Connect to a live camera feed")
            rtsp_url = st.text_input("RTSP URL", placeholder="rtsp://username:password@ip:port/stream", help="Enter the RTSP URL of your camera")
            file_uploaded = None
        else:
            st.info("📁 **Upload File**: Use a video file for testing/demo")
            file_uploaded = st.file_uploader("Upload MP4 Video", type=["mp4"], help="Upload a video file for demo/testing")
            rtsp_url = ""

        st.markdown("---")
        
        if st.button("🚀 Add Stream", type="primary", use_container_width=True):
            if not name:
                st.error("Please enter a stream name")
            elif input_method == "RTSP Stream" and not rtsp_url:
                st.error("Please enter an RTSP URL")
            elif input_method == "Upload File" and not file_uploaded:
                st.error("Please upload a video file")
            else:
                url_or_file = rtsp_url if input_method == "RTSP Stream" else ""
                upload_flag = False

                if file_uploaded:
                    upload_flag = True
                    tmp_path = Path("uploads") / file_uploaded.name
                    tmp_path.parent.mkdir(exist_ok=True)
                    with open(tmp_path, "wb") as f:
                        f.write(file_uploaded.getbuffer())
                    url_or_file = str(tmp_path)

                payload = {
                    "name": name,
                    "url_or_file": url_or_file,
                    "threshold": threshold,
                    "phone": phone,
                    "file_uploaded": upload_flag
                }

                try:
                    resp = requests.post(f"{BACKEND_URL}/add_stream", params=payload)
                    if resp.ok:
                        st.success(f"✅ Stream added successfully! Stream ID: {resp.json().get('stream_id')}")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to add stream: {resp.text}")
                except Exception as e:
                    st.error(f"❌ Error adding stream: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Quick help section
    st.markdown("---")
    st.subheader("📖 Quick Help")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🌐 RTSP Streams
        - **Format**: `rtsp://username:password@ip:port/stream`
        - **Example**: `rtsp://admin:password@192.168.1.100:554/stream1`
        - **Requirements**: Camera must support RTSP protocol
        """)
    
    with col2:
        st.markdown("""
        ### 📁 Upload Files
        - **Supported**: MP4 video files
        - **Use Case**: Testing, demos, offline analysis
        - **Note**: Files are stored locally for processing
        """)

# -------------------------------
# 2️⃣ Manage Streams Tab
# -------------------------------
with tabs[1]:
    st.header("📋 Manage Existing Streams")
    
    # Fetch and display active streams
    try:
        resp = requests.get(f"{BACKEND_URL}/active_streams")
        if resp.ok:
            streams = resp.json()
            
            if not streams:
                st.info("No streams found. Add a stream to get started!")
                st.markdown("""
                ### 🚀 Get Started:
                1. Go to **Add Stream** tab to create your first stream
                2. Come back here to manage your streams
                3. Use **Live Preview** to view your streams in action
                """)
            else:
                # Stream statistics
                running_count = len([s for s in streams if s.get('running', False)])
                stopped_count = len([s for s in streams if not s.get('running', False)])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Streams", len(streams))
                with col2:
                    st.metric("Running", running_count, delta=f"+{running_count}" if running_count > 0 else None)
                with col3:
                    st.metric("Stopped", stopped_count, delta=f"-{stopped_count}" if stopped_count > 0 else None)
                with col4:
                    st.metric("Detection Active", running_count)
                
                st.markdown("---")
                
                # Stream management section
                for i, stream in enumerate(streams):
                    with st.container():
                        # Stream card with enhanced styling
                        status_color = "#28a745" if stream.get('running', False) else "#ffc107"
                        status_icon = "🟢" if stream.get('running', False) else "🟡"
                        status_text = "RUNNING" if stream.get('running', False) else "STOPPED"
                        
                        st.markdown(f"""
                        <div style="border: 2px solid {status_color}; border-radius: 12px; padding: 20px; margin: 15px 0; background: {'linear-gradient(135deg, #e8f5e8, #f0f8f0)' if stream.get('running', False) else 'linear-gradient(135deg, #fff3cd, #fef9e7)'}; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                <h3 style="margin: 0; color: {status_color}; display: flex; align-items: center;">
                                    {status_icon} {stream['name']}
                                    <span style="background: {status_color}; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.7em; margin-left: 10px;">{status_text}</span>
                                </h3>
                                <div style="color: #666; font-size: 0.9em;">
                                    ID: {stream['stream_id'][:8]}...
                                </div>
                            </div>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;" classname="black">
                                <div style="color:black">
                                    <strong>Type:</strong> {'Demo Video' if stream.get('is_demo') else 'RTSP Stream'}<br>
                                    <strong>Threshold:</strong> {stream.get('threshold', 0.5):.2f}
                                </div>
                                <div style="color:black">
                                    <strong>Phone:</strong> {stream.get('phone', 'Not set')}<br>
                                    <strong>Created:</strong> Recently
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Stream controls
                        col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])
                        
                        with col_a:
                            if stream.get('running', False):
                                if st.button("⏹️ Stop Stream", key=f"stop_{stream['stream_id']}", help="Stop this stream"):
                                    try:
                                        resp = requests.post(f"{BACKEND_URL}/stop_stream/{stream['stream_id']}")
                                        if resp.ok:
                                            st.success("Stream stopped!")
                                            st.rerun()
                                        else:
                                            st.error("Failed to stop stream")
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                            else:
                                if st.button("▶️ Start Stream", key=f"start_{stream['stream_id']}", help="Start this stream"):
                                    try:
                                        resp = requests.post(f"{BACKEND_URL}/start_stream/{stream['stream_id']}")
                                        if resp.ok:
                                            st.success("Stream started!")
                                            st.rerun()
                                        else:
                                            st.error("Failed to start stream")
                                    except Exception as e:
                                        st.error(f"Error: {e}")
                        
                        with col_b:
                            if st.button("👁️ View Stream", key=f"view_{stream['stream_id']}", help="View this stream in Live Preview"):
                                st.session_state['selected_stream'] = stream['stream_id']
                                st.switch_page("Live Preview")
                        
                        with col_c:
                            if st.button("⚙️ Settings", key=f"settings_{stream['stream_id']}", help="Configure stream settings"):
                                st.info("Settings panel coming soon!")
                        
                        with col_d:
                            if st.button("🗑️ Delete", key=f"delete_{stream['stream_id']}", help="Delete this stream permanently"):
                                try:
                                    resp = requests.delete(f"{BACKEND_URL}/delete_stream/{stream['stream_id']}")
                                    if resp.ok:
                                        st.success("Stream deleted!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete stream")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        
                        st.markdown("---")
        else:
            st.error("Failed to fetch streams")
    except Exception as e:
        st.error(f"Error fetching streams: {e}")

# -------------------------------
# 3️⃣ Live Preview Tab
# -------------------------------
with tabs[2]:
    st.header("🔴 Live Video Stream - Real-time Violence Detection")
    
    # Check if stream was selected from dashboard
    if 'selected_stream' in st.session_state:
        selected_stream_id = st.session_state['selected_stream']
        del st.session_state['selected_stream']  # Clear after use
    else:
        selected_stream_id = None
    
    # Fetch active streams from backend
    try:
        resp = requests.get(f"{BACKEND_URL}/active_streams")
        active_streams = resp.json() if resp.ok else []
    except Exception as e:
        st.error(f"Failed to fetch active streams: {e}")
        active_streams = []

    if not active_streams:
        st.info("No streams found. Add a stream first.")
        st.markdown("""
        ### 🚀 Quick Start:
        1. Go to **Stream Dashboard** tab
        2. Add a new stream (RTSP URL or upload MP4)
        3. Come back here to see **live video** automatically!
        """)
    else:
        # Show stream selection
        st.subheader(f"📡 Select Live Stream ({len(active_streams)} available)")
        
        # Stream selection with better formatting
        running_streams = [s for s in active_streams if s.get('running', False)]
        stopped_streams = [s for s in active_streams if not s.get('running', False)]
        
        if running_streams:
            st.success(f"🟢 {len(running_streams)} streams running")
        if stopped_streams:
            st.warning(f"🟡 {len(stopped_streams)} streams stopped")
        
        # Create stream options with status indicators
        stream_options = {}
        for s in active_streams:
            status_icon = "🟢" if s.get('running', False) else "🟡"
            stream_options[f"{status_icon} {s['name']} ({'Running' if s.get('running', False) else 'Stopped'})"] = s['stream_id']
        
        # Auto-select if coming from dashboard
        if selected_stream_id:
            # Find the stream name for the selected ID
            selected_stream_name = next((name for name, sid in stream_options.items() if sid == selected_stream_id), None)
        else:
            selected_stream_name = st.selectbox("Choose Stream", list(stream_options.keys()))
        
        if selected_stream_name:
            selected_stream_id = stream_options[selected_stream_name]
            current_stream = next((s for s in active_streams if s['stream_id'] == selected_stream_id), None)
            
            if current_stream:
                # Stream status header
                status_color = "#28a745" if current_stream.get('running', False) else "#ffc107"
                status_text = "LIVE" if current_stream.get('running', False) else "STOPPED"
                
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, {status_color}, {status_color}88); color: white; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                    <h3 style="margin: 0; color: white;">🔴 {status_text}: {current_stream['name']}</h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Real-time Violence Detection Video Stream</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Stream controls
                col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
                
                with col1:
                    if st.button("🔄 Refresh", help="Refresh the stream display"):
                        st.rerun()
                
                with col2:
                    if current_stream.get('running', False):
                        if st.button("⏹️ Stop Stream", help="Stop this stream"):
                            try:
                                resp = requests.post(f"{BACKEND_URL}/stop_stream/{selected_stream_id}")
                                if resp.ok:
                                    st.success("Stream stopped!")
                                    st.rerun()
                                else:
                                    st.error("Failed to stop stream")
                            except Exception as e:
                                st.error(f"Error: {e}")
                    else:
                        if st.button("▶️ Start Stream", help="Start this stream"):
                            try:
                                resp = requests.post(f"{BACKEND_URL}/start_stream/{selected_stream_id}")
                                if resp.ok:
                                    st.success("Stream started!")
                                    st.rerun()
                                else:
                                    st.error("Failed to start stream")
                            except Exception as e:
                                st.error(f"Error: {e}")
                
                with col3:
                    if st.button("📸 Snapshot", help="Take a snapshot"):
                        st.info("📸 Snapshot saved! (Feature coming soon)")
                
                with col4:
                    if current_stream.get('running', False):
                        st.success("🔴 Stream is live and processing")
                    else:
                        st.warning("⏸️ Stream is stopped - click Start to begin")
                
                # Video display
                if current_stream.get('running', False):
                    video_url = f"{BACKEND_URL}/video/{selected_stream_id}"
                    
                    # Main video player
                    st.markdown("### 📹 Live Video Feed")
                    st.markdown(f"""
                    <div style="text-align: center; margin: 20px 0;">
                        <video width="100%" height="500" controls autoplay muted loop style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.3); background: #000;">
                            <source src="{video_url}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                    """, unsafe_allow_html=True)
            
                    # Alternative MJPEG stream for better compatibility
                    st.markdown("### 🔄 Alternative Stream (MJPEG)")
                    st.markdown(f"""
                    <div style="text-align: center; border: 2px solid #ddd; border-radius: 10px; padding: 10px; background: #f8f9fa;">
                        <img src="{video_url}" width="100%" style="border-radius: 8px; max-height: 400px; object-fit: contain;" />
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("⏸️ Stream is stopped. Click 'Start Stream' to begin live video feed.")
                
                # Stream information
                st.markdown("---")
                st.subheader("📊 Stream Information")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Stream Status", "🟢 Live" if current_stream.get('running', False) else "🟡 Stopped")
                with col2:
                    st.metric("Stream ID", selected_stream_id[:8] + "...")
                with col3:
                    st.metric("Detection", "Active" if current_stream.get('running', False) else "Inactive")
                with col4:
                    st.metric("Type", "Demo Video" if current_stream.get('is_demo') else "RTSP Stream")
                
                # Additional info
                st.markdown("---")
                st.markdown("""
                ### 📖 How to Use:
                - **Video Player**: Use controls to play/pause/seek through the stream
                - **MJPEG Stream**: Continuous image stream below for better compatibility
                - **Violence Detection**: Red boxes and alerts appear when violence is detected
                - **Real-time Updates**: Both streams update automatically with AI detection overlays
                - **Controls**: Use the buttons above to start/stop/refresh the stream
                """)
            else:
                st.error("Stream not found or not accessible")


# -------------------------------
# 4️⃣ Detection Clips Tab
# -------------------------------
with tabs[3]:
    st.header("Detection Clips")

    stream_filter = st.text_input("Filter by Stream (optional)")
    sort_order = st.selectbox("Sort Order", ["Newest First", "Oldest First"])

    sort_param = "desc" if sort_order == "Newest First" else "asc"
    params = {"sort": sort_param}
    if stream_filter:
        params["stream"] = stream_filter

    try:
        resp = requests.get(f"{BACKEND_URL}/logs", params=params, timeout=10)
        logs = resp.json() if resp.ok else []
    except Exception as e:
        st.error(f"Error fetching logs: {e}")
        logs = []

    # Apply client-side, case-insensitive substring filter on stream name
    if stream_filter:
        needle = stream_filter.strip().lower()
        logs = [l for l in logs if needle in (l.get("stream", "").lower())]

    # Client-side sort by parsed timestamp to ensure correct ordering
    from datetime import datetime as _dt
    def _parse_ts(s: str):
        fmts = [
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
        ]
        for f in fmts:
            try:
                return _dt.strptime((s or "").strip(), f)
            except Exception:
                continue
        return _dt.min

    logs.sort(key=lambda x: _parse_ts(x.get("timestamp")), reverse=(sort_order == "Newest First"))

    if not logs:
        st.info("No detection clips available.")
    else:
        for entry in logs:
            ts = entry.get("timestamp", "N/A")
            stream_name = entry.get("stream", "Unknown")
            conf = round(entry.get("confidence", 0.0), 2)
            clip_url = entry.get("clip_url")
            snapshot_url = entry.get("snapshot_url")

            with st.container():
                st.markdown(f"### 📌 {stream_name} - {ts}")
                st.write(f"**Confidence:** {conf}")

                cols = st.columns(2)

                # Snapshot preview
                with cols[0]:
                    st.write(f"Clip URL: {clip_url}")
                    if snapshot_url:
                        st.image(snapshot_url, caption="Snapshot", use_container_width=True)
                    else:
                        st.warning("No snapshot available")

                # Clip preview
                with cols[1]:
                    st.write(f"Snapshot URL: {snapshot_url}")
                    if clip_url:
                        st.video(clip_url)
                        st.markdown(f"[🔗 Download Clip]({clip_url})", unsafe_allow_html=True)
                    else:
                        st.warning("No clip available")

                st.markdown("---")




