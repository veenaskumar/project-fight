# app.py
import streamlit as st
import requests, base64, cv2, numpy as np
from pathlib import Path
from io import BytesIO
import threading
import asyncio
import websockets
from datetime import datetime

BACKEND_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"

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
        <p>© 2024 Violence Detection System</p>
        <p>Powered by AI & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Tabs
# -------------------------------
tabs = st.tabs(["Stream Dashboard", "Live Preview", "Detection Clips"])

# -------------------------------
# 1️⃣ Stream Dashboard Tab
# -------------------------------
with tabs[0]:
    st.header("Add New Stream / Demo Video")

    name = st.text_input("Stream Name")
    threshold = st.slider("Violence Threshold", 0.1, 1.0, 0.5)
    phone = st.text_input("Alert Phone (+countrycode)")
    rtsp_url = st.text_input("RTSP URL")
    file_uploaded = st.file_uploader("Or upload local MP4", type=["mp4"])

    if st.button("Add Stream"):
        url_or_file = rtsp_url
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

        resp = requests.post(f"{BACKEND_URL}/add_stream", params=payload)
        if resp.ok:
            st.success(f"Stream added successfully! Stream ID: {resp.json().get('stream_id')}")
        else:
            st.error("Failed to add stream")

# -------------------------------
# 2️⃣ Live Preview Tab
# -------------------------------
# -------------------------------
# 2️⃣ Live Preview Tab (True Live Video Streaming)
# -------------------------------
with tabs[1]:
    st.header("🔴 Live Video Stream - Real-time Violence Detection")
    
    # Fetch active streams from backend
    try:
        resp = requests.get(f"{BACKEND_URL}/active_streams")
        active_streams = resp.json() if resp.ok else []
    except Exception as e:
        st.error(f"Failed to fetch active streams: {e}")
        active_streams = []

    if not active_streams:
        st.info("No active streams. Add a stream first.")
        st.markdown("""
        ### 🚀 Quick Start:
        1. Go to **Stream Dashboard** tab
        2. Add a new stream (RTSP URL or upload MP4)
        3. Come back here to see **live video** automatically!
        """)
    else:
        # Show stream selection
        st.subheader(f"📡 Select Live Stream ({len(active_streams)} available)")
        
        # Stream selection
        stream_options = {f"{s['name']} (ID: {s['stream_id']})": s['stream_id'] for s in active_streams}
        selected_stream_name = st.selectbox("Choose Stream", list(stream_options.keys()))
        
        if selected_stream_name:
            selected_stream_id = stream_options[selected_stream_name]
            
            # Create live video container
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #ff4444, #ff6666); color: white; padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                <h3 style="margin: 0; color: white;">🔴 LIVE: {selected_stream_name}</h3>
                <p style="margin: 5px 0 0 0; opacity: 0.9;">Real-time Violence Detection Video Stream</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Controls
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("🔄 Refresh Stream"):
                    st.rerun()
            with col2:
                if st.button("📸 Take Snapshot"):
                    st.info("Snapshot feature coming soon!")
            with col3:
                st.info("🔴 Live video streaming with real-time violence detection")
            
            # Live video streaming using HTML video element
            video_url = f"{BACKEND_URL}/video/{selected_stream_id}"
            
            st.markdown(f"""
            <div style="text-align: center; margin: 20px 0;">
                <video width="100%" height="400" controls autoplay muted style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                    <source src="{video_url}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            """, unsafe_allow_html=True)
            
            # Status information
            st.markdown("---")
            st.subheader("📊 Stream Information")
            
            # Get stream info
            try:
                resp = requests.get(f"{BACKEND_URL}/active_streams")
                if resp.ok:
                    streams = resp.json()
                    current_stream = next((s for s in streams if s['stream_id'] == selected_stream_id), None)
                    
                    if current_stream:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Stream Status", "🟢 Live")
                        with col2:
                            st.metric("Stream ID", selected_stream_id[:8] + "...")
                        with col3:
                            st.metric("Detection", "Active")
                        with col4:
                            st.metric("Last Check", datetime.now().strftime("%H:%M:%S"))
                    else:
                        st.error("Stream not found or not running")
                else:
                    st.error("Failed to get stream information")
            except Exception as e:
                st.error(f"Error getting stream info: {e}")
            
            # Alternative: MJPEG stream for better compatibility
            st.markdown("---")
            st.subheader("🔄 Alternative Stream (MJPEG)")
            st.markdown(f"""
            <div style="text-align: center;">
                <img src="{video_url}" width="100%" style="border-radius: 10px; max-height: 400px; object-fit: contain;" />
            </div>
            """, unsafe_allow_html=True)
            
            # Instructions
            st.markdown("---")
            st.markdown("""
            ### 📖 How to Use:
            1. **Video Player**: Use the video controls above to play/pause/seek
            2. **MJPEG Stream**: The image below shows a continuous stream
            3. **Violence Detection**: Red boxes and alerts appear when violence is detected
            4. **Real-time Updates**: Both streams update automatically with detection overlays
            """)


# -------------------------------
# 3️⃣ Detection Clips Tab
# -------------------------------
with tabs[2]:
    st.header("Detection Clips")
    # Fetch logs from backend S3
    try:
        logs_resp = requests.get(f"{BACKEND_URL}/logs")  # Implement /logs endpoint in backend
        logs = logs_resp.json() if logs_resp.ok else []
    except:
        logs = []

    for entry in logs[::-1]:  # show latest first
        ts = entry.get("timestamp")
        stream_name = entry.get("stream")
        conf = entry.get("confidence")
        clip = entry.get("clip")
        snapshot = entry.get("snapshot")

        st.subheader(f"{stream_name} - {ts} - Confidence: {conf:.2f}")
        cols = st.columns(2)
        if snapshot:
            cols[0].image(requests.get(snapshot).content, caption="Snapshot")
        if clip:
            cols[1].video(requests.get(clip).content, format="video/mp4")
