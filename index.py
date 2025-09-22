# index.py

import streamlit as st
import requests
from pathlib import Path

# --- Configuration ---
BACKEND_URL = "http://18.170.163.99:8000"

# --- Page Setup ---
st.set_page_config(
    page_title="Violence Detection Dashboard",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
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
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #ff4444;
        color: #ff4444;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff4444;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- State Management ---
if 'selected_stream' not in st.session_state:
    st.session_state['selected_stream'] = None
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = "Add Stream"

# --- Helper Functions ---
def get_active_streams():
    try:
        resp = requests.get(f"{BACKEND_URL}/active_streams", timeout=5)
        return resp.json() if resp.ok else []
    except requests.exceptions.ConnectionError:
        return None # Indicates backend is offline
    except Exception:
        return []

# --- Header ---
with st.container():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("logo.png", width=120)
    with col2:
        st.markdown('<p class="system-title">AI VIOLENCE DETECTION SYSTEM</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.image("logo.png")
    st.title("Navigation")
    st.info("Select a page to view its contents.")
    st.markdown("---")
    st.markdown("#### 📊 System Status")
    active_streams = get_active_streams()
    if active_streams is None:
        st.error("🔴 Backend Offline")
    else:
        st.success(f"🟢 Backend Online")
        st.info(f"📡 Active Streams: {len(active_streams)}")


# --- Main Content ---
tab1, tab2, tab3, tab4 = st.tabs(["➕ Add Stream", "📋 Manage Streams", "🔴 Live Preview", "🎬 Detection Clips"])

# -------------------------------
# 1️⃣ Add Stream Tab
# -------------------------------
with tab1:
    st.header("Add New Video Stream")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("add_stream_form"):
            name = st.text_input("Stream Name*", placeholder="e.g., Lobby Camera")
            threshold = st.slider("Violence Threshold", 0.1, 1.0, 0.5)
            phone = st.text_input("Alert Phone (+countrycode)", placeholder="+1234567890")
            
            st.markdown("---")
            input_method = st.radio("Choose Video Source", ["RTSP Stream", "Upload File"], horizontal=True)
            
            url_or_file = ""
            file_uploaded = None
            if input_method == "RTSP Stream":
                url_or_file = st.text_input("RTSP URL*", placeholder="rtsp://user:pass@ip:port/stream")
            else:
                file_uploaded = st.file_uploader("Upload MP4 Video*", type=["mp4"])

            submitted = st.form_submit_button("🚀 Add Stream", type="primary", use_container_width=True)
            
            if submitted:
                is_valid = True
                if not name:
                    st.error("Please enter a stream name.")
                    is_valid = False
                if input_method == "RTSP Stream" and not url_or_file:
                    st.error("Please enter an RTSP URL.")
                    is_valid = False
                if input_method == "Upload File" and not file_uploaded:
                    st.error("Please upload a video file.")
                    is_valid = False

                if is_valid:
                    payload = {"name": name, "threshold": threshold, "phone": phone}
                    
                    if input_method == "RTSP Stream":
                        payload["url_or_file"] = url_or_file
                        payload["file_uploaded"] = False
                    else:
                        tmp_path = Path("uploads") / file_uploaded.name
                        tmp_path.parent.mkdir(exist_ok=True)
                        with open(tmp_path, "wb") as f:
                            f.write(file_uploaded.getbuffer())
                        payload["url_or_file"] = file_uploaded.name
                        payload["file_uploaded"] = True
                    
                    try:
                        resp = requests.post(f"{BACKEND_URL}/add_stream", params=payload)
                        if resp.ok:
                            st.success(f"✅ Stream '{name}' added successfully!")
                            st.balloons()
                        else:
                            st.error(f"❌ Failed to add stream: {resp.text}")
                    except Exception as e:
                        st.error(f"❌ Connection error: {e}")

# -------------------------------
# 2️⃣ Manage Streams Tab
# -------------------------------
with tab2:
    st.header("Manage Existing Streams")
    if st.button("🔄 Refresh Streams"):
        st.rerun()
    st.markdown("---")
    
    streams = get_active_streams()
    
    if streams is None:
        st.error("Could not connect to the backend. Please ensure it is running.")
    elif not streams:
        st.info("No streams found. Add a stream to get started!")
    else:
        for stream in streams:
            status_icon = "🟢" if stream.get('running') else "🟡"
            status_text = "RUNNING" if stream.get('running') else "STOPPED"
            
            with st.expander(f"{status_icon} **{stream['name']}** - {status_text}", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**ID:** `{stream['stream_id'][:8]}...`")
                    st.write(f"**Type:** {'Demo Video' if stream.get('is_demo') else 'RTSP Stream'}")
                with col_b:
                    st.write(f"**Alert Phone:** {stream.get('phone') or 'Not set'}")
                    st.write(f"**Threshold:** {stream.get('threshold', 0.5):.2f}")

                btn_cols = st.columns(4)
                
                if btn_cols[0].button("👁️ View", key=f"view_{stream['stream_id']}", help="Select this stream for Live Preview"):
                    st.session_state['selected_stream'] = stream['stream_id']
                    st.success(f"'{stream['name']}' selected. Please switch to the '🔴 Live Preview' tab.")
                    # No rerun needed here, the message is enough guidance.

                if stream.get('running'):
                    if btn_cols[1].button("⏹️ Stop", key=f"stop_{stream['stream_id']}", type="secondary"):
                        requests.post(f"{BACKEND_URL}/stop_stream/{stream['stream_id']}")
                        st.rerun()
                else:
                    if btn_cols[1].button("▶️ Start", key=f"start_{stream['stream_id']}", type="secondary"):
                        requests.post(f"{BACKEND_URL}/start_stream/{stream['stream_id']}")
                        st.rerun()

                if btn_cols[3].button("🗑️ Delete", key=f"delete_{stream['stream_id']}", type="primary"):
                    requests.delete(f"{BACKEND_URL}/delete_stream/{stream['stream_id']}")
                    if st.session_state['selected_stream'] == stream['stream_id']:
                         st.session_state['selected_stream'] = None
                    st.rerun()

# -------------------------------
# 3️⃣ Live Preview Tab
# -------------------------------
with tab3:
    st.header("🔴 Live Video Preview")
    
    streams = get_active_streams()
    running_streams = [s for s in streams if s.get('running')] if streams else []
    
    if not running_streams:
        st.warning("No streams are currently running. Please start a stream from the '📋 Manage Streams' tab to view a live preview.")
    else:
        stream_options = {s['name']: s['stream_id'] for s in running_streams}
        
        selected_index = 0
        if st.session_state.selected_stream in stream_options.values():
            selected_index = list(stream_options.values()).index(st.session_state.selected_stream)
        else:
            # If the selected stream isn't running, clear it and default to the first running one
            st.session_state.selected_stream = None

        selected_name = st.selectbox("Choose a running stream to view:", options=stream_options.keys(), index=selected_index)
        
        if selected_name:
            selected_stream_id = stream_options[selected_name]
            video_url = f"{BACKEND_URL}/video/{selected_stream_id}"
            
            st.markdown(f"#### Now Viewing: **{selected_name}**")
            st.image(video_url, caption=f"Live Stream: {selected_name}")

# -------------------------------
# 4️⃣ Detection Clips Tab
# -------------------------------
with tab4:
    st.header("🎬 Review Detection Clips")
    
    col1, col2 = st.columns(2)
    stream_filter = col1.text_input("Filter by Stream Name")
    sort_order = col2.selectbox("Sort Order", ["Newest First", "Oldest First"])
    
    if st.button("🔄 Refresh Logs"):
        st.rerun()
    st.markdown("---")
    
    try:
        params = {"sort": "desc" if sort_order == "Newest First" else "asc"}
        resp = requests.get(f"{BACKEND_URL}/logs", params=params, timeout=10)
        logs = resp.json() if resp.ok else []

        if stream_filter:
            logs = [log for log in logs if stream_filter.lower() in log.get("stream", "").lower()]
            
        if not logs:
            st.info("No detection clips found matching your criteria.")
        else:
            for entry in logs:
                with st.container():
                    col_a, col_b = st.columns([2, 1])
                    
                    with col_a:
                        st.subheader(f"Event on: {entry.get('stream', 'N/A')}")
                        st.write(f"**Timestamp:** {entry.get('timestamp', 'N/A')}")
                        st.write(f"**Detection Confidence:** {entry.get('confidence', 0):.2f}")
                        
                        clip_url = entry.get("clip_url")
                        if clip_url:
                            st.video(clip_url)
                        else:
                            st.warning("No video clip available.")
                            
                    with col_b:
                        snapshot_url = entry.get("snapshot_url")
                        if snapshot_url:
                            st.image(snapshot_url, caption="Detection Snapshot")
                        else:
                            st.warning("No snapshot available.")
                    
                    st.markdown("---")

    except Exception as e:
        st.error(f"Failed to fetch detection logs: {e}")