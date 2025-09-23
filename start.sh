#!/bin/bash

# Activate virtual environment
source /home/ubuntu/project-fight/.venv/bin/activate

# Log files
SERVER_LOG="uvicorn.log"
STREAMLIT_LOG="streamlit.log"

# Kill any previous instances
echo "Killing previous instances..."
pkill -f "uvicorn server:app"
pkill -f "streamlit run index.py"

# Start FastAPI server in background
echo "Starting FastAPI server..."
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > "$SERVER_LOG" 2>&1 &

# Start Streamlit app in background
echo "Starting Streamlit..."
nohup streamlit run index.py --server.address=0.0.0.0 --server.port=8501 > "$STREAMLIT_LOG" 2>&1 &

echo "Both FastAPI and Streamlit started in background."
echo "Server log: $SERVER_LOG"
echo "Streamlit log: $STREAMLIT_LOG"

