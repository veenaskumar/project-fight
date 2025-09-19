#!/bin/bash

# Activate virtual environment
source /home/ubuntu/project-fight/.venv/bin/activate

# SSL cert + key paths
CERT=/etc/ssl/nightshield/origin-certificate.crt
KEY=/etc/ssl/nightshield/private-key.key

# Log files
SERVER_LOG="uvicorn.log"
STREAMLIT_LOG="streamlit.log"

# Kill any previous instances
echo "Killing previous instances..."
pkill -f "uvicorn server:app"
pkill -f "streamlit run index.py"

# Start FastAPI server in background with SSL
echo "Starting FastAPI server (with SSL)..."
nohup uvicorn server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --ssl-certfile $CERT \
  --ssl-keyfile $KEY \
  > "$SERVER_LOG" 2>&1 &

# Start Streamlit app in background with SSL
echo "Starting Streamlit (with SSL)..."
nohup streamlit run index.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.sslCertFile $CERT \
  --server.sslKeyFile $KEY \
  > "$STREAMLIT_LOG" 2>&1 &

echo "✅ Both FastAPI and Streamlit started with SSL in background."
echo "📜 Server log: $SERVER_LOG"
echo "📜 Streamlit log: $STREAMLIT_LOG"

