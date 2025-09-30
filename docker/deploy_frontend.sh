#!/bin/bash
# Deploy Frontend on any machine

set -euo pipefail

echo "🚀 Deploying Frontend (Streamlit)..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed. Please log out and log back in."
    exit 1
fi

if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose v2 not found. Installing docker-compose (fallback)..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Ensure env file exists at project root
if [ ! -f ../.env ]; then
    echo "❌ ../.env not found. Creating from template..."
    cp ../env.example ../.env
    echo "⚠️  Edit ../.env with your CPU and GPU service URLs, then rerun."
    exit 1
fi

# Build and start frontend
echo "🔨 Building frontend..."
docker compose --env-file ../.env -f docker-compose.frontend-cpu.yml build frontend

echo "🚀 Starting frontend..."
docker compose --env-file ../.env -f docker-compose.frontend-cpu.yml up -d --no-deps frontend

# Wait for service to start
echo "⏳ Waiting for service to start..."
sleep 10

echo "🔍 Checking service health..."
if curl -fsS http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Frontend is running and healthy!"
    echo "🌐 Frontend available at: http://$(curl -s ifconfig.me):8501"
else
    echo "❌ Frontend health check failed. Check logs:"
    docker compose --env-file ../.env -f docker-compose.frontend-cpu.yml logs --no-color frontend | tail -n 200
fi

echo "📊 Service status:"
docker compose --env-file ../.env -f docker-compose.frontend-cpu.yml ps
