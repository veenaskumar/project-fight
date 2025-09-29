#!/bin/bash
# Deploy Frontend on any machine

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

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Creating from template..."
    cp env.example .env
    echo "⚠️  Please edit .env file with your CPU and GPU service URLs before running again."
    exit 1
fi

# Build and start frontend
echo "🔨 Building frontend..."
docker-compose --profile frontend build

echo "🚀 Starting frontend..."
docker-compose --profile frontend up -d

# Wait for service to start
echo "⏳ Waiting for service to start..."
sleep 10

# Check health
echo "🔍 Checking service health..."
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Frontend is running and healthy!"
    echo "🌐 Frontend available at: http://$(curl -s ifconfig.me):8501"
else
    echo "❌ Frontend health check failed. Check logs:"
    docker-compose --profile frontend logs
fi

echo "📊 Service status:"
docker-compose --profile frontend ps
