#!/bin/bash
# Deploy Frontend + CPU Service on Instance 1

set -euo pipefail

echo "🚀 Deploying Frontend + CPU Service on Instance 1..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed. Please log out and log back in."
    exit 1
fi

# Check if Docker Compose v2 is installed
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose v2 not found. Installing docker-compose (fallback)..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Check if .env exists in parent directory
if [ ! -f ../.env ]; then
    echo "❌ .env file not found. Creating from template..."
    cp ../env.example ../.env
    echo "⚠️  Please edit .env file with your configuration before running again."
    echo "   Make sure to set GPU_SERVICE_URL to your GPU instance IP"
    exit 1
fi

# Build and start services
echo "🔨 Building services (cpu-service + frontend)..."
docker compose --env-file ../.env -f docker-compose.frontend-cpu.yml build

echo "🚀 Starting services..."
docker compose --env-file ../.env -f docker-compose.frontend-cpu.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 15

# Check CPU service health
echo "🔍 Checking CPU service health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ CPU service is running and healthy!"
else
    echo "❌ CPU service health check failed. Check logs:"
    docker compose --env-file ../.env -f docker-compose.frontend-cpu.yml logs --no-color cpu-service | tail -n 200
fi

# Check frontend health
echo "🔍 Checking frontend health..."
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Frontend is running and healthy!"
    echo "🌐 Frontend available at: http://$(curl -s ifconfig.me):8501"
else
    echo "❌ Frontend health check failed. Check logs:"
    docker compose --env-file ../.env -f docker-compose.frontend-cpu.yml logs --no-color frontend | tail -n 200
fi

echo "📊 Service status:"
docker compose --env-file ../.env -f docker-compose.frontend-cpu.yml ps

echo ""
echo "🎉 Deployment complete!"
echo "📱 Frontend: http://$(curl -s ifconfig.me):8501"
echo "🔧 CPU API: http://$(curl -s ifconfig.me):8000"
echo "📋 Check logs: docker compose --env-file ../.env -f docker-compose.frontend-cpu.yml logs -f"
