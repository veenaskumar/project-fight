#!/bin/bash
# Deploy CPU Service on CPU Instance

echo "🚀 Deploying CPU Service (Manager)..."

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


# Build and start CPU service
echo "🔨 Building CPU service..."
docker-compose -f docker-compose.cpu.yml build

echo "🚀 Starting CPU service..."
docker-compose -f docker-compose.cpu.yml up -d

# Wait for service to start
echo "⏳ Waiting for service to start..."
sleep 10

# Check health
echo "🔍 Checking service health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ CPU service is running and healthy!"
    echo "🌐 Service available at: http://$(curl -s ifconfig.me):8000"
else
    echo "❌ CPU service health check failed. Check logs:"
    docker-compose -f docker-compose.cpu.yml logs
fi

echo "📊 Service status:"
docker-compose -f docker-compose.cpu.yml ps
