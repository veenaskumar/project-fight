#!/bin/bash
# Deploy GPU Service on GPU Instance

echo "🚀 Deploying GPU Service (Worker)..."

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

# Check if NVIDIA Docker is installed
if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
    echo "❌ NVIDIA Docker not found. Installing NVIDIA Docker..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update && sudo apt-get install -y nvidia-docker2
    sudo systemctl restart docker
    echo "✅ NVIDIA Docker installed. Please run the script again."
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Creating from template..."
    cp env.example .env
    echo "⚠️  Please edit .env file with your configuration before running again."
    exit 1
fi

# Check if model file exists
if [ ! -f violence_detection_v4.pt ]; then
    echo "❌ Model file violence_detection_v4.pt not found!"
    echo "Please ensure the model file is in the current directory."
    exit 1
fi

# Build and start GPU service
echo "🔨 Building GPU service..."
docker-compose -f docker-compose.gpu.yml build

echo "🚀 Starting GPU service..."
docker-compose -f docker-compose.gpu.yml up -d

# Wait for service to start
echo "⏳ Waiting for service to start..."
sleep 15

# Check health
echo "🔍 Checking service health..."
if curl -f http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ GPU service is running and healthy!"
    echo "🌐 Service available at: http://$(curl -s ifconfig.me):8001"
else
    echo "❌ GPU service health check failed. Check logs:"
    docker-compose -f docker-compose.gpu.yml logs
fi

echo "📊 Service status:"
docker-compose -f docker-compose.gpu.yml ps
