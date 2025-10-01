#!/bin/bash
# Deploy GPU Service on Instance 2

echo "🚀 Deploying GPU Service on Instance 2..."

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

# Check if NVIDIA drivers are available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA drivers not found. Please install NVIDIA drivers first."
    echo "On Ubuntu 24.04, you can install them with:"
    echo "sudo apt update && sudo apt install nvidia-driver-535"
    exit 1
fi

echo "✅ NVIDIA drivers detected:"
nvidia-smi --query-gpu=name --format=csv,noheader

# Improved GPU runtime check
check_gpu_runtime() {
    echo "🔍 Checking GPU container runtime..."
    if docker run --rm --runtime=nvidia nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        return 0
    elif docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Ensure NVIDIA Container Toolkit is installed/configured
if ! check_gpu_runtime; then
    echo "❌ NVIDIA container runtime not detected. Installing NVIDIA Container Toolkit..."

    # Remove any existing conflicting installations
    sudo apt-get remove -y nvidia-docker2 nvidia-docker docker-nvidia 2>/dev/null || true
    
    # Add package repositories
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update -y
    sudo apt-get install -y nvidia-container-toolkit nvidia-container-runtime

    # Configure Docker
    sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
    sudo systemctl restart docker

    echo "⏳ Waiting for Docker to restart..."
    sleep 10

    # Verify installation
    if check_gpu_runtime; then
        echo "✅ NVIDIA Container Toolkit installed and working."
    else
        echo "❌ GPU runtime still not available after installation."
        echo "Troubleshooting steps:"
        echo "1. Check if NVIDIA drivers are properly installed: nvidia-smi"
        echo "2. Check Docker daemon logs: sudo journalctl -u docker -f"
        echo "3. Try manual test: docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi"
        exit 1
    fi
else
    echo "✅ NVIDIA container runtime is already working."
fi

# Check if .env exists in parent directory
if [ ! -f ../.env ]; then
    echo "❌ .env file not found. Creating from template..."
    cp ../env.example ../.env
    echo "⚠️  Please edit .env file with your configuration before running again."
    exit 1
fi

# Check if model file exists in parent directory
if [ ! -f ../violence_detection_v4.pt ]; then
    echo "❌ Model file violence_detection_v4.pt not found!"
    echo "Please ensure the model file is in the parent directory."
    exit 1
fi

# Build and start GPU service
echo "🔨 Building GPU service..."
docker-compose -f docker-compose.gpu.yml build

echo "🚀 Starting GPU service..."
docker-compose -f docker-compose.gpu.yml up -d

# Wait for service to start
echo "⏳ Waiting for service to start..."
sleep 20

# Check health
echo "🔍 Checking service health..."
if curl -f http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ GPU service is running and healthy!"
    echo "🌐 GPU service available at: http://$(curl -s ifconfig.me):8001"
else
    echo "❌ GPU service health check failed. Check logs:"
    docker-compose -f docker-compose.gpu.yml logs
fi

echo "📊 Service status:"
docker-compose -f docker-compose.gpu.yml ps

echo ""
echo "🎉 GPU deployment complete!"
echo "🔧 GPU API: http://$(curl -s ifconfig.me):8001"
echo "📋 Check logs: docker-compose -f docker-compose.gpu.yml logs -f"