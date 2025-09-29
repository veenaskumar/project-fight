# Violence Detection System - Split Architecture

A real-time violence detection system using YOLO with a split CPU/GPU architecture for cost optimization.

## 🏗️ Architecture

- **CPU Service (Manager)**: Manages metadata, S3 storage, GPU lifecycle control
- **GPU Service (Worker)**: Handles YOLO detection and video streaming
- **Frontend**: Streamlit UI for stream management

## 📁 Project Structure

```
project-fight/
├── CPU_Server.py              # CPU service (manager)
├── GUP_server.py              # GPU service (worker)
├── index.py                   # Streamlit frontend
├── violence_detection_v4.pt   # YOLO model file
├── logo.png                   # Frontend logo
├── styles.css                 # Frontend styles
├── requirements.txt           # Python dependencies
├── env.example               # Environment template
├── .env                      # Environment variables (create from template)
└── docker/                   # Docker deployment files
    ├── README.md
    ├── Dockerfile.cpu
    ├── Dockerfile.gpu
    ├── Dockerfile.frontend
    ├── docker-compose.frontend-cpu.yml
    ├── docker-compose.gpu.yml
    ├── deploy_instance1.sh
    └── deploy_instance2.sh
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create environment file
cp env.example .env

# Edit with your configuration
nano .env
```

### 2. Deploy on Instance 1 (Frontend + CPU)

```bash
# Copy files to instance
scp -r project-fight/ user@instance1-ip:/home/user/

# SSH and deploy
ssh user@instance1-ip
cd project-fight/docker
chmod +x deploy_instance1.sh
./deploy_instance1.sh
```

### 3. Deploy on Instance 2 (GPU Only)

```bash
# Copy files to instance
scp -r project-fight/ user@instance2-ip:/home/user/

# SSH and deploy
ssh user@instance2-ip
cd project-fight/docker
chmod +x deploy_instance2.sh
./deploy_instance2.sh
```

## 🔧 Configuration

### Environment Variables

Create `.env` file with:

```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name

# GPU Instance Configuration
GPU_INSTANCE_ID=i-1234567890abcdef0
GPU_SERVICE_URL=http://instance2-ip:8001

# API Security
API_KEY=your-secure-api-key

# Frontend Configuration
CPU_SERVICE_URL=http://localhost:8000
GPU_SERVICE_URL=http://instance2-ip:8001
```

## 📊 Access Points

- **Frontend**: `http://instance1-ip:8501`
- **CPU API**: `http://instance1-ip:8000`
- **GPU API**: `http://instance2-ip:8001`

## 🔄 How It Works

1. **User adds stream** via frontend
2. **CPU service** starts GPU instance if needed
3. **GPU service** runs YOLO detection
4. **Video streams** available via WebSocket/MJPEG
5. **Detection clips** saved to S3

## 🛠️ Management

### Check Services

```bash
# Instance 1
docker-compose -f docker/docker-compose.frontend-cpu.yml ps

# Instance 2
docker-compose -f docker/docker-compose.gpu.yml ps
```

### View Logs

```bash
# Instance 1
docker-compose -f docker/docker-compose.frontend-cpu.yml logs -f

# Instance 2
docker-compose -f docker/docker-compose.gpu.yml logs -f
```

### Restart Services

```bash
# Instance 1
docker-compose -f docker/docker-compose.frontend-cpu.yml restart

# Instance 2
docker-compose -f docker/docker-compose.gpu.yml restart
```

## 📋 Requirements

- Python 3.8+
- Docker & Docker Compose
- NVIDIA Docker (for GPU instance)
- AWS Account with EC2 and S3 access
- Two EC2 instances (CPU + GPU)

## 🔒 Security

- API key authentication between services
- AWS IAM roles with minimal permissions
- Non-root Docker containers
- Environment variable configuration

## 📈 Cost Optimization

- GPU instance starts only when streams are active
- CPU instance runs continuously (cheap)
- Automatic GPU lifecycle management
- S3 storage for persistence