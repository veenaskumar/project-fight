# Docker Deployment Files

This folder contains all Docker-related files for the Violence Detection System.

## Files Overview

### Dockerfiles
- `Dockerfile.cpu` - CPU Service (Manager)
- `Dockerfile.gpu` - GPU Service (Worker)
- `Dockerfile.frontend` - Streamlit Frontend

### Docker Compose Files
- `docker-compose.frontend-cpu.yml` - Instance 1: Frontend + CPU Service
- `docker-compose.gpu.yml` - Instance 2: GPU Service only
- `docker-compose.cpu.yml` - CPU Service only (alternative)
- `docker-compose.yml` - Full stack (all services)

### Deployment Scripts
- `deploy_instance1.sh` - Deploy Frontend + CPU on Instance 1
- `deploy_instance2.sh` - Deploy GPU Service on Instance 2
- `deploy_cpu.sh` - Deploy CPU Service only
- `deploy_gpu.sh` - Deploy GPU Service only
- `deploy_frontend.sh` - Deploy Frontend only

## Quick Start

### Instance 1 (Frontend + CPU):
```bash
cd docker
chmod +x deploy_instance1.sh
./deploy_instance1.sh
```

### Instance 2 (GPU Only):
```bash
cd docker
chmod +x deploy_instance2.sh
./deploy_instance2.sh
```

## Configuration

Make sure to create `.env` file in the parent directory with your configuration:

```env
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1
S3_BUCKET=your-bucket-name
GPU_INSTANCE_ID=i-1234567890abcdef0
GPU_SERVICE_URL=http://instance2-ip:8001
API_KEY=your-secure-api-key
```

## Notes

- All Docker files expect to be run from the parent directory
- Model file `violence_detection_v4.pt` should be in the parent directory
- Environment variables are loaded from parent directory's `.env` file
