# Project Cleanup Summary

## ✅ Cleaned Up Project Structure

### **Main Directory (Root)**
```
project-fight/
├── CPU_Server.py              # CPU service (manager)
├── GUP_server.py              # GPU service (worker)  
├── index.py                   # Streamlit frontend
├── violence_detection_v4.pt   # YOLO model file (KEEP THIS ONE)
├── logo.png                   # Frontend logo
├── styles.css                 # Frontend styles
├── requirements.txt           # Python dependencies
├── env.example               # Environment template
├── README.md                 # Project documentation
└── docker/                   # Docker deployment files
```

### **Docker Directory**
```
docker/
├── README.md                 # Docker documentation
├── Dockerfile.cpu            # CPU service container
├── Dockerfile.gpu            # GPU service container
├── Dockerfile.frontend       # Frontend container
├── docker-compose.frontend-cpu.yml  # Instance 1 deployment
├── docker-compose.gpu.yml    # Instance 2 deployment
├── docker-compose.cpu.yml    # CPU only deployment
├── docker-compose.yml        # Full stack deployment
├── deploy_instance1.sh       # Deploy Frontend + CPU
├── deploy_instance2.sh       # Deploy GPU only
├── deploy_cpu.sh             # Deploy CPU only
├── deploy_gpu.sh             # Deploy GPU only
└── deploy_frontend.sh        # Deploy Frontend only
```

## 🗑️ Files Removed

### **Old/Unused Files:**
- `app.py` - Old Streamlit app
- `server.py` - Old single server
- `start.sh`, `start_cpu.sh`, `start_gpu.sh` - Old startup scripts
- `test_*.py` - Test files
- `violence_detection_log.json` - Old log file
- `annotated_video*.mp4` - Old video files

### **Old Model Files (kept only violence_detection_v4.pt):**
- `fight_detection-m9aq1_step2 (1).pt`
- `fina_violenec_fall.pt`
- `final_fight_fall.pt`
- `final_fight_updated.pt`
- `final_fight.pt`
- `final_violenec_fall_version5.pt`
- `rwf-2000_step1.pt`
- `violence_detection_v2.pt`
- `violence_detection_v3.pt`
- `violence-detection-through-cctv_step3.pt`

### **Old Documentation:**
- `DEPLOYMENT_GUIDE.md`
- `DOCKER_DEPLOYMENT.md`
- `TWO_INSTANCE_DEPLOYMENT.md`

## 🚀 Quick Deployment

### **Instance 1 (Frontend + CPU):**
```bash
cd docker
chmod +x deploy_instance1.sh
./deploy_instance1.sh
```

### **Instance 2 (GPU Only):**
```bash
cd docker
chmod +x deploy_instance2.sh
./deploy_instance2.sh
```

## 📋 Next Steps

1. **Create .env file** from `env.example`
2. **Configure your AWS credentials** and instance IDs
3. **Deploy on Instance 1** using `docker/deploy_instance1.sh`
4. **Deploy on Instance 2** using `docker/deploy_instance2.sh`
5. **Access frontend** at `http://instance1-ip:8501`

The project is now clean and organized with all Docker files in a separate folder!
