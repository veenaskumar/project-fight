import boto3, botocore, json, uuid, requests
from fastapi import FastAPI, HTTPException, Header, Query
from datetime import datetime
from pathlib import Path
import os
from typing import Optional

app = FastAPI()

# Configuration
S3_BUCKET = os.getenv("S3_BUCKET", "violence-detector-bucket")
GPU_INSTANCE_ID = os.getenv("GPU_INSTANCE_ID")
GPU_SERVICE_URL = os.getenv("GPU_SERVICE_URL")
API_KEY = os.getenv("API_KEY", "default-secret-key")
S3_STREAMS_KEY = "streams/streams.json"
S3_LOGS_KEY = "logs/violence_detection_log.json"

# AWS clients
s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "eu-west-2"))
ec2 = boto3.resource("ec2", region_name=os.getenv("AWS_REGION", "eu-west-2"))

def load_streams_from_s3():
    """Load streams metadata from S3"""
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_STREAMS_KEY)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return {}
        raise e

def save_streams_to_s3(streams):
    """Save streams metadata to S3"""
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=S3_STREAMS_KEY,
        Body=json.dumps(streams, indent=2).encode("utf-8"),
        ContentType="application/json"
    )

def load_logs_from_s3():
    """Load detection logs from S3"""
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_LOGS_KEY)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return []
        return []

def generate_presigned_url(key, expires=86400):
    """Generate presigned URL for S3 object"""
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": key},
            ExpiresIn=expires
        )
    except Exception as e:
        print(f"Presigned URL generation failed for {key}: {e}")
        return None

async def call_gpu_service(endpoint: str, method: str = "post", data: dict = None):
    """Make authenticated call to GPU service"""
    headers = {"X-API-KEY": API_KEY}
    url = f"{GPU_SERVICE_URL}/{endpoint}"
    
    try:
        if method.lower() == "post":
            response = requests.post(url, json=data, headers=headers, timeout=30)
        else:
            response = requests.get(url, headers=headers, timeout=30)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"GPU service call failed: {e}")
        raise HTTPException(status_code=500, detail="GPU service unavailable")

async def manage_gpu_instance():
    """Start or stop GPU instance based on active streams"""
    streams = load_streams_from_s3()
    active_streams = [s for s in streams.values() if s.get("running", False)]
    
    try:
        instance = ec2.Instance(GPU_INSTANCE_ID)
        state = instance.state['Name']
        
        if active_streams and state in ['stopped', 'stopping']:
            # Start GPU instance
            instance.start()
            instance.wait_until_running()
            print("GPU instance started")
            
        elif not active_streams and state == 'running':
            # Stop GPU instance
            instance.stop()
            instance.wait_until_stopped()
            print("GPU instance stopped")
            
    except Exception as e:
        print(f"GPU instance management error: {e}")

@app.post("/add_stream")
async def add_stream(
    name: str = Query(...),
    url_or_file: str = Query(...),
    threshold: float = Query(0.5),
    phone: str = Query(""),
    file_uploaded: bool = Query(False)
):
    """Add a new stream - managed by CPU, processed by GPU"""
    stream_id = str(uuid.uuid4())
    
    # Load existing streams
    streams = load_streams_from_s3()
    
    # Create stream metadata
    streams[stream_id] = {
        "stream_id": stream_id,
        "name": name,
        "url": url_or_file,
        "threshold": threshold,
        "phone": phone,
        "running": True,
        "is_demo": file_uploaded,
        "created_at": datetime.now().isoformat()
    }
    
    # Save to S3
    save_streams_to_s3(streams)
    
    
    # Instruct GPU service to start processing
    try:
        await call_gpu_service("start_stream", data={
            "stream_id": stream_id,
            "name": name,
            "url": url_or_file,
            "threshold": threshold,
            "phone": phone,
            "is_demo": file_uploaded
        })
    except HTTPException:
        # GPU service might not be immediately available
        print("GPU service not ready yet, stream will start when available")
    
    return {"stream_id": stream_id, "status": "started"}

@app.get("/active_streams")
async def get_active_streams():
    """Get list of all streams"""
    streams = load_streams_from_s3()
    return [
        {
            "stream_id": stream_id,
            "name": data["name"],
            "is_demo": data.get("is_demo", False),
            "running": data.get("running", False),
            "threshold": data.get("threshold", 0.5),
            "phone": data.get("phone", "")
        }
        for stream_id, data in streams.items()
    ]

@app.post("/stop_stream/{stream_id}")
async def stop_stream(stream_id: str):
    """Stop a running stream"""
    streams = load_streams_from_s3()
    
    if stream_id not in streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    # Update metadata
    streams[stream_id]["running"] = False
    save_streams_to_s3(streams)
    
    # Instruct GPU service to stop
    try:
        await call_gpu_service("stop_stream", data={"stream_id": stream_id})
    except HTTPException:
        print("GPU service unavailable, but stream metadata updated")
    
    # Manage GPU instance lifecycle
    await manage_gpu_instance()
    
    return {"message": f"Stream {stream_id} stopped"}

@app.post("/start_stream/{stream_id}")
async def start_stream(stream_id: str):
    """Start a stopped stream"""
    streams = load_streams_from_s3()
    
    if stream_id not in streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    # Update metadata
    streams[stream_id]["running"] = True
    save_streams_to_s3(streams)
    
    # Start GPU instance if needed
    await manage_gpu_instance()
    
    # Instruct GPU service to start
    try:
        await call_gpu_service("start_stream", data={
            "stream_id": stream_id,
            **streams[stream_id]
        })
    except HTTPException:
        print("GPU service unavailable, stream will start when available")
    
    return {"message": f"Stream {stream_id} started"}

@app.delete("/delete_stream/{stream_id}")
async def delete_stream(stream_id: str):
    """Delete a stream completely"""
    streams = load_streams_from_s3()
    
    if stream_id not in streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    # Stop stream first
    streams[stream_id]["running"] = False
    
    # Remove from metadata
    del streams[stream_id]
    save_streams_to_s3(streams)
    
    # Instruct GPU service to delete
    try:
        await call_gpu_service("delete_stream", data={"stream_id": stream_id})
    except HTTPException:
        print("GPU service unavailable, but stream metadata deleted")
    
    # Manage GPU instance lifecycle
    await manage_gpu_instance()
    
    return {"message": f"Stream {stream_id} deleted"}

@app.get("/logs")
async def get_logs(stream: Optional[str] = None, sort: str = "desc"):
    """Get detection logs with presigned URLs"""
    logs = load_logs_from_s3()
    
    # Generate presigned URLs
    for entry in logs:
        if entry.get("clip"):
            entry["clip_url"] = generate_presigned_url(entry["clip"])
        if entry.get("snapshot"):
            entry["snapshot_url"] = generate_presigned_url(entry["snapshot"])
    
    # Filter by stream if specified
    if stream:
        logs = [l for l in logs if l.get("stream", "").lower() == stream.lower()]
    
    # Sort by timestamp
    logs.sort(key=lambda x: x.get("timestamp", ""), reverse=(sort == "desc"))
    
    return logs

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "cpu_manager"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)