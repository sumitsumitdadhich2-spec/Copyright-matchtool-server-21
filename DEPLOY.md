# Deployment Guide for Google Cloud

This guide explains how to deploy the Video Copyright-Matching Tool on Google Cloud. Since this tool performs CPU-intensive video processing that can take extended periods (especially for full-length movies), we **highly recommend Option A: Google Compute Engine (GCE)**. 

While Cloud Run is a popular serverless option (Option B), it has strict request timeouts (maximum 60 minutes) and limits on background execution, which makes it less suitable for multi-hour extraction jobs.

---

## OPTION A: Google Compute Engine (Recommended)

This option provisions a dedicated 64-vCPU VM to maximize processing speed for long video files.

### 1. Provision a VM
1. Go to the [Google Cloud Console](https://console.cloud.google.com/compute/instances).
2. Click **Create Instance**.
3. **Name:** `video-matching-vm`
4. **Region/Zone:** Choose a region close to you.
5. **Machine configuration:** 
   - Series: **N2** or **E2**
   - Machine type: **n2-standard-64** (64 vCPU, 256 GB RAM) or equivalent to get 64 CPU cores.
6. **Boot disk:**
   - Change to **Ubuntu 22.04 LTS** (or Debian).
   - Increase disk size to **100 GB** or more (video files can be large).
7. **Firewall:**
   - Check **Allow HTTP traffic** (if you want to use port 80).
8. Click **Create**.

### 2. Connect and Install Dependencies
SSH into the newly created VM using the Cloud Console "SSH" button, and run:

```bash
# Update package list and install Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Ensure docker runs without sudo
sudo usermod -aG docker $USER
newgrp docker
```

### 3. Deploy the Application
Clone your repository (or copy your files over using `gcloud compute scp`), then build and run the Docker container:

```bash
# In the directory containing the Dockerfile and docker-compose.yml:
docker-compose up -d --build
```

### 4. Configure Firewall to expose Port 8080 (Optional)
If you didn't configure a reverse proxy to port 80, you must open port 8080:

```bash
gcloud compute firewall-rules create allow-8080 \
    --action=ALLOW \
    --rules=tcp:8080 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=http-server
```
*(Make sure your VM has the `http-server` network tag applied.)*

### 5. Access and Verify
Open your browser and navigate to:
`http://<VM_EXTERNAL_IP>:8080`

**Verify worker scaling logs:**
Check the logs of the running container to ensure it correctly detected all 64 cores:
```bash
docker logs $(docker ps -q -f name=app)
```
You should see:
`Server ready. Detected 64 CPU cores. Worker pool sized to 64 workers.`

---

## OPTION B: Cloud Run (Alternative)

If you only plan to process shorter clips where extraction stays well under the 60-minute limit, you can deploy to Cloud Run.

> **Warning:** Cloud Run is request-driven. The background processing might be throttled or killed if the client disconnects or if the job takes longer than the maximum timeout (3600 seconds).

1. Ensure the Google Cloud SDK is installed and authenticated.
2. Build and deploy using `gcloud`:

```bash
gcloud run deploy video-copyright-tool \
  --source . \
  --port 8080 \
  --allow-unauthenticated \
  --timeout=3600 \
  --cpu=8 \
  --memory=32Gi
```

*(Note: Cloud Run currently caps at 8 vCPUs per instance, so it will be significantly slower than a 64-core Compute Engine VM for video extraction.)*
