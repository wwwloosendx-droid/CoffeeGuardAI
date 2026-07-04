# 🚀 CoffeeGuardAI Deployment Guide

Complete step-by-step guide to deploy CoffeeGuardAI on various platforms.

---

## 📋 Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [Training the Model](#training-the-model)
3. [Running Locally](#running-locally)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Production Checklist](#production-checklist)

---

## 🖥️ Local Development Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)
- 4GB RAM minimum
- CUDA 11.8 (optional, for GPU acceleration)

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd CoffeeGuardAI
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** This installs PyTorch CPU version by default. For GPU support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🎓 Training the Model

### Configure Dataset Path

Edit `train_cnn_model.py` and update the dataset path:

```python
dataset_path = r"YOUR_DATASET_PATH"
```

Example:
```python
dataset_path = r"C:\Users\hp\Desktop\Coffee and Cashew Nut Dataset\Coffee\Coffee"
```

### Run Training

```bash
python train_cnn_model.py
```

### Expected Training Output

```
========================================================
STARTING CNN TRAINING FOR COFFEE LEAF CLASSIFICATION
========================================================
Loading dataset from C:\Users\hp\Desktop\...
Loading Batch1
Loading Batch2
...
Total samples loaded: 3257
Class mapping: {'Batch1': 0, 'Batch2': 1, ...}

============================================================
TRAINING STARTED
============================================================
Epoch 1/50
  Train Loss: 2.1234, Train Acc: 0.2456
  Val Loss: 1.8765, Val Acc: 0.4321
  ✓ Best model saved! (Acc: 0.4321)

...

============================================================
TEST SET EVALUATION
============================================================
Test Loss: 0.4567, Test Accuracy: 0.8344

--- Classification Report ---
              precision    recall  f1-score   support
      Batch1       0.88      0.80      0.84        70
      ...

============================================================
✓ MODEL TRAINING COMPLETED SUCCESSFULLY
============================================================
Model saved to: coffee_cnn_best.pth
Metadata saved to: coffee_cnn_metadata.json
Logs saved to: training.log
```

### Generated Files After Training

After successful training, you'll have:

```
CoffeeGuardAI/
├── coffee_cnn_best.pth          # Model weights
├── coffee_cnn_metadata.json      # Model metadata
└── training.log                  # Training logs
```

---

## 🌐 Running Locally

### Start the Application

```bash
python app.py
```

### Access the Web Interface

Open your browser and navigate to:

```
http://localhost:5000
```

### Web Interface Features

1. **Login Page** (`/`) - User authentication
2. **Register Page** (`/register`) - Create new account
3. **Upload Page** (`/upload`) - Upload leaf images
4. **Dashboard** (`/dashboard`) - View predictions and history

### Test Predictions

1. Navigate to `/upload`
2. Select a coffee leaf image
3. Click "Predict Disease"
4. View result with confidence score

### View Logs

**Application Logs:**
```bash
tail -f app.log
```

**Training Logs:**
```bash
tail -f training.log
```

---

## 🐳 Docker Deployment

### Prerequisites

- Docker installed (https://docs.docker.com/install/)
- Docker Compose installed (optional but recommended)

### Option 1: Docker Compose (Recommended)

**Step 1: Build and Run**

```bash
docker-compose up -d
```

**Step 2: Verify Container**

```bash
docker-compose ps
```

**Step 3: Check Logs**

```bash
docker-compose logs -f coffee-guard-ai
```

**Step 4: Access Application**

```
http://localhost:5000
```

**Step 5: Stop Container**

```bash
docker-compose down
```

### Option 2: Manual Docker Build

**Step 1: Build Image**

```bash
docker build -t coffee-guard-ai:latest .
```

**Step 2: Run Container**

```bash
docker run -d \
  --name coffee-guard-ai \
  -p 5000:5000 \
  -v $(pwd)/static/uploads:/app/static/uploads \
  -v $(pwd)/*.pth:/app/*.pth \
  -v $(pwd)/*.json:/app/*.json \
  coffee-guard-ai:latest
```

**Step 3: Check Logs**

```bash
docker logs -f coffee-guard-ai
```

**Step 4: Stop Container**

```bash
docker stop coffee-guard-ai
docker rm coffee-guard-ai
```

### Docker Environment Variables

Create `.env` file:

```env
FLASK_ENV=production
FLASK_APP=app.py
PYTHONUNBUFFERED=1
```

Pass to Docker:

```bash
docker run --env-file .env -p 5000:5000 coffee-guard-ai:latest
```

---

## ☁️ Cloud Deployment

### AWS EC2 Deployment

**Step 1: Launch EC2 Instance**

- AMI: Ubuntu 22.04 LTS
- Instance Type: t3.medium or larger
- Storage: 20GB
- Security Group: Allow HTTP (80), HTTPS (443), Custom (5000)

**Step 2: SSH into Instance**

```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

**Step 3: Install Dependencies**

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git docker.io docker-compose

sudo usermod -aG docker ubuntu
```

**Step 4: Clone Repository**

```bash
git clone <your-repo-url>
cd CoffeeGuardAI
```

**Step 5: Run with Docker Compose**

```bash
sudo docker-compose up -d
```

**Step 6: Setup Reverse Proxy (Nginx)**

```bash
sudo apt install -y nginx

# Create Nginx config
sudo nano /etc/nginx/sites-available/coffee-guard
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/coffee-guard /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Google Cloud Run Deployment

**Step 1: Create Google Cloud Project**

```bash
gcloud projects create coffee-guard-ai
gcloud config set project coffee-guard-ai
```

**Step 2: Build and Push to Container Registry**

```bash
gcloud builds submit --tag gcr.io/coffee-guard-ai/coffee-guard
```

**Step 3: Deploy to Cloud Run**

```bash
gcloud run deploy coffee-guard \
  --image gcr.io/coffee-guard-ai/coffee-guard \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 300 \
  --set-env-vars FLASK_ENV=production
```

### Heroku Deployment

**Step 1: Install Heroku CLI**

```bash
curl https://cli.heroku.com/install.sh | sh
```

**Step 2: Login to Heroku**

```bash
heroku login
```

**Step 3: Create Heroku App**

```bash
heroku create coffee-guard-ai
```

**Step 4: Create Procfile**

```bash
echo "web: gunicorn -w 4 -b 0.0.0.0:\$PORT app:app" > Procfile
```

**Step 5: Deploy**

```bash
git push heroku main
```

**Step 6: View Logs**

```bash
heroku logs --tail
```

---

## 🔒 Production Checklist

- [ ] Update `FLASK_ENV=production` in app.py
- [ ] Set `debug=False` in Flask app
- [ ] Use strong database passwords
- [ ] Enable HTTPS/SSL certificate
- [ ] Setup proper logging
- [ ] Configure firewall rules
- [ ] Setup backup strategy for models and database
- [ ] Configure rate limiting
- [ ] Setup monitoring and alerts
- [ ] Create disaster recovery plan
- [ ] Document API endpoints
- [ ] Setup CI/CD pipeline

### Production Flask Configuration

```python
# app.py
if __name__ == "__main__":
    app.run(
        debug=False,              # Disable debug mode
        host='0.0.0.0',          # Listen on all interfaces
        port=5000,
        use_reloader=False        # Disable auto-reload
    )
```

### Using Gunicorn for Production

```bash
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 app:app

# With logging
gunicorn -w 4 -b 0.0.0.0:5000 \
  --access-logfile - \
  --error-logfile - \
  app:app
```

### Monitoring

**Install Prometheus metrics:**

```bash
pip install prometheus-client
```

**Add to app.py:**

```python
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
```

---

## 📊 Performance Optimization

### For CPU Systems

```bash
# Install optimized PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Run with multiple threads
export OMP_NUM_THREADS=4
python app.py
```

### For GPU Systems

```bash
# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# PyTorch automatically uses GPU
# No additional configuration needed!
```

### Docker GPU Support

```bash
# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Run container with GPU
docker run --gpus all -p 5000:5000 coffee-guard-ai:latest
```

---

## 🐛 Troubleshooting

### Model Loading Issues

```
Error: Model weights file not found!
```

**Solution:**
```bash
# Ensure training is completed
python train_cnn_model.py

# Check files exist
ls -la *.pth *.json
```

### Port Already in Use

```bash
# Find process using port 5000
lsof -i :5000

# Kill process
kill -9 <PID>

# Or change port in app.py
```

### Out of Memory

```bash
# Reduce batch size in train_cnn_model.py
batch_size = 16  # Default is 32

# Or use gradient accumulation
```

### Docker Build Fails

```bash
# Clear Docker cache
docker system prune -a

# Rebuild
docker-compose up --build
```

---

## 📞 Support & Help

For issues or questions:

1. Check logs: `docker-compose logs -f`
2. Review README.md
3. Check GitHub issues
4. Create detailed bug report

---

**Last Updated:** 2024-06-18
**Version:** 2.0.0
