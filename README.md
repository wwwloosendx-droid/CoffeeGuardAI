# ☕ CoffeeGuardAI - Coffee Leaf Disease Detection System

A production-ready deep learning system for detecting coffee leaf diseases using ResNet50 CNN with PyTorch.

## 📋 Features

✅ **Advanced CNN Model** - ResNet50 with transfer learning
✅ **Data Augmentation** - 8+ augmentation techniques for robust training
✅ **High Accuracy** - 83%+ test accuracy on coffee disease classification
✅ **Web Interface** - Flask-based web app with upload & prediction
✅ **Database Logging** - SQLite database for prediction history
✅ **Comprehensive Logging** - File and console logging for debugging
✅ **Error Handling** - Robust error handling and validation
✅ **Deployment Ready** - Docker & Docker Compose support
✅ **REST API** - JSON-based prediction API

## 🎯 Diseases Detected

- **Healthy** - Batch 1, 2, 7, 10
- **Rust** - Batch 3, 4, 8
- **Wilt** - Batch 5, 6, 9

---

## 📦 Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd CoffeeGuardAI
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Step 1: Train the CNN Model

```bash
python train_cnn_model.py
```

This will:
- Load images from your dataset
- Apply data augmentation
- Train ResNet50 for 50 epochs
- Save best model as `coffee_cnn_best.pth`
- Save metadata as `coffee_cnn_metadata.json`
- Generate training logs in `training.log`

**Expected Output:**
```
Test Accuracy: 0.8344
Mean CV Accuracy: 0.7231 (+/- 0.0796)
✓ MODEL TRAINING COMPLETED SUCCESSFULLY
```

### Step 2: Run Web Application

```bash
python app.py
```

Open browser to: **http://localhost:5000**

---

## 📡 Deployment Options

### Option 1: Local Deployment (Recommended for Testing)

```bash
python app.py
```

**Requirements:**
- Python 3.8+
- CUDA 11.8 (optional, for GPU acceleration)
- 4GB RAM minimum

### Option 2: Docker Deployment

```bash
# Build image
docker build -t coffee-guard-ai .

# Run container
docker run -p 5000:5000 \
  -v $(pwd)/static/uploads:/app/static/uploads \
  -v $(pwd)/*.pth:/app/ \
  coffee-guard-ai
```

### Option 3: Docker Compose (Recommended)

```bash
# Start application
docker-compose up -d

# View logs
docker-compose logs -f coffee-guard-ai

# Stop application
docker-compose down
```

---

## 📝 API Documentation

### 1. **Predict Disease** (Main Endpoint)

**URL:** `/predict`
**Method:** `POST`
**Content-Type:** `multipart/form-data`

**Request:**
```bash
curl -X POST -F "image=@leaf.jpg" http://localhost:5000/predict
```

**Response:**
```json
{
  "status": "success",
  "prediction": "Rust",
  "batch": "Batch3",
  "confidence": 0.9234,
  "confidence_percent": 92.34
}
```

### 2. **Get Model Info**

**URL:** `/api/model-info`
**Method:** `GET`

**Response:**
```json
{
  "status": "success",
  "model_architecture": "ResNet50",
  "num_classes": 10,
  "test_accuracy": 0.8344,
  "training_date": "2024-06-18 10:30:00"
}
```

### 3. **Get Prediction History**

**URL:** `/api/history`
**Method:** `GET`

**Response:**
```json
{
  "status": "success",
  "history": [
    [1, "leaf1.jpg", "Healthy", "Batch1", 0.95, "2024-06-18 10:35:22"],
    [2, "leaf2.jpg", "Rust", "Batch3", 0.87, "2024-06-18 10:36:15"]
  ]
}
```

---

## 📊 Model Architecture

```
ResNet50 (ImageNet Pretrained)
    ↓
Freeze Early Layers (Transfer Learning)
    ↓
Custom Head:
    - Dropout(0.5)
    - Linear(2048 → 512)
    - ReLU
    - Dropout(0.3)
    - Linear(512 → num_classes)
```

**Training Configuration:**
- Batch Size: 32
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Epochs: 50 (with early stopping)
- Learning Rate Schedule: ReduceLROnPlateau

---

## 🎛️ Data Augmentation Pipeline

During training, images are augmented with:

- Horizontal & Vertical Flips (50% & 30% probability)
- Random Rotation (±30°)
- Color Jitter (brightness, contrast, saturation ±20%)
- Gaussian Blur (σ: 0.1-2.0)
- Random Affine (translation ±10%, scale 0.8-1.2x)
- ImageNet Normalization

This significantly improves model robustness and generalization!

---

## 📁 Project Structure

```
CoffeeGuardAI/
├── train_model.py                 # Old Random Forest trainer
├── train_cnn_model.py            # NEW: CNN trainer with augmentation
├── app.py                         # Flask web app (updated)
├── model.py                       # Model utilities
├── database.py                    # Database utilities
├── generate_graph.py              # Graph visualization
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose configuration
├── training.log                   # Training logs
├── app.log                        # Application logs
├── coffee_cnn_best.pth           # Trained model weights
├── coffee_cnn_metadata.json       # Model metadata
├── predictions.db                 # Prediction history database
├── dataset/                       # Training dataset
├── static/
│   ├── uploads/                  # User uploads
│   └── ...
├── templates/
│   ├── login.html
│   ├── register.html
│   ├── upload.html
│   └── dashboard.html
└── instance/                      # Flask instance folder
```

---

## 🔧 Configuration

### Training Configuration (train_cnn_model.py)

```python
dataset_path = r"C:\Users\hp\Desktop\Coffee and Cashew Nut Dataset\Coffee\Coffee"
batch_size = 32
num_epochs = 50
learning_rate = 0.001
```

### Flask Configuration (app.py)

```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file
app.config['UPLOAD_FOLDER'] = 'static/uploads'
```

---

## 📊 Performance Metrics

### Current Model Performance:
- **Test Accuracy:** 83.44%
- **Cross-Validation Accuracy:** 72.31% ± 7.96%
- **Total Samples:** 3,257
- **Classes:** 10 batches (3 disease types)
- **Training Time:** ~15-20 minutes (GPU) / ~1 hour (CPU)

### Classification Report:
```
              precision    recall  f1-score   support
      Batch1       0.88      0.80      0.84        70
     Batch10       0.89      0.97      0.93        60
      Batch2       0.38      0.27      0.32        11
      Batch3       0.94      1.00      0.97        29
      Batch4       0.91      0.97      0.94        63
      Batch5       0.91      0.92      0.92        78
      Batch6       0.49      0.86      0.62        21
      Batch7       0.80      0.82      0.81       134
      Batch8       0.92      0.79      0.85       128
      Batch9       0.67      0.62      0.64        58

    accuracy                           0.83       652
   macro avg       0.78      0.80      0.78       652
weighted avg       0.84      0.83      0.83       652
```

---

## 🐛 Troubleshooting

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Model File Not Found
```
Error: Model weights file not found!
Solution: Run train_cnn_model.py first
```

### Port Already in Use
```bash
# Change port in app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Docker Build Issues
```bash
# Clear Docker cache
docker system prune -a

# Rebuild
docker-compose up --build
```

---

## 🚀 Production Deployment

### Using Gunicorn (Recommended)

```bash
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using AWS EC2

```bash
# Install Docker
sudo yum install -y docker

# Clone repository
git clone <your-repo>
cd CoffeeGuardAI

# Run with Docker Compose
sudo docker-compose up -d
```

### Using Heroku

```bash
# Create Procfile
echo "web: gunicorn app:app" > Procfile

# Deploy
git push heroku main
```

---

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Documentation](https://docs.docker.com/)

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👨‍💻 Author

**CoffeeGuardAI Team**
- Created with ❤️ for coffee farmers worldwide

---

## 📞 Support

For issues, questions, or contributions:
1. Check existing issues
2. Create a detailed issue report
3. Include logs and error messages
4. Provide steps to reproduce

---

## 🎉 Acknowledgments

- Dataset: Coffee and Cashew Nut Dataset
- Model: ResNet50 (Microsoft Research)
- Framework: PyTorch & Flask Community

---

**Last Updated:** 2024-06-18
**Version:** 2.0.0 (CNN Edition)
