# ⚡ CoffeeGuardAI Quick Start Guide

Get CoffeeGuardAI up and running in 5 minutes!

---

## 🚀 Quick Start (5 Minutes)

### 1. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model

```bash
python train_cnn_model.py
```

⏳ **Wait ~15-20 minutes** (or ~1 hour on CPU)

You'll see:
```
✓ Model training completed successfully
Model saved to: coffee_cnn_best.pth
Metadata saved to: coffee_cnn_metadata.json
```

### 4. Run Web App

```bash
python app.py
```

### 5. Open in Browser

```
http://localhost:5000
```

---

## 🎯 Next Steps

### Upload an Image

1. Go to `/upload`
2. Select a coffee leaf image
3. Click "Predict Disease"
4. See prediction with confidence score!

### View Predictions

1. Go to `/dashboard`
2. See all previous predictions
3. Check prediction history

### Check Model Info

Visit `/api/model-info` to see model details

---

## 🐳 Docker Quick Start

```bash
# Single command deployment
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Open browser
# http://localhost:5000

# Stop
docker-compose down
```

---

## 📊 Expected Results

After training:
- **Test Accuracy:** ~83%
- **Classes:** 3 disease types (Healthy, Rust, Wilt)
- **Confidence:** Shows prediction confidence %

Example prediction:
```
Disease: Rust
Confidence: 92.34%
Batch: Batch3
```

---

## 🆘 Troubleshooting

### "No module named 'torch'"

```bash
pip install torch torchvision
```

### "Model not found"

```bash
# Make sure you ran training first
python train_cnn_model.py
```

### Port 5000 in use

```bash
# Use different port in app.py
app.run(port=5001)
```

---

## 📚 Full Documentation

For complete setup and deployment guide, see:
- [README.md](README.md) - Full documentation
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Advanced deployment

---

## 🎉 You're All Set!

Your CoffeeGuardAI is ready to use! 🍃

Questions? Check the documentation or create an issue!

---

**Time to First Prediction:** ~20 minutes ⏱️
