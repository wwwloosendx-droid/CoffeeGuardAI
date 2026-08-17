# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create required directories
RUN mkdir -p static/uploads instance

# Expose Flask port
EXPOSE 5000

# Flask environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/').read()"

# The Render container does not ship the trained model, so download it on first
# startup if it is missing. This keeps the app deployable without committing a
# large .pt file to GitHub.
CMD ["sh", "-c", "if [ ! -f best.pt ] && [ ! -f decafia_best.onnx ]; then python download_model.py || true; fi; gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-5000} app:app"]
