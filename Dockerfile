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

# The build environment already has a compatible pip runtime. Avoid upgrading pip
# here because PyPI sometimes returns 502s during the upgrade step, which breaks
# the container build even though the project itself is fine.
RUN python -m pip config set global.retries 5 && \
    python -m pip config set global.timeout 120

# Install Python dependencies with retries to reduce Render/Docker network flakiness.
RUN python -m pip install --no-cache-dir --prefer-binary --retries 5 --timeout 120 -r requirements.txt

# Copy all application files
COPY . .

# Create required directories
RUN mkdir -p static/uploads instance

# Expose Flask port
EXPOSE 5000

# Flask environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Health check (uses Render-assigned PORT if provided)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; port=os.environ.get('PORT','5000'); urllib.request.urlopen(f'http://localhost:{port}/health').read()"

# decafia_best.onnx is the bundled coffee-leaf disease model.  Never download
# or select best.pt here: it is a different coffee-cherry model.
CMD ["sh", "-c", "exec gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-5000} app:app"]
