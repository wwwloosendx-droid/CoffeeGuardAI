#!/usr/bin/env python
"""
CoffeeGuard Setup Script
Automatically installs all required dependencies
"""

import subprocess
import sys
import os

def install_packages():
    """Install all required packages"""
    
    packages = [
        # Core
        'Flask==2.3.3',
        'Flask-Cors==4.0.0',
        'Flask-Limiter==3.3.1',
        
        # AI/ML
        'torch==2.12.1',
        'torchvision==0.27.1',
        'ultralytics==8.0.196',
        'opencv-python==4.8.0.74',
        'opencv-python-headless==4.8.0.74',
        'Pillow==10.0.0',
        'numpy==2.5.0',
        'scikit-learn==1.9.0',
        'matplotlib==3.7.2',
        
        # Database
        'sqlalchemy==2.0.19',
        'bcrypt==4.0.1',
        
        # Utilities
        'python-dotenv==1.0.0',
        'requests==2.31.0',
        'tqdm==4.65.0',
        
        # Image Processing
        'imageio==2.31.1',
        'scipy==1.11.1',
        
        # Web Scraping
        'beautifulsoup4==4.12.2',
        'lxml==4.9.3',
        
        # Additional
        'pandas==2.0.3',
        'seaborn==0.12.2',
        'plotly==5.15.0'
    ]
    
    print("☕ Installing CoffeeGuard Dependencies...")
    print("=" * 60)
    
    for package in packages:
        print(f"📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")
    
    print("=" * 60)
    print("✅ Installation Complete!")
    print("Verifying key packages...")
    
    # Verify installations
    verify_installations()

def verify_installations():
    """Verify key packages are installed"""
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch: Not installed")
    
    try:
        import ultralytics
        print(f"✅ Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics: Not installed")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV: Not installed")
    
    try:
        import flask
        print(f"✅ Flask: {flask.__version__}")
    except ImportError:
        print("❌ Flask: Not installed")
    
    print("=" * 60)
    print("To run the app: python app.py")

if __name__ == "__main__":
    install_packages()