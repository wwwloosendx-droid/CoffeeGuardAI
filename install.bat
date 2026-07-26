@echo off
echo ☕ Installing CoffeeGuard Dependencies...
echo ==========================================

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install all requirements
pip install -r requirements.txt

echo ==========================================
echo ✅ Installation Complete!
echo Verifying key packages...
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print(f'Ultralytics: {ultralytics.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import flask; print(f'Flask: {flask.__version__}')"

echo ==========================================
echo To activate the virtual environment: venv\Scripts\activate
echo To run the app: python app.py
pause