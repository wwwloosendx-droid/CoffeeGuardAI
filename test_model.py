from ultralytics import YOLO
import cv2
import torch

# Check if CUDA (GPU) is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Load the model
model = YOLO('best.pt')
print(f"Model loaded successfully!")
print(f"Classes: {model.names}")

# Test on an image - you can change this path
# Option 1: If you have an image in your folder
# results = model('test.jpg')

# Option 2: Use webcam (press 'q' to quit)
print("Starting webcam detection... Press 'q' to quit")
results = model(0)  # 0 = default webcam

# Display results
for r in results:
    r.show()