from ultralytics import YOLO
import cv2

# Load model
model = YOLO('best.pt')

# Use webcam (0 = default camera)
print("🔄 Starting webcam detection...")
print("⌨️ Press 'q' to quit")

results = model.predict(
    source=0,           # Webcam
    show=True,          # Show results
    conf=0.25,          # Confidence threshold
    save=True,          # Save results
    project='runs',     # Save folder
    name='detect'       # Save subfolder
)