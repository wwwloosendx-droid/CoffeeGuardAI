import os
from ultralytics import YOLO

model_path = 'best.pt'
if not os.path.exists(model_path):
    print(f"❌ Error: {model_path} not found!")
    print(f"📍 Current directory: {os.getcwd()}")
    exit()

print("🔄 Loading model...")
model = YOLO(model_path)
print(f"✅ Model loaded! Classes: {model.names}")

image_path = 'test2.jpg'
if not os.path.exists(image_path):
    print(f"⚠️ {image_path} not found — place a test image in this folder.")
    exit()

# Test at multiple thresholds to see how sensitive detection is
for conf in [0.25, 0.10, 0.05]:
    print(f"\n🔍 Running at conf={conf}...")
    results = model(image_path, conf=conf)
    r = results[0]
    n = 0 if r.boxes is None else len(r.boxes)
    print(f"   → {n} detection(s)")
    if n > 0:
        for box in r.boxes:
            class_name = model.names[int(box.cls)]
            confidence = float(box.conf) * 100
            print(f"     - {class_name}: {confidence:.1f}%")

# Save the lowest-threshold result so you can SEE what it found (or missed)
results = model(image_path, conf=0.05)
results[0].save('output_debug.jpg')
print("\n💾 Saved output_debug.jpg — open it and check if boxes land on real cherries")