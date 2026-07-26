import os
from ultralytics import YOLO

print(f"📁 Current directory: {os.getcwd()}")

# Check files
files = os.listdir()
print(f"\n📄 Files in this folder:")
for f in files:
    if f.endswith('.pt'):
        print(f"   ✅ Found: {f}")
    elif f.endswith('.jpg') or f.endswith('.png'):
        print(f"   📸 Found image: {f}")

# Try loading model
try:
    model = YOLO('best.pt')
    print(f"\n✅ Model loaded successfully!")
    print(f"📋 Classes: {model.names}")
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    print("\n💡 Make sure:")
    print("1. best.pt is in the current folder")
    print("2. You have installed ultralytics")