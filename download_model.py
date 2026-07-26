# download_model.py
import requests
import os
import sys
from tqdm import tqdm

def download_model():
    """Download the YOLO coffee cherry detection model"""
    print("☕ Downloading CoffeeGuard AI Model...")
    print("=" * 60)
    
    model_path = "best.pt"
    
    # Check if model already exists
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024*1024)
        print(f"✅ Model already exists: {model_path} ({size:.2f} MB)")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != 'y':
            print("Keeping existing model.")
            return True
    
    # Try multiple sources
    sources = [
        {
            "url": "https://huggingface.co/keremberke/yolov8m-coffee/resolve/main/best.pt",
            "name": "Coffee-specific YOLOv8m"
        },
        {
            "url": "https://huggingface.co/akashds/yolov8-coffee/resolve/main/best.pt", 
            "name": "Coffee-specific YOLOv8"
        },
        {
            "url": "https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8n.pt",
            "name": "Generic YOLOv8n (fallback)"
        }
    ]
    
    for source in sources:
        try:
            print(f"\n⬇️ Attempting download from: {source['name']}")
            print(f"   URL: {source['url']}")
            
            response = requests.get(source['url'], stream=True, timeout=60)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                
                print(f"   File size: {total_size / (1024*1024):.2f} MB")
                print("   Downloading...")
                
                with open(model_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc='Progress') as pbar:
                        for data in response.iter_content(block_size):
                            f.write(data)
                            pbar.update(len(data))
                
                print(f"\n✅ Model downloaded successfully: {model_path}")
                print(f"📦 Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
                return True
            else:
                print(f"   ❌ HTTP {response.status_code}: Failed to download")
                
        except requests.exceptions.Timeout:
            print("   ❌ Connection timeout")
        except requests.exceptions.ConnectionError:
            print("   ❌ Connection error - check your internet")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("❌ Could not download model from any source.")
    print("\n📝 Manual download options:")
    print("1. Download from: https://huggingface.co/keremberke/yolov8m-coffee")
    print("2. Save the file as 'best.pt' in the project folder")
    print("3. Or use: https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8n.pt")
    print("=" * 60)
    return False

def test_model():
    """Test if the model loads properly"""
    print("\n🧪 Testing model loading...")
    try:
        from ultralytics import YOLO
        import numpy as np
        
        if not os.path.exists("best.pt"):
            print("❌ Model file not found")
            return False
        
        print("🔄 Loading model...")
        model = YOLO("best.pt")
        
        # Test with dummy image
        print("🔄 Testing inference...")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        results = model(dummy_img, conf=0.1)
        
        print("✅ Model loaded and tested successfully!")
        
        # Show class names
        if hasattr(model, 'names'):
            print(f"\n📋 Model classes: {model.names}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure ultralytics is installed: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("☕ CoffeeGuard Model Downloader")
    print("=" * 60)
    
    # Download the model
    success = download_model()
    
    if success:
        # Test the model
        test_model()
        
        print("\n" + "=" * 60)
        print("✅ Setup complete!")
        print("🚀 Run: python app.py")
    else:
        print("\n" + "=" * 60)
        print("❌ Setup failed. Please try manual download.")
    
    print("=" * 60)