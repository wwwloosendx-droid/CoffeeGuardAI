from ultralytics import YOLO
import cv2
from PIL import Image
import requests
from io import BytesIO

# Load model
model = YOLO('best.pt')
print(f"Detecting: {model.names}")

# Download a coffee cherry image
url = "https://images.unsplash.com/photo-1588260403493-08f0e36d0a38?w=600"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Run detection
results = model(img)

# Show results
for r in results:
    # Print detections
    if r.boxes is not None:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            class_name = model.names[int(cls)]
            confidence = r.boxes.conf
            print(f"Detected: {class_name} with confidence {confidence:.2f}")
    
    # Display image with boxes
    annotated = r.plot()
    cv2.imshow('Coffee Detection', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    r.save('coffee_detection_result.jpg')
    print("Result saved as 'coffee_detection_result.jpg'")