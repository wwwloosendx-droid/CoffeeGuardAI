from ultralytics import YOLO
import cv2

# Load your model
model = YOLO('best.pt')  # Make sure best.pt is in same folder

# Run detection on an image
results = model('path/to/your/image.jpg')  # Replace with your image path

# Show results
for r in results:
    r.show()  # Displays the image with bounding boxes
    print(r.boxes)  # Print detection details