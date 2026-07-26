from ultralytics import YOLO
import cv2
import time
from datetime import datetime

# Load the model
model = YOLO('best.pt')
print(f"Model loaded! Detecting: {model.names}")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Get webcam properties
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Webcam FPS: {fps}")

# Variables for stats
frame_count = 0
total_detections = {'unripe': 0, 'ripening': 0, 'ripe': 0, 'spoilt': 0}
start_time = time.time()

print("\nPress 'q' to quit, 's' to save screenshot")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Run detection with stream=True to prevent memory issues
    results = model(frame, stream=True)
    
    # Process results
    for r in results:
        # Annotate frame with detections
        annotated_frame = r.plot()
        
        # Count detections
        if r.boxes is not None:
            for cls in r.boxes.cls:
                class_name = model.names[int(cls)]
                total_detections[class_name] += 1
        
        # Show FPS on frame
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show detection counts
        y_pos = 60
        for name, count in total_detections.items():
            text = f"{name}: {count}"
            cv2.putText(annotated_frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_pos += 25
        
        # Display the frame
        cv2.imshow('Coffee Cherry Detector', annotated_frame)
    
    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, annotated_frame)
        print(f"Screenshot saved: {filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Print summary
elapsed_time = time.time() - start_time
print(f"\n--- Summary ---")
print(f"Frames processed: {frame_count}")
print(f"Time elapsed: {elapsed_time:.1f} seconds")
print(f"Detections: {total_detections}")