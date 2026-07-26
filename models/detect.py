import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ... (your model loading code) ...

def detect_coffee_cherries(image, conf=0.25):
    """
    Detect coffee cherries using your best.pt YOLO model
    Returns detailed detection results
    """
    if not MODEL_AVAILABLE or model is None:
        return {
            'success': False,
            'error': 'Model not available',
            'detections': [],
            'total_cherries': 0,
            'class_counts': {'ripe': 0, 'ripening': 0, 'unripe': 0, 'spoilt': 0},
            'has_cherries': False,
            'primary_class': 'unknown',
            'avg_confidence': 0
        }

    try:
        # --- START CRITICAL PREPROCESSING ---
        # 1. Convert to OpenCV format (BGR) and ensure it's a numpy array
        if isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            # If it's a numpy array, assume it's in BGR format from OpenCV
            img = image
        else:
            # Try to read as an image
            img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return {
                'success': False,
                'error': 'Invalid image',
                'detections': [],
                'total_cherries': 0,
                'class_counts': {'ripe': 0, 'ripening': 0, 'unripe': 0, 'spoilt': 0},
                'has_cherries': False,
                'primary_class': 'unknown',
                'avg_confidence': 0
            }

        # 2. Resize the image to the model's expected input size.
        #    This is the most important step! Your model was trained on 640x640.
        #    We'll resize while maintaining aspect ratio by adding padding.
        target_size = 640
        h, w = img.shape[:2]
        
        # Calculate the scaling factor
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize the image
        resized = cv2.resize(img, (new_w, new_h))
        
        # Create a square canvas (target_size x target_size) with black padding
        padded_img = np.full((target_size, target_size, 3), 0, dtype=np.uint8)
        
        # Calculate padding to center the resized image
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        
        # Place the resized image on the padded canvas
        padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # 3. Convert to float and normalize to [0, 1]
        normalized_img = padded_img.astype(np.float32) / 255.0
        
        # 4. Convert HWC to CHW (Height, Width, Channel -> Channel, Height, Width)
        #    YOLO expects [batch, channels, height, width]
        #    PyTorch models expect a tensor, not a numpy array.
        input_tensor = torch.from_numpy(normalized_img).permute(2, 0, 1).unsqueeze(0)
        
        # --- END CRITICAL PREPROCESSING ---

        # Run the detection with the preprocessed tensor
        # The model's inference should handle the tensor directly.
        results = model(input_tensor, conf=conf, iou=0.45)

        # --- Process results and map back to original image coordinates ---
        all_detections = []
        
        # If the result is a list (e.g., from a batch), take the first one.
        # But if it's a Results object, it's handled directly.
        if isinstance(results, list):
            results = results[0]

        if results.boxes is not None:
            # Get the original image dimensions
            orig_h, orig_w = h, w
            
            # Calculate the scaling factor used for the model input
            # The model input is 640x640, but the original was padded.
            # We need to map coordinates back to the original image.
            
            # The scaling factor for the image content itself
            scale_x = orig_w / target_size
            scale_y = orig_h / target_size
            
            # The model's output boxes are in the 640x640 coordinate space.
            # We need to map them back to the original image.
            for box in results.boxes:
                cls_id = int(box.cls)
                confidence = float(box.conf)
                
                # Get the box coordinates in the padded 640x640 space
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Remove the padding and map back to original image coordinates
                # We need to subtract the padding and then scale.
                # The padding was: x_offset, y_offset
                # We'll calculate the original coordinates.
                
                # Map the coordinates to the original image
                # First, remove the padding offset
                orig_x1 = (x1 - x_offset) / scale
                orig_y1 = (y1 - y_offset) / scale
                orig_x2 = (x2 - x_offset) / scale
                orig_y2 = (y2 - y_offset) / scale
                
                # Clamp the coordinates to the original image size
                orig_x1 = max(0, min(orig_w, orig_x1))
                orig_y1 = max(0, min(orig_h, orig_y1))
                orig_x2 = max(0, min(orig_w, orig_x2))
                orig_y2 = max(0, min(orig_h, orig_y2))
                
                all_detections.append({
                    'class': cls_id,
                    'class_name': CLASS_MAP.get(cls_id, f"class_{cls_id}"),
                    'confidence': confidence,
                    'bbox': [orig_x1, orig_y1, orig_x2, orig_y2]
                })

        # --- Count cherries by class ---
        class_counts = {'ripe': 0, 'ripening': 0, 'unripe': 0, 'spoilt': 0}
        for det in all_detections:
            class_name = det['class_name'].lower()
            if class_name in class_counts:
                class_counts[class_name] += 1

        total_cherries = len(all_detections)

        # Determine overall ripeness
        if total_cherries > 0:
            # Get the most confident detection
            best_det = max(all_detections, key=lambda x: x['confidence'])
            primary_class = best_det['class_name'].lower()
            
            # But also consider counts for better accuracy
            if class_counts.get('ripe', 0) > total_cherries * 0.3:
                primary_class = 'ripe'
            elif class_counts.get('ripening', 0) > total_cherries * 0.3:
                primary_class = 'ripening'
            elif class_counts.get('unripe', 0) > total_cherries * 0.3:
                primary_class = 'unripe'
            elif class_counts.get('spoilt', 0) > total_cherries * 0.3:
                primary_class = 'spoilt'
        else:
            primary_class = 'no_cherries'

        # Calculate average confidence
        avg_confidence = sum(d['confidence'] for d in all_detections) / max(len(all_detections), 1)

        return {
            'success': True,
            'total_cherries': total_cherries,
            'class_counts': class_counts,
            'primary_class': primary_class,
            'avg_confidence': avg_confidence * 100,
            'detections': all_detections,
            'has_cherries': total_cherries > 0
        }

    except Exception as e:
        print(f"❌ Detection error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'detections': [],
            'total_cherries': 0,
            'class_counts': {'ripe': 0, 'ripening': 0, 'unripe': 0, 'spoilt': 0},
            'has_cherries': False,
            'primary_class': 'unknown',
            'avg_confidence': 0
        }