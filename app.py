import os

# Keep optional Hugging Face model files inside this project.  The app uses the
# cached copy only, so a slow or unavailable internet connection never delays a
# farmer's prediction request.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('HF_HOME', os.path.join(PROJECT_DIR, '.model_cache'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import sqlite3
import json
import base64
import hashlib
import random
import string
import logging
import io
import csv
import re
import smtplib
from io import BytesIO
from datetime import datetime, timedelta
from contextlib import contextmanager
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import requests
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from dotenv import load_dotenv

import torch
import torch.nn as nn
from ultralytics import YOLO
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    timm = None
    TIMM_AVAILABLE = False

from flask import Flask, render_template, request, redirect, session, jsonify, send_file, make_response
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =========================
# LOAD ENV
# =========================
load_dotenv()

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "predictions.db")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# Email configuration
EMAIL_USER = os.getenv('EMAIL_USER', '')
EMAIL_PASS = os.getenv('EMAIL_PASS', '')

# Your phone number for receiving Mobile Money payments
FARMER_PHONE = os.getenv('FARMER_PHONE', "0759471328")

# News API configuration
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
NEWS_API_URL = "https://newsapi.org/v2/everything"

# =========================
# FLASK APP
# =========================
app = Flask(__name__, template_folder=TEMPLATE_DIR)
# Always give Flask a secret key so the container can boot cleanly even when the
# Render environment has not populated SECRET_KEY yet. The dashboard should still
# set a real value in production for better session security.
app.secret_key = os.getenv('SECRET_KEY') or 'coffee-guard-ai-dev-secret-change-me'
if not os.getenv('SECRET_KEY'):
    print('⚠️ SECRET_KEY not set in environment; using safe fallback for startup only.')

# Session configuration
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', 'false').lower() == 'true'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['MAX_COOKIE_SIZE'] = 8192

# Upload folders
app.config['UPLOAD_FOLDER'] = os.path.join(STATIC_DIR, 'uploads')
app.config['HEATMAP_FOLDER'] = os.path.join(STATIC_DIR, 'heatmaps')
app.config['REPORTS_FOLDER'] = os.path.join(STATIC_DIR, 'reports')
app.config['REFERENCE_FOLDER'] = os.path.join(STATIC_DIR, 'references')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['REFERENCE_FOLDER'], exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp'}

# =========================
# DISEASE REFERENCE DATASET (Updated from Cherry to Disease)
# =========================
class DiseaseReference:
    """Maintains reference color profiles for coffee leaf diseases"""
    
    def __init__(self):
        self.reference_colors = {
            'brown_eye_spot': {
                'hsv_ranges': [(10, 40, 30), (25, 255, 200)],  # Brown lesions with reddish margins
                'rgb_avg': [130, 75, 35],
                'description': 'Brown leaf spots that enlarge and develop reddish-brown margins; repeated lesions may make leaves appear burnt.',
                'emoji': '🍂',
                'treatment': 'Monitor affected leaves, improve field sanitation, and consider an appropriate fungicide if the disease is spreading.'
            },
            'leaf_miner': {
                'hsv_ranges': [(10, 40, 40), (35, 255, 220)],  # Irregular brown/orange mining damage
                'rgb_avg': [145, 95, 50],
                'description': 'Internal leaf mines producing irregular brown or necrotic areas caused by larvae feeding between the leaf surfaces.',
                'emoji': '🐛',
                'treatment': 'Monitor infestation and use integrated pest management, including sanitation and biological controls where available.'
            },
            'leaf_rust': {
                'hsv_ranges': [(5, 80, 100), (25, 255, 255)],  # Pale yellow to orange rust spots
                'rgb_avg': [210, 110, 40],
                'description': 'Pale yellow spots that develop orange/rust-coloured pustules, often on the lower leaf surface.',
                'emoji': '🔥',
                'treatment': 'Apply a recommended fungicide and remove heavily infected leaves where appropriate.'
            }
        }
        self.reference_images = {}
        self._load_reference_images()
    
    def _load_reference_images(self):
        """Load reference images from files or generate defaults"""
        for class_name in self.reference_colors:
            img_path = os.path.join(app.config['REFERENCE_FOLDER'], f'disease_{class_name}.jpg')
            if os.path.exists(img_path):
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        self.reference_images[class_name] = img
                        continue
                except:
                    pass
            self.reference_images[class_name] = self._generate_reference_image(class_name)
    
    def _generate_reference_image(self, class_name):
        """Generate a synthetic reference image for a disease class"""
        color_info = self.reference_colors.get(class_name, {})
        rgb_avg = color_info.get('rgb_avg', [128, 128, 128])
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = rgb_avg
        noise = np.random.randint(-20, 20, (100, 100, 3))
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img
    
    def _hsv_match_ratio(self, patch_hsv, hsv_ranges):
        """Return the fraction of patch pixels inside the expected HSV range."""
        if patch_hsv is None or hsv_ranges is None:
            return 0.0
        lower, upper = hsv_ranges
        h, s, v = cv2.split(patch_hsv)
        hue_mask = (h >= lower[0]) & (h <= upper[0])
        sat_mask = (s >= lower[1]) & (s <= upper[1])
        val_mask = (v >= lower[2]) & (v <= upper[2])
        match = hue_mask & sat_mask & val_mask
        return float(np.count_nonzero(match)) / float(match.size)
    
    def _color_ratio(self, patch_hsv, lower, upper):
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(patch_hsv, lower, upper)
        return float(np.count_nonzero(mask)) / float(mask.size)
    
    def _largest_contour_metrics(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                'area': 0,
                'aspect_ratio': 0,
                'solidity': 0,
                'circularity': 0
            }
        largest = max(contours, key=cv2.contourArea)
        area = max(cv2.contourArea(largest), 0.0)
        if area <= 0:
            return {
                'area': 0,
                'aspect_ratio': 0,
                'solidity': 0,
                'circularity': 0
            }
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = float(w) / float(h) if h > 0 else 0
        hull = cv2.convexHull(largest)
        hull_area = max(cv2.contourArea(hull), 1.0)
        solidity = min(area / hull_area, 1.0)
        perimeter = cv2.arcLength(largest, True)
        circularity = min((4 * np.pi * area) / (perimeter ** 2), 1.0) if perimeter > 0 else 0
        return {
            'area': area,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'circularity': circularity
        }
    
    def _class_feature_score(self, patch, class_name):
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        orange_ratio = self._color_ratio(patch_hsv, (0, 100, 120), (25, 255, 255))
        brown_ratio = self._color_ratio(patch_hsv, (10, 40, 30), (30, 255, 200))
        green_ratio = self._color_ratio(patch_hsv, (30, 30, 40), (90, 255, 200))
        bronzing_ratio = self._color_ratio(patch_hsv, (0, 30, 50), (30, 255, 190))
        edge_density = np.count_nonzero(cv2.Canny(patch_gray, 60, 120)) / float(patch_gray.size)

        orange_mask = cv2.inRange(patch_hsv, np.array((0, 100, 120), dtype=np.uint8), np.array((25, 255, 255), dtype=np.uint8))
        orange_metrics = self._largest_contour_metrics(orange_mask)
        green_mask = cv2.inRange(patch_hsv, np.array((30, 30, 40), dtype=np.uint8), np.array((90, 255, 200), dtype=np.uint8))
        green_metrics = self._largest_contour_metrics(green_mask)
        brown_mask = cv2.inRange(patch_hsv, np.array((10, 40, 30), dtype=np.uint8), np.array((30, 255, 200), dtype=np.uint8))
        brown_metrics = self._largest_contour_metrics(brown_mask)

        hsv_match_score = self._hsv_match_ratio(patch_hsv, self.reference_colors.get(class_name, {}).get('hsv_ranges'))
        shape_bonus = min(edge_density * 1.8, 1.0)

        if class_name == 'brown_eye_spot':
            pale_center = self._color_ratio(patch_hsv, (0, 0, 160), (180, 110, 255))
            dark_ring = self._color_ratio(patch_hsv, (0, 30, 15), (25, 255, 140))
            center_contrast = max(0.0, pale_center - dark_ring * 0.2)
            score = (
                0.30 * brown_ratio +
                0.20 * min(brown_metrics['circularity'] * 1.5, 1.0) +
                0.22 * pale_center +
                0.14 * dark_ring +
                0.10 * center_contrast +
                0.04 * hsv_match_score +
                0.05 * shape_bonus
            )
        elif class_name == 'leaf_miner':
            mine_ratio = self._color_ratio(patch_hsv, (10, 40, 40), (35, 255, 220))
            irregularity = min(1.0, max(0.0, 1.0 - min(green_metrics['solidity'], brown_metrics['solidity'])))
            score = (
                0.35 * mine_ratio +
                0.25 * edge_density +
                0.20 * irregularity +
                0.10 * hsv_match_score +
                0.10 * (1.0 - green_ratio)
            )
        elif class_name == 'leaf_rust':
            score = (
                0.40 * orange_ratio +
                0.20 * min(orange_metrics['circularity'] * 1.2, 1.0) +
                0.15 * edge_density +
                0.10 * min(orange_metrics['solidity'] + 0.1, 1.0) +
                0.10 * hsv_match_score +
                0.05 * shape_bonus
            )
        else:
            score = 0.0

        return float(max(0.0, min(score, 1.0)))
    
    def get_color_distance(self, image_patch, class_name):
        """Calculate how similar an image patch is to a reference class"""
        if class_name not in self.reference_images:
            return float('inf')
        
        ref_img = self.reference_images[class_name]
        if ref_img is None:
            return float('inf')
        
        if image_patch.shape != ref_img.shape:
            patch_resized = cv2.resize(image_patch, (100, 100))
        else:
            patch_resized = image_patch
        
        patch_hsv = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2HSV)
        ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
        
        patch_mean = np.mean(patch_hsv, axis=(0, 1))
        ref_mean = np.mean(ref_hsv, axis=(0, 1))
        
        hue_diff = min(abs(patch_mean[0] - ref_mean[0]), 180 - abs(patch_mean[0] - ref_mean[0]))
        sat_diff = abs(patch_mean[1] - ref_mean[1])
        val_diff = abs(patch_mean[2] - ref_mean[2])
        
        distance = (hue_diff * 2) + (sat_diff * 0.5) + (val_diff * 0.5)
        hsv_ranges = self.reference_colors[class_name].get('hsv_ranges')
        match_ratio = self._hsv_match_ratio(patch_hsv, hsv_ranges)
        if match_ratio < 0.15:
            distance += 45
        elif match_ratio < 0.30:
            distance += 28
        elif match_ratio < 0.50:
            distance += 16
        elif match_ratio < 0.70:
            distance += 8
        else:
            distance -= 10

        feature_score = self._class_feature_score(patch_resized, class_name)
        distance -= feature_score * 32
        distance = max(distance, 0.0)
        distance -= min(match_ratio, 1.0) * 6
        return max(distance, 0.0)
    
    def classify_by_color(self, image_patch):
        """Classify an image patch by color similarity to references"""
        distances = {}
        for class_name in self.reference_colors:
            distances[class_name] = self.get_color_distance(image_patch, class_name)
        
        if not distances:
            return None, float('inf')
        
        best_class = min(distances, key=distances.get)
        return best_class, distances[best_class]
    
    def get_reference_image(self, class_name):
        """Get the reference image for a class"""
        return self.reference_images.get(class_name)
    
    def get_recommendation(self, disease_name):
        """Get treatment recommendation for a disease"""
        disease_name_lower = disease_name.lower().replace(' ', '_')
        return self.reference_colors.get(disease_name_lower, {}).get('treatment', 'Monitor and consult a local agricultural expert.')

disease_reference = DiseaseReference()

# =========================
# ENHANCED IMAGE PREPROCESSING (Same as before)
# =========================
class ImagePreprocessor:
    """Advanced image preprocessing for better detection"""
    
    def __init__(self, target_size=640):
        self.target_size = target_size
    
    def preprocess(self, image):
        """Enhanced preprocessing pipeline"""
        if isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            return None
        
        img = self._enhance_contrast(img)
        img = self._white_balance(img)
        img = self._enhance_saturation(img)
        img = self._resize_and_pad(img)
        return img
    
    def _enhance_contrast(self, img):
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        except:
            return img
    
    def _white_balance(self, img):
        try:
            result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            avg_a = np.mean(result[:, :, 1])
            avg_b = np.mean(result[:, :, 2])
            result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * 0.5)
            result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * 0.5)
            return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        except:
            return img
    
    def _enhance_saturation(self, img):
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = np.clip(s * 1.2, 0, 255).astype(np.uint8)
            hsv = cv2.merge((h, s, v))
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        except:
            return img
    
    def _resize_and_pad(self, img):
        h, w = img.shape[:2]
        target = self.target_size
        
        scale = target / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h))
        
        padded = np.full((target, target, 3), 0, dtype=np.uint8)
        x_offset = (target - new_w) // 2
        y_offset = (target - new_h) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded

preprocessor = ImagePreprocessor(target_size=640)

# =========================
# MODEL LOADING - UPDATED FOR DISEASE DETECTION
# =========================
print("=" * 60)
print("☕ CoffeeGuard AI - Disease Detection Model Loading...")
print("=" * 60)

MODEL_AVAILABLE = False
model = None
CLASS_MAP = {
    0: "leaf_rust",
    1: "brown_eye_spot",
    2: "no_disease",
    3: "leaf_miner"
}
MODEL_CLASS_ALIASES = {
    'roya': 'leaf_rust',
    'minador': 'leaf_miner',
    'sano': 'no_disease',
    'coco': 'brown_eye_spot',
    'healthy': 'no_disease',
    'rust': 'leaf_rust',
    'leaf_rust': 'leaf_rust',
    'leaf_miner': 'leaf_miner',
    'brown_eye_spot': 'brown_eye_spot',
    'no_disease': 'no_disease',
}


def resolve_model_class_name(class_name, cls_id=None):
    """Normalize a model label to the disease names used throughout the app."""
    if class_name is None:
        if cls_id is not None:
            disease_keys = list(DISEASE_LABELS.keys())
            if cls_id < len(disease_keys):
                return disease_keys[cls_id]
            return f'class_{cls_id}'
        return 'unknown'

    normalized = str(class_name).strip().lower()
    normalized = re.sub(r'[^a-z0-9]+', '_', normalized).strip('_')

    if normalized in MODEL_CLASS_ALIASES:
        return MODEL_CLASS_ALIASES[normalized]
    if normalized in DISEASE_LABELS:
        return normalized
    if normalized == 'healthy':
        return 'no_disease'
    if cls_id is not None:
        disease_keys = list(DISEASE_LABELS.keys())
        if cls_id < len(disease_keys):
            return disease_keys[cls_id]
    return normalized

# Detection settings. These match the image size used when best.pt was trained.
# A very low threshold, test-time augmentation and always-on tiling caused the
# same lesion to be reported many times and made false positives look reliable.
# Keep lower-confidence boxes as *possible* symptoms.  Field photos from
# Ugandan gardens commonly contain small, shaded lesions; at 0.25 those boxes
# were silently discarded, leaving a misleading single-lesion result.
DETECTION_CONF = 0.10
CONFIRMED_DETECTION_CONFIDENCE = 0.45
# The exported ONNX model is trained at 640x640. Sending 1280x1280 causes an
# ONNXRuntime invalid-dimension error before detection can run.
INFERENCE_IMAGE_SIZE = 640
LOW_RES_INFERENCE_IMAGE_SIZE = 640
LOW_RES_SOURCE_DIMENSION = 0
TILE_OVERLAP = 0.20
TILE_MIN_DIMENSION = 1400
REFERENCE_DISTANCE_THRESHOLD = 85
# Healthy leaves can trigger a single weak noise box from the model. Raise the
# minimum confidence floor so only genuinely visible lesions survive validation.
VALIDATION_CONF_THRESHOLD = 0.15
MIN_DETECTION_AREA_RATIO = 0.00010
# This model was trained without inference augmentation.  Keep predictions
# deterministic; augmentation can introduce extra boxes for small lesions.
USE_AUGMENTATION = False

CLASS_SPECIFIC_VALIDATION = {
    'brown_eye_spot': {
        'reference_distance': 110,
        'max_reference_offset': 18,
        'min_area_ratio': 0.00005,
        'min_confidence': 0.14,
        'feature_threshold': 0.15,
    },
    'leaf_miner': {
        'reference_distance': 105,
        'max_reference_offset': 16,
        'min_area_ratio': 0.00007,
        'min_confidence': 0.15,
        'feature_threshold': 0.22,
    },
    'leaf_rust': {
        'reference_distance': 120,
        'max_reference_offset': 18,
        'min_area_ratio': 0.00005,
        'min_confidence': 0.15,
        'feature_threshold': 0.28,
    },
    'no_disease': {
        'reference_distance': 90,
        'max_reference_offset': 12,
        'min_area_ratio': 0.00004,
        'min_confidence': 0.15,
        'feature_threshold': 0.08,
    }
}

# Human-readable labels for display
DISEASE_LABELS = {
    'leaf_rust': 'Leaf Rust',
    'brown_eye_spot': 'Brown Eye Spot',
    'no_disease': 'No Disease',
    'leaf_miner': 'Leaf Miner',
    'screening_required': 'Leaf symptoms need agronomist review'
}

# Disease emojis for display
DISEASE_EMOJIS = {
    'leaf_rust': '🔥',
    'brown_eye_spot': '🍂',
    'no_disease': '✅',
    'leaf_miner': '🐛'
}

# Disease severity levels
DISEASE_SEVERITY = {
    'leaf_rust': 'severe',
    'brown_eye_spot': 'moderate',
    'no_disease': 'healthy',
    'leaf_miner': 'moderate'
}

def load_model():
    global model, MODEL_AVAILABLE, CLASS_MAP

    candidate_paths = [
        os.path.join(BASE_DIR, 'best.pt'),
        os.path.join(BASE_DIR, 'best.onnx'),
        os.path.join(BASE_DIR, 'decafia_best.onnx'),
        os.path.join(os.path.expanduser('~'), 'Downloads', 'decafia_best.onnx'),
    ]

    model_path = next((path for path in candidate_paths if os.path.exists(path)), None)

    if model_path is None:
        print("❌ No model file found. Looked for: best.pt, best.onnx, decafia_best.onnx")
        return False

    try:
        model = YOLO(model_path, task='detect')

        raw_class_map = getattr(model, 'names', None)
        if raw_class_map is None and hasattr(model, 'model') and hasattr(model.model, 'names'):
            raw_class_map = model.model.names

        if raw_class_map is not None:
            CLASS_MAP = {
                int(index): resolve_model_class_name(name, int(index))
                for index, name in raw_class_map.items()
            }

        MODEL_AVAILABLE = True
        print(f"✅ Model loaded successfully from: {model_path}")
        print(f"✅ Model classes: {CLASS_MAP}")

        try:
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            model(dummy_img, conf=0.1, imgsz=INFERENCE_IMAGE_SIZE, verbose=False)
            print("✅ Model inference test passed!")
        except Exception as e:
            print(f"⚠️ Model inference test failed: {e}")
            MODEL_AVAILABLE = False

        return MODEL_AVAILABLE

    except ImportError as e:
        print(f"❌ Error importing YOLO: {e}")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

load_model()

# This classifier is a secondary screen for full-leaf photographs when YOLO
# finds no bounding boxes. Its classes differ from best.pt and it is not used
# to fabricate YOLO detections or disease counts.
ARABICA_CLASSIFIER_ID = 'hf-hub:Huyt/arabica-coffee-leaf-disease-efficientnet-b0'
ARABICA_CLASSIFIER_LABELS = {
    'Cerscospora': 'Cercospora / brown-eye-spot-like symptoms',
    'Leaf_rust': 'Coffee leaf rust',
    'Miner': 'Leaf miner',
    'Phoma': 'Phoma leaf spot',
}
SECONDARY_SCREENING_CONFIDENCE = 0.60
secondary_classifier = None
secondary_classifier_transform = None


def screen_leaf_with_secondary_classifier(image):
    """Return a cautious screening alert, or None when screening is unavailable.

    The model works on whole-image classification rather than lesion boxes, so
    it must never be presented as a confirmed field diagnosis.
    """
    global secondary_classifier, secondary_classifier_transform

    if not TIMM_AVAILABLE:
        return None

    try:
        if secondary_classifier is None:
            secondary_classifier = timm.create_model(
                ARABICA_CLASSIFIER_ID,
                pretrained=True,
            ).eval()
            config = timm.data.resolve_data_config({}, model=secondary_classifier)
            secondary_classifier_transform = timm.data.create_transform(**config)

        if isinstance(image, Image.Image):
            source_image = image.convert('RGB')
        else:
            source_image = Image.open(BytesIO(image)).convert('RGB')

        with torch.no_grad():
            probabilities = secondary_classifier(
                secondary_classifier_transform(source_image).unsqueeze(0)
            ).softmax(-1)[0]

        predicted_index = int(probabilities.argmax())
        confidence = float(probabilities[predicted_index])
        labels = secondary_classifier.pretrained_cfg.get('label_names', [])
        predicted_label = labels[predicted_index] if predicted_index < len(labels) else None

        if predicted_label not in ARABICA_CLASSIFIER_LABELS or confidence < SECONDARY_SCREENING_CONFIDENCE:
            return None

        return {
            'label': ARABICA_CLASSIFIER_LABELS[predicted_label],
            'raw_label': predicted_label,
            'confidence': round(confidence * 100, 2),
            'model': 'Arabica leaf symptom screening model',
            'is_screening_only': True,
        }
    except Exception as error:
        # This extra safeguard must not make the primary predictor fail if its
        # optional cached classifier is unavailable.
        print(f"Secondary screening unavailable: {error}")
        return None

def _box_iou(first_box, second_box):
    """Return IoU for two [x1, y1, x2, y2] boxes."""
    x1 = max(first_box[0], second_box[0])
    y1 = max(first_box[1], second_box[1])
    x2 = min(first_box[2], second_box[2])
    y2 = min(first_box[3], second_box[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first_box[2] - first_box[0]) * max(0, first_box[3] - first_box[1])
    second_area = max(0, second_box[2] - second_box[0]) * max(0, second_box[3] - second_box[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0

def _deduplicate_detections(detections, iou_threshold=0.90):
    """Remove same-class duplicates introduced by overlapping inference tiles.
    Use a high IoU threshold so nearby but distinct spots are not merged.
    """
    kept = []
    for detection in sorted(detections, key=lambda item: item['confidence'], reverse=True):
        duplicate = any(
            detection['class_id'] == existing['class_id']
            and _box_iou(detection['bbox'], existing['bbox']) >= iou_threshold
            for existing in kept
        )
        if not duplicate:
            kept.append(detection)
    return kept


def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def is_coffee_leaf_image(image):
    """Use a simple green leaf heuristic to reject non-leaf uploads."""
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, bytes):
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        return False

    if img is None or img.size == 0:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([25, 40, 40]), np.array([90, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([10, 40, 40]), np.array([40, 255, 255]))
    brown_mask = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([35, 255, 180]))

    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    yellow_mask = cv2.morphologyEx(
        yellow_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    brown_mask = cv2.morphologyEx(
        brown_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )

    leaf_mask = cv2.bitwise_or(green_mask, cv2.bitwise_or(yellow_mask, brown_mask))
    leaf_ratio = np.count_nonzero(leaf_mask) / float(img.shape[0] * img.shape[1])
    green_ratio = np.count_nonzero(green_mask) / float(img.shape[0] * img.shape[1])

    if green_ratio < 0.02 and leaf_ratio < 0.08:
        return False
    if leaf_ratio < 0.05:
        return False

    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest_area = max(cv2.contourArea(c) for c in contours)
    return largest_area / float(img.shape[0] * img.shape[1]) >= 0.015

# =========================
# DISEASE DETECTION FUNCTION - UPDATED
# =========================
def detect_diseases(image, conf=DETECTION_CONF):
    """Run best.pt on the original upload and return one count per model box."""
    # Keep this field present in every response, including early failures.
    # Callers use it to distinguish an optional whole-leaf screening alert from
    # a YOLO lesion detection, and a missing key previously caused the upload
    # route to fail while trying to display an otherwise useful error.
    secondary_screening = None
    if not MODEL_AVAILABLE or model is None:
        return {
            'success': False,
            'error': 'Model not available',
            'detections': [],
            'total_detections': 0,
            'class_counts': {
                'leaf_rust': 0,
                'brown_eye_spot': 0,
                'no_disease': 0,
                'leaf_miner': 0
            },
            'has_disease': False,
            'primary_disease': 'no_disease',
            'avg_confidence': 0,
            'reference_validated': False,
            'secondary_screening': secondary_screening,
            'severity': 'healthy',
            'recommendation': 'No diseases detected. Keep monitoring your coffee plants regularly.'
        }
    
    try:
        # Do not apply the old contrast/colour/resize pipeline here.  best.pt
        # performs its own letterboxing, and changing colour balance before
        # inference was causing genuine detections to be lost.
        if isinstance(image, Image.Image):
            source_image = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError('Image must have three colour channels')
            # OpenCV images are BGR; PIL gives YOLO a correctly ordered RGB image.
            source_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            source_image = Image.open(BytesIO(image)).convert('RGB')

        if source_image is None:
            return {
                'success': False,
                'error': 'Invalid image',
                'detections': [],
                'total_detections': 0,
                'class_counts': {
                    'leaf_rust': 0,
                    'brown_eye_spot': 0,
                    'no_disease': 0,
                    'leaf_miner': 0
                },
                'has_disease': False,
                'primary_disease': 'no_disease',
                'avg_confidence': 0,
                'secondary_screening': secondary_screening,
                'severity': 'healthy',
                'recommendation': 'No diseases detected. Keep monitoring your coffee plants regularly.'
            }

        if not is_coffee_leaf_image(source_image):
            return {
                'success': False,
                'error': 'Non-leaf image detected',
                'detections': [],
                'total_detections': 0,
                'class_counts': {
                    'leaf_rust': 0,
                    'brown_eye_spot': 0,
                    'no_disease': 0,
                    'leaf_miner': 0
                },
                'has_disease': False,
                'primary_disease': 'no_disease',
                'avg_confidence': 0,
                'secondary_screening': secondary_screening,
                'severity': 'healthy',
                'recommendation': 'Upload a clear coffee leaf image for disease detection.'
            }
        
        orig_w, orig_h = source_image.size
        # Keep the input at the exported model size. The YOLO ONNX export used in
        # this project is fixed to 640x640; higher sizes trigger invalid ONNX
        # dimensions even when the image content is otherwise valid.
        inference_image_size = INFERENCE_IMAGE_SIZE
        # Most uploads should be evaluated as one photo. Tiling is reserved for
        # exceptionally large images, where a small lesion might otherwise be
        # lost during resizing. Same-class overlap boxes are merged later.
        if min(orig_w, orig_h) >= TILE_MIN_DIMENSION:
            tile_width = int(orig_w * (1 - TILE_OVERLAP * 2))
            tile_height = int(orig_h * (1 - TILE_OVERLAP * 2))
            x_starts = (0, max(0, orig_w - tile_width))
            y_starts = (0, max(0, orig_h - tile_height))
            inference_tiles = [
                (
                    source_image.crop((x_start, y_start, min(orig_w, x_start + tile_width), min(orig_h, y_start + tile_height))),
                    x_start,
                    y_start,
                )
                for y_start in y_starts
                for x_start in x_starts
            ]
        else:
            inference_tiles = [(source_image, 0, 0)]

        all_detections = []
        for tile, x_offset, y_offset in inference_tiles:
            results = model(
                tile,
                conf=conf,
                iou=0.45,
                imgsz=inference_image_size,
                verbose=False,
                augment=USE_AUGMENTATION,
            )
            if not results:
                continue
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    confidence = float(box.conf)

                    raw_class_name = CLASS_MAP.get(cls_id, f"class_{cls_id}")
                    class_name = resolve_model_class_name(raw_class_name, cls_id)

                    if class_name == 'no_disease':
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Ultralytics returns boxes in the original image's pixel
                    # coordinates when it receives a PIL image.
                    orig_x1 = max(0, min(orig_w, x1 + x_offset))
                    orig_y1 = max(0, min(orig_h, y1 + y_offset))
                    orig_x2 = max(0, min(orig_w, x2 + x_offset))
                    orig_y2 = max(0, min(orig_h, y2 + y_offset))
                    
                    all_detections.append({
                        'class_id': cls_id,
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [orig_x1, orig_y1, orig_x2, orig_y2],
                        'reference_validated': False,
                        'display_name': DISEASE_LABELS.get(class_name, class_name),
                        'emoji': DISEASE_EMOJIS.get(class_name, '🔴')
                    })
        
        # Remove overlapping duplicate detections from tiled inference.
        all_detections = _deduplicate_detections(all_detections, iou_threshold=0.45)
        # Validate each detection minimally, but trust the model first.
        image_bgr = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
        validated_detections = []
        for det in all_detections:
            x1, y1, x2, y2 = [int(max(0, val)) for val in det['bbox']]
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)

            class_name_lower = det['class_name'].lower()
            class_settings = CLASS_SPECIFIC_VALIDATION.get(class_name_lower, {})
            min_box_area = max(
                24,
                int(orig_w * orig_h * class_settings.get('min_area_ratio', MIN_DETECTION_AREA_RATIO))
            )
            if width * height < min_box_area:
                continue

            patch = image_bgr[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            # Brown-eye lesions often show a darker ring with a lighter center and
            # more compact circular lesions than rust and miner. Prefer Brown Eye Spot
            # when the patch matches that combination, even if the raw model is close.
            feature_scores = {
                name: disease_reference._class_feature_score(patch, name)
                for name in ['brown_eye_spot', 'leaf_rust', 'leaf_miner']
            }
            brown_score = feature_scores.get('brown_eye_spot', 0.0)
            rust_score = feature_scores.get('leaf_rust', 0.0)
            miner_score = feature_scores.get('leaf_miner', 0.0)
            if (
                brown_score >= 0.32 and
                brown_score >= max(rust_score, miner_score) * 0.98
            ):
                det['class_name'] = 'brown_eye_spot'
                det['class_id'] = 1
                det['display_name'] = DISEASE_LABELS.get('brown_eye_spot', 'Brown Eye Spot')
                det['emoji'] = DISEASE_EMOJIS.get('brown_eye_spot', '🍂')

            ref_class, ref_distance = disease_reference.classify_by_color(patch)
            ref_threshold = class_settings.get('reference_distance', REFERENCE_DISTANCE_THRESHOLD)
            # OpenCV/NumPy comparisons can produce numpy.bool_, which Flask
            # cannot serialize in a JSON response. Convert at the boundary.
            det['reference_validated'] = bool(
                ref_class == det['class_name'] and ref_distance < ref_threshold
            )
            det['reference_label'] = ref_class
            det['reference_distance'] = float(ref_distance)
            det['feature_score'] = disease_reference._class_feature_score(patch, det['class_name'])

            # Ignore weak model noise. Healthy leaves can trigger a single low-score box,
            # but those are not valid disease instances and should not be counted.
            model_confidence_threshold = max(
                VALIDATION_CONF_THRESHOLD,
                class_settings.get('min_confidence', VALIDATION_CONF_THRESHOLD),
                conf * 0.85,
            )
            if det['confidence'] < model_confidence_threshold:
                continue

            validated_detections.append(det)

        all_detections = validated_detections

        # Count actual model detections. Do not infer extra lesions from colour
        # contours inside a box: doing so previously produced counts in the
        # hundreds from a handful of YOLO detections.
        class_counts = {
            'leaf_rust': 0,
            'brown_eye_spot': 0,
            'no_disease': 0,
            'leaf_miner': 0
        }
        
        for det in all_detections:
            class_name = det['class_name'].lower()
            spot_count = 1
            if class_name in class_counts:
                class_counts[class_name] += spot_count
            det['spot_count'] = spot_count
        
        total_detections = len(all_detections)
        
        # A box is a candidate lesion, not automatically a confirmed field
        # diagnosis.  Keep candidates for the farmer to see, but flag the
        # result for review until the model is confident enough.
        review_required = any(
            det['confidence'] < CONFIRMED_DETECTION_CONFIDENCE
            for det in all_detections
        )

        # Determine primary disease
        if total_detections > 0:
            # Find the disease with the highest count
            primary_disease = max(class_counts, key=class_counts.get)
            if class_counts[primary_disease] == 0:
                # If all counts are 0, use the most confident detection
                best_det = max(all_detections, key=lambda x: x['confidence'])
                primary_disease = best_det['class_name'].lower()
                if primary_disease not in class_counts:
                    primary_disease = 'no_disease'
        else:
            primary_disease = 'no_disease'
        
        # A whole-leaf classifier is only a safety net when the lesion detector
        # returns no boxes. It surfaces an alert without converting the alert
        # into a fabricated YOLO box or an exact disease count.
        if total_detections == 0:
            secondary_screening = screen_leaf_with_secondary_classifier(source_image)

        # Determine severity based on actual YOLO detections or a cautious
        # secondary screening alert.
        if total_detections > 0 and review_required:
            severity = 'moderate'
            recommendation = (
                'The model found possible leaf-symptom areas, but at least one '
                'classification is below the confirmation threshold. Inspect both '
                'leaf surfaces and consult an extension officer or agronomist before treatment.'
            )
        elif total_detections == 0 and secondary_screening:
            severity = 'moderate'
            recommendation = (
                f"A secondary visual screen flagged possible {secondary_screening['label']} "
                f"({secondary_screening['confidence']}% confidence). The primary detector did not "
                "recognise a matching lesion pattern, so inspect the plant and seek an agronomist's "
                "confirmation before treatment."
            )
        elif total_detections == 0:
            severity = 'healthy'
            recommendation = 'The model found no matching disease or pest pattern. This is not a guarantee that the leaf is healthy; inspect visible symptoms or consult an agronomist.'
        elif total_detections < 3:
            severity = 'low'
            recommendation = '🟢 Low issue presence detected. Monitor closely and consider preventive measures.'
        elif total_detections < 8:
            severity = 'moderate'
            recommendation = '🟡 Moderate issue presence. Take action to control the spread.'
        else:
            severity = 'severe'
            recommendation = '🔴 High issue presence detected. Immediate action required to protect your crop.'
        
        # Get specific recommendation for the primary disease
        if primary_disease != 'no_disease' and not review_required:
            specific_rec = disease_reference.get_recommendation(primary_disease)
            if specific_rec:
                recommendation = f"**{DISEASE_LABELS.get(primary_disease, primary_disease)} detected.** {specific_rec}"
        
        avg_confidence = sum(d['confidence'] for d in all_detections) / max(len(all_detections), 1)
        reference_validated = any(d.get('reference_validated', False) for d in all_detections)

        # Build debug details for each detection to help troubleshooting
        debug_detections = []
        for d in all_detections:
            debug_detections.append({
                'class_name': d.get('class_name'),
                'confidence': float(d.get('confidence', 0.0)),
                'bbox': [int(x) for x in d.get('bbox', [])],
                'spot_count': int(d.get('spot_count', 1)),
                'feature_score': float(d.get('feature_score', 0.0)),
                'reference_validated': bool(d.get('reference_validated', False)),
            })

        return {
            'success': True,
            'total_detections': total_detections,
            'class_counts': class_counts,
            'primary_disease': primary_disease,
            'display_primary': DISEASE_LABELS.get(primary_disease, primary_disease),
            'avg_confidence': avg_confidence * 100,
            'detections': all_detections,
            'debug_detections': debug_detections,
            'has_disease': total_detections > 0,
            'review_required': review_required,
            'confirmed_detections': sum(
                det['confidence'] >= CONFIRMED_DETECTION_CONFIDENCE
                for det in all_detections
            ),
            'secondary_screening': secondary_screening,
            'reference_validated': reference_validated,
            'severity': severity,
            'recommendation': recommendation,
            'class_labels': {
                'leaf_rust': 'Leaf Rust',
                'brown_eye_spot': 'Brown Eye Spot',
                'no_disease': 'No Disease',
                'leaf_miner': 'Leaf Miner'
            },
            'emojis': DISEASE_EMOJIS
        }
        
    except Exception as e:
        print(f"❌ Disease detection error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'detections': [],
            'total_detections': 0,
            'class_counts': {
                'leaf_rust': 0,
                'brown_eye_spot': 0,
                'no_disease': 0,
                'leaf_miner': 0
            },
            'has_disease': False,
            'primary_disease': 'no_disease',
            'avg_confidence': 0,
            'secondary_screening': secondary_screening,
            'severity': 'healthy',
            'recommendation': 'No diseases detected. Keep monitoring your coffee plants regularly.'
        }

# =========================
# DATABASE HELPER (Same as before)
# =========================
@contextmanager
def get_db():
    conn = None
    try:
        conn = sqlite3.connect(
            DB_PATH,
            timeout=30,
            check_same_thread=False
        )
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

# =========================
# DATABASE INIT - UPDATED FOR DISEASE FIELDS
# =========================
def init_db():
    try:
        with get_db() as conn:
            c = conn.cursor()

            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fullname TEXT,
                    email TEXT UNIQUE,
                    password TEXT,
                    phone TEXT,
                    location TEXT,
                    avatar_data TEXT,
                    created_at TEXT
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT,
                    filename TEXT,
                    result TEXT,
                    confidence REAL,
                    timestamp TEXT,
                    image_data TEXT,
                    disease_count INTEGER DEFAULT 0,
                    class_counts TEXT,
                    total_detections INTEGER DEFAULT 0,
                    detection_type TEXT DEFAULT 'leaf',
                    reference_validated INTEGER DEFAULT 0,
                    severity TEXT DEFAULT 'healthy',
                    recommendation TEXT
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT,
                    report_name TEXT,
                    report_data TEXT,
                    created_at TEXT
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS payments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT,
                    fullname TEXT,
                    phone TEXT,
                    network TEXT,
                    amount REAL,
                    status TEXT,
                    transaction_id TEXT,
                    created_at TEXT
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE,
                    notification TEXT,
                    default_view TEXT,
                    language TEXT,
                    theme TEXT,
                    updated_at TEXT
                )
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS coffee_news_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    news_data TEXT,
                    fetched_at TEXT
                )
            """)

            print("✅ Database initialized successfully!")
            return True

    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        return False

def migrate_database():
    try:
        with get_db() as conn:
            c = conn.cursor()
            
            c.execute("PRAGMA table_info(predictions)")
            columns = [col[1] for col in c.fetchall()]
            
            if 'reference_validated' not in columns:
                print("📌 Adding missing column: reference_validated to predictions table...")
                c.execute("ALTER TABLE predictions ADD COLUMN reference_validated INTEGER DEFAULT 0")
                print("✅ Column 'reference_validated' added successfully!")
            
            if 'detection_type' not in columns:
                print("📌 Adding missing column: detection_type to predictions table...")
                c.execute("ALTER TABLE predictions ADD COLUMN detection_type TEXT DEFAULT 'leaf'")
                print("✅ Column 'detection_type' added successfully!")
            
            if 'class_counts' not in columns:
                print("📌 Adding missing column: class_counts to predictions table...")
                c.execute("ALTER TABLE predictions ADD COLUMN class_counts TEXT")
                print("✅ Column 'class_counts' added successfully!")
            
            if 'disease_count' not in columns:
                print("📌 Adding missing column: disease_count to predictions table...")
                c.execute("ALTER TABLE predictions ADD COLUMN disease_count INTEGER DEFAULT 0")
                print("✅ Column 'disease_count' added successfully!")

            if 'total_detections' not in columns:
                print("📌 Adding missing column: total_detections to predictions table...")
                c.execute("ALTER TABLE predictions ADD COLUMN total_detections INTEGER DEFAULT 0")
                print("✅ Column 'total_detections' added successfully!")
            
            if 'severity' not in columns:
                print("📌 Adding missing column: severity to predictions table...")
                c.execute("ALTER TABLE predictions ADD COLUMN severity TEXT DEFAULT 'healthy'")
                print("✅ Column 'severity' added successfully!")
            
            if 'recommendation' not in columns:
                print("📌 Adding missing column: recommendation to predictions table...")
                c.execute("ALTER TABLE predictions ADD COLUMN recommendation TEXT")
                print("✅ Column 'recommendation' added successfully!")
            
            return True
    except Exception as e:
        print(f"❌ Migration error: {e}")
        return False

init_db()
migrate_database()

# =========================
# HELPER FUNCTIONS - UPDATED FOR DISEASE STATS
# =========================
def get_user_settings(email):
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("SELECT notification, default_view, language, theme FROM settings WHERE email=?", (email,))
            row = c.fetchone()
            if row:
                return {
                    "notification": row[0] or "All Notifications",
                    "default_view": row[1] or "Overview",
                    "language": row[2] or "en",
                    "theme": row[3] or "light"
                }
    except Exception as e:
        print(f"Error getting settings: {e}")

    return {
        "notification": "All Notifications",
        "default_view": "Overview",
        "language": "en",
        "theme": "light"
    }

def save_user_settings(email, data):
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT OR REPLACE INTO settings (email, notification, default_view, language, theme, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (email, data.get("notification", "All Notifications"),
                  data.get("default_view", "Overview"), data.get("language", "en"),
                  data.get("theme", "light"), datetime.now().isoformat()))
            return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

def detect_network(phone):
    digits = "".join(ch for ch in phone if ch.isdigit())
    if digits.startswith("256"):
        digits = digits[3:]
    elif digits.startswith("0"):
        digits = digits[1:]
    prefix = digits[:2] if len(digits) >= 2 else digits
    mtn_prefixes = {"77", "78", "76", "39"}
    airtel_prefixes = {"70", "75", "74", "20"}
    if prefix in mtn_prefixes:
        return "MTN"
    elif prefix in airtel_prefixes:
        return "Airtel"
    return "Unknown"

def send_email(to_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = EMAIL_USER
        msg["To"] = to_email
        msg["Reply-To"] = EMAIL_USER
        msg.attach(MIMEText(body, "plain"))
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.set_debuglevel(0)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"❌ Email error: {e}")
        return False

def get_disease_stats(email):
    """Get disease detection statistics for dashboard"""
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT result, confidence, disease_count, total_detections, detection_type, severity
                FROM predictions
                WHERE email=?
            """, (email,))
            rows = c.fetchall()

            total = len(rows)
            
            # Count by disease type
            leaf_rust = sum(1 for r in rows if r[0] and r[0].lower() == "leaf_rust")
            brown_eye = sum(1 for r in rows if r[0] and r[0].lower() == "brown_eye_spot")
            no_disease = sum(1 for r in rows if r[0] and r[0].lower() == "no_disease")
            leaf_miner = sum(1 for r in rows if r[0] and r[0].lower() == "leaf_miner")

            total_detections = sum(r[3] or 0 for r in rows)
            
            # Count by severity
            healthy = sum(1 for r in rows if r[5] and r[5].lower() == "healthy")
            low_risk = sum(1 for r in rows if r[5] and r[5].lower() == "low")
            moderate_risk = sum(1 for r in rows if r[5] and r[5].lower() == "moderate")
            high_risk = sum(1 for r in rows if r[5] and r[5].lower() == "severe")

            confidences = [r[1] for r in rows if r[1] is not None]
            accuracy = round(np.mean(confidences) * 100, 2) if confidences else 0

            return {
                "total": total,
                "leaf_rust": leaf_rust,
                "brown_eye_spot": brown_eye,
                "no_disease": no_disease,
                "leaf_miner": leaf_miner,
                "accuracy": accuracy,
                "total_detections": total_detections,
                "healthy": healthy,
                "low_risk": low_risk,
                "moderate_risk": moderate_risk,
                "high_risk": high_risk,
                "grand_total": total_detections
            }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {
            "total": 0,
            "leaf_rust": 0,
            "brown_eye_spot": 0,
            "no_disease": 0,
            "leaf_miner": 0,
            "accuracy": 0,
            "total_detections": 0,
            "healthy": 0,
            "low_risk": 0,
            "moderate_risk": 0,
            "high_risk": 0,
            "grand_total": 0
        }

# News functions (same as before)
def get_cached_news():
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("SELECT news_data, fetched_at FROM coffee_news_cache ORDER BY id DESC LIMIT 1")
            row = c.fetchone()
            if row:
                return {"news_data": json.loads(row[0]), "fetched_at": row[1]}
        return None
    except Exception as e:
        print(f"Error getting cached news: {e}")
        return None

def save_news_cache(news_data):
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM coffee_news_cache")
            c.execute("""
                INSERT INTO coffee_news_cache (news_data, fetched_at)
                VALUES (?, ?)
            """, (json.dumps(news_data), datetime.now().isoformat()))
            return True
    except Exception as e:
        print(f"Error saving news cache: {e}")
        return False

def fetch_coffee_news():
    try:
        cached = get_cached_news()
        if cached:
            cache_age = datetime.now() - datetime.fromisoformat(cached["fetched_at"])
            if cache_age < timedelta(minutes=5):
                return cached["news_data"]

        params = {
            "q": "coffee leaf disease Uganda OR Ugandan coffee farming OR coffee rust",
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 15
        }
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            uganda_articles = []
            for article in articles:
                title = article.get("title", "").lower()
                desc = article.get("description", "").lower()
                content = article.get("content", "").lower()
                combined = title + " " + desc + " " + content
                if "uganda" in combined or "ugandan" in combined or "coffee" in combined:
                    uganda_articles.append(article)
            if not uganda_articles:
                uganda_articles = articles[:5]
            if uganda_articles:
                for article in uganda_articles:
                    if not article.get("urlToImage"):
                        article["urlToImage"] = get_fallback_image()
                save_news_cache(uganda_articles)
                return uganda_articles
        if cached:
            return cached["news_data"]
        return get_mock_news_with_images()
    except Exception as e:
        print(f"Error fetching news: {e}")
        cached = get_cached_news()
        if cached:
            return cached["news_data"]
        return get_mock_news_with_images()

def get_fallback_image():
    coffee_images = [
        "https://images.unsplash.com/photo-1447933601403-0c6688de566e?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1559056199-641a0ac8b55e?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1511537190424-bbbab87ac5eb?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1517959100558-1b3bae50cd9f?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=400&h=200&fit=crop"
    ]
    return random.choice(coffee_images)

def get_mock_news_with_images():
    now = datetime.now()
    images = [
        "https://images.unsplash.com/photo-1447933601403-0c6688de566e?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1559056199-641a0ac8b55e?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1511537190424-bbbab87ac5eb?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1517959100558-1b3bae50cd9f?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=400&h=200&fit=crop"
    ]
    return [
        {"title": "Coffee Leaf Rust Outbreak in Uganda - Farmers Urged to Act",
         "description": "Coffee leaf rust has been detected in several regions. Farmers are advised to apply recommended fungicides and remove infected leaves.",
         "url": "https://www.monitor.co.ug/agriculture/coffee-leaf-rust-outbreak",
         "urlToImage": images[0], "publishedAt": (now - timedelta(hours=1)).isoformat(), "source": {"name": "Daily Monitor"}},
        {"title": "Uganda Coffee Exports Surge Despite Disease Challenges",
         "description": "Uganda's coffee exports have reached record levels, with farmers adopting new disease management strategies.",
         "url": "https://www.newvision.co.ug/business/uganda-coffee-exports-surge",
         "urlToImage": images[1], "publishedAt": (now - timedelta(hours=2)).isoformat(), "source": {"name": "New Vision"}},
        {"title": "AI Tool CoffeeGuard Helps Detect Leaf Diseases Early",
         "description": "CoffeeGuard, an AI-powered tool, is helping Ugandan farmers detect coffee leaf diseases early for better crop management.",
         "url": "https://www.ugandacoffee.org/ai-coffeeguard-disease-detection",
         "urlToImage": images[2], "publishedAt": (now - timedelta(hours=3)).isoformat(), "source": {"name": "Uganda Coffee Daily"}},
        {"title": "Integrated Pest Management for Coffee Leaf Miner",
         "description": "Coffee farmers are being trained on integrated pest management to control leaf miner infestations.",
         "url": "https://www.businessfocus.co.ug/ipm-coffee-leaf-miner",
         "urlToImage": images[3], "publishedAt": (now - timedelta(hours=4)).isoformat(), "source": {"name": "Business Focus"}},
        {"title": "Climate Change and Coffee Leaf Diseases in Uganda",
         "description": "Climate change is affecting coffee leaf disease patterns. Farmers are adapting with new farming techniques.",
         "url": "https://www.theugandan.co.ug/climate-change-coffee-diseases",
         "urlToImage": images[4], "publishedAt": (now - timedelta(hours=5)).isoformat(), "source": {"name": "The Ugandan"}}
    ]

# =========================
# FLASK ROUTES (Same structure, updated for disease detection)
# =========================
@app.route('/')
def home():
    if 'email' in session:
        return redirect('/dashboard')
    return redirect('/login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form.get("fullname", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        phone = request.form.get("phone", "").strip()

        if not fullname or len(fullname) < 2:
            return render_template("register.html", error="Please enter your full name (minimum 2 characters).")
        if not email or '@' not in email or '.' not in email:
            return render_template("register.html", error="Please enter a valid email address.", fullname=fullname, phone=phone)
        if not password or len(password) < 6:
            return render_template("register.html", error="Password must be at least 6 characters.", fullname=fullname, email=email, phone=phone)

        try:
            with get_db() as conn:
                c = conn.cursor()
                c.execute("""
                    INSERT INTO users (fullname, email, password, phone, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (fullname, email, password, phone, datetime.now().isoformat()))
            send_email(email, "Welcome to CoffeeGuard! ☕",
                      f"Hello {fullname},\n\nWelcome to CoffeeGuard! 🎉\n\nYou've successfully created your account. Start detecting coffee leaf diseases today!\n\nBest regards,\nThe CoffeeGuard Team")
            return render_template("login.html", success="✅ Account created successfully! Please login.")
        except sqlite3.IntegrityError:
            return render_template("register.html", error="❌ Email already exists. Please use a different email.", fullname=fullname, phone=phone)
        except Exception as e:
            print(f"Registration error: {e}")
            return render_template("register.html", error="❌ An error occurred. Please try again.", fullname=fullname, phone=phone)

    return render_template("register.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/login_user', methods=['POST'])
def login_user():
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '').strip()
    if not email or not password:
        return render_template("login.html", error="⚠️ Please fill in all fields.")
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT fullname, email, phone, location, avatar_data
                FROM users
                WHERE email=? AND password=?
            """, (email, password))
            user = c.fetchone()
            if user:
                session['email'] = user[1]
                session['fullname'] = user[0]
                session['phone'] = user[2] if user[2] else ""
                session['location'] = user[3] if user[3] else "Uganda"
                return redirect('/dashboard')
            return render_template("login.html", error="❌ Invalid email or password. Please try again.")
    except Exception as e:
        print(f"❌ Login error: {e}")
        return render_template("login.html", error="❌ An error occurred. Please try again.")

@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect('/login')
    try:
        stats = get_disease_stats(session['email'])
        avatar_data = None
        try:
            with get_db() as conn:
                c = conn.cursor()
                c.execute("SELECT avatar_data FROM users WHERE email=?", (session['email'],))
                row = c.fetchone()
                if row and row[0]:
                    avatar_data = row[0]
        except:
            pass
        
        # Calculate health status for display
        if stats['total'] == 0:
            health_status = 'Healthy'
        elif stats['high_risk'] > 0:
            health_status = 'High Risk'
        elif stats['moderate_risk'] > 0:
            health_status = 'Moderate Risk'
        elif stats['low_risk'] > 0:
            health_status = 'Low Risk'
        else:
            health_status = 'Healthy'
        
        return render_template(
            "dashboard.html",
            fullname=session.get('fullname', 'Coffee Farmer'),
            email=session.get('email'),
            phone=session.get('phone', ''),
            location=session.get('location', 'Uganda'),
            brown_eye_spot=stats.get('brown_eye_spot', 0),
            leaf_miner=stats.get('leaf_miner', 0),
            leaf_rust=stats.get('leaf_rust', 0),
            no_disease=stats.get('no_disease', 0),
            total=stats.get('total', 0),
            accuracy=stats.get('accuracy', 0),
            total_detections=stats.get('total_detections', 0),
            healthy=stats.get('healthy', 0),
            low_risk=stats.get('low_risk', 0),
            moderate_risk=stats.get('moderate_risk', 0),
            high_risk=stats.get('high_risk', 0),
            health_status=health_status,
            avatar_data=avatar_data,
            session=session,
            model_available=MODEL_AVAILABLE,
            disease_labels=DISEASE_LABELS,
            disease_emojis=DISEASE_EMOJIS
        )
    except Exception as e:
        print(f"Dashboard error: {e}")
        return render_template("dashboard.html", fullname=session.get("fullname", "Coffee Farmer"), session=session)


# Explicit GET handler for the upload page with error logging to capture render failures.
@app.route('/upload', methods=['GET'])
def upload_page():
    try:
        return render_template('upload.html')
    except Exception as e:
        logging.exception("Error rendering upload page")
        return "Internal server error", 500

@app.route('/api/dashboard_stats')
def dashboard_stats():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    try:
        stats = get_disease_stats(session['email'])
        
        # Calculate health status
        if stats['total'] == 0:
            health_status = 'Healthy'
        elif stats['high_risk'] > 0:
            health_status = 'High Risk'
        elif stats['moderate_risk'] > 0:
            health_status = 'Moderate Risk'
        elif stats['low_risk'] > 0:
            health_status = 'Low Risk'
        else:
            health_status = 'Healthy'
        
        return jsonify({
            "leaf_rust": stats['leaf_rust'],
            "brown_eye_spot": stats['brown_eye_spot'],
            "no_disease": stats['no_disease'],
            "leaf_miner": stats['leaf_miner'],
            "total": stats['total'],
            "accuracy": stats['accuracy'],
            "total_detections": stats.get('total_detections', 0),
            "healthy": stats.get('healthy', 0),
            "low_risk": stats.get('low_risk', 0),
            "moderate_risk": stats.get('moderate_risk', 0),
            "high_risk": stats.get('high_risk', 0),
            "health_status": health_status,
            "model_available": MODEL_AVAILABLE
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/coffee_news')
def coffee_news():
    try:
        news = fetch_coffee_news()
        for article in news:
            if not article.get('urlToImage'):
                article['urlToImage'] = get_fallback_image()
        return jsonify(news)
    except Exception as e:
        print(f"Error fetching coffee news: {e}")
        mock_news = get_mock_news_with_images()
        return jsonify(mock_news)

# =========================
# DISEASE DETECTION ENDPOINTS - UPDATED
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_image(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported image format. Please upload a JPG, PNG, BMP, TIFF, or WEBP file."
        }), 400

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        image = Image.open(filepath).convert('RGB')
        if not is_coffee_leaf_image(image):
            return jsonify({
                "success": False,
                "error": "Please upload a coffee leaf image only. Non-leaf images are not allowed."
            }), 400
        
        detection_result = detect_diseases(image)

        secondary_screening = detection_result.get('secondary_screening')
        if detection_result['success'] and detection_result['has_disease']:
            result = detection_result['primary_disease']
            confidence = detection_result['avg_confidence']
            total_detections = detection_result['total_detections']
            class_counts = detection_result['class_counts']
            has_disease = True
            ref_validated = detection_result.get('reference_validated', False)
            severity = detection_result.get('severity', 'healthy')
            recommendation = detection_result.get('recommendation', '')
        elif detection_result['success'] and secondary_screening:
            # Preserve the distinction between a model box and a whole-image
            # screening signal. The count stays zero because no YOLO lesion was
            # located, while the UI clearly explains that review is needed.
            result = 'screening_required'
            confidence = secondary_screening['confidence']
            total_detections = 0
            class_counts = {"leaf_rust": 0, "brown_eye_spot": 0, "no_disease": 0, "leaf_miner": 0}
            has_disease = True
            ref_validated = False
            severity = detection_result.get('severity', 'moderate')
            recommendation = detection_result.get('recommendation', '')
        else:
            result = "no_disease"
            confidence = 0
            total_detections = 0
            class_counts = {"leaf_rust": 0, "brown_eye_spot": 0, "no_disease": 0, "leaf_miner": 0}
            has_disease = False
            ref_validated = False
            severity = 'healthy'
            recommendation = 'No diseases detected. Keep monitoring your coffee plants regularly.'

        confidence = max(0, min(100, confidence))
        confidence_score = round(confidence / 100, 2)

        with open(filepath, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO predictions (
                    email, filename, result, confidence, timestamp,
                    image_data, disease_count, class_counts, total_detections, 
                    detection_type, reference_validated, severity, recommendation
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session['email'], filename, result, confidence_score,
                datetime.now().isoformat(), image_data,
                total_detections, json.dumps(class_counts), total_detections,
                "leaf", 1 if ref_validated else 0, severity, recommendation
            ))

        # Prepare display names
        display_counts = {}
        for key, value in class_counts.items():
            display_counts[DISEASE_LABELS.get(key, key)] = value

        return jsonify({
            "success": True,
            "result": DISEASE_LABELS.get(result, result),
            "result_key": result,
            "confidence": round(confidence, 2),
            "filename": filename,
            "disease_count": total_detections,
            "class_counts": class_counts,
            "display_counts": display_counts,
            "has_disease": has_disease,
            "detection_type": "leaf",
            "model_available": MODEL_AVAILABLE,
            "reference_validated": ref_validated,
            "severity": severity,
            "recommendation": recommendation,
            "total_detected": total_detections,
            "emojis": DISEASE_EMOJIS,
            "message": f"Detected {total_detections} disease instances" if has_disease else "No matching disease pattern was detected by the model"
        })

    except Exception as e:
        print(f"❌ Disease detection error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/validate_image', methods=['POST'])
def validate_image():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        if not allowed_image(file.filename):
            return jsonify({
                "valid": False,
                "error": "Unsupported image format",
                "message": "Please upload a JPG, PNG, BMP, TIFF, or WEBP file."
            }), 400

        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({
                "valid": False,
                "error": "Invalid image format",
                "message": "The image could not be read. Please upload a valid image file."
            }), 400

        if not is_coffee_leaf_image(img):
            return jsonify({
                "valid": False,
                "error": "Non-leaf image detected",
                "message": "Please upload a coffee leaf image only."
            }), 400
        
        detection_result = detect_diseases(img)
        secondary_screening = detection_result.get('secondary_screening')
        
        if detection_result['success'] and (detection_result['has_disease'] or secondary_screening):
            return jsonify({
                "valid": True,
                "disease_count": detection_result['total_detections'],
                "class_counts": detection_result.get('class_counts', {}),
                "severity": detection_result.get('severity', 'healthy'),
                "confidence": (
                    detection_result['avg_confidence']
                    if detection_result['has_disease']
                    else secondary_screening['confidence']
                ),
                "reference_validated": detection_result.get('reference_validated', False),
                "message": (
                    f"✅ {detection_result['total_detections']} disease instances detected!"
                    if detection_result['has_disease']
                    else "Visual symptoms need review before treatment."
                ),
                "has_disease": bool(detection_result['has_disease'] or secondary_screening),
                "primary_disease": detection_result.get('primary_disease', 'no_disease'),
                "recommendation": detection_result.get('recommendation', ''),
                "secondary_screening": secondary_screening,
            })
        else:
            return jsonify({
                "valid": True,  # Still valid as an image, just no disease
                "disease_count": 0,
                "severity": 'healthy',
                "message": "No matching disease pattern was detected by the model.",
                "has_disease": False,
                "primary_disease": 'no_disease',
                "recommendation": 'No diseases detected. Keep monitoring your coffee plants regularly.'
            })
            
    except Exception as e:
        print(f"❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "valid": False,
            "error": str(e),
            "message": f"Error validating image: {str(e)}"
        }), 500

@app.route('/validate_and_predict', methods=['POST'])
def validate_and_predict():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_image(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported image format. Please upload a JPG, PNG, BMP, TIFF, or WEBP file."
        }), 400

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        image = Image.open(filepath).convert('RGB')
        if not is_coffee_leaf_image(image):
            return jsonify({
                "success": False,
                "error": "Please upload a coffee leaf image only. Non-leaf images are not allowed."
            }), 400
        
        detection_result = detect_diseases(image)
        secondary_screening = detection_result.get('secondary_screening')
        
        if detection_result['success'] and detection_result['has_disease']:
            result = detection_result['primary_disease']
            confidence = detection_result['avg_confidence']
            total_detections = detection_result['total_detections']
            class_counts = detection_result['class_counts']
            has_disease = True
            ref_validated = detection_result.get('reference_validated', False)
            severity = detection_result.get('severity', 'healthy')
            recommendation = detection_result.get('recommendation', '')
        elif detection_result['success'] and secondary_screening:
            result = 'screening_required'
            confidence = secondary_screening['confidence']
            total_detections = 0
            class_counts = {"leaf_rust": 0, "brown_eye_spot": 0, "no_disease": 0, "leaf_miner": 0}
            has_disease = True
            ref_validated = False
            severity = detection_result.get('severity', 'moderate')
            recommendation = detection_result.get('recommendation', '')
        else:
            result = "no_disease"
            confidence = 0
            total_detections = 0
            class_counts = {"leaf_rust": 0, "brown_eye_spot": 0, "no_disease": 0, "leaf_miner": 0}
            has_disease = False
            ref_validated = False
            severity = 'healthy'
            recommendation = 'No diseases detected. Keep monitoring your coffee plants regularly.'

        confidence = max(0, min(100, confidence))
        confidence_score = round(confidence / 100, 2)

        with open(filepath, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO predictions (
                    email, filename, result, confidence, timestamp,
                    image_data, disease_count, class_counts, total_detections, 
                    detection_type, reference_validated, severity, recommendation
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session['email'], filename, result, confidence_score,
                datetime.now().isoformat(), image_data,
                total_detections, json.dumps(class_counts), total_detections,
                "leaf", 1 if ref_validated else 0, severity, recommendation
            ))

        response_data = {
            "success": True,
            "result": DISEASE_LABELS.get(result, result),
            "result_key": result,
            "confidence": round(confidence, 2),
            "filename": filename,
            "disease_count": total_detections,
            "class_counts": class_counts,
            "detections": detection_result.get('detections', []),
            "debug_detections": detection_result.get('debug_detections', []),
            "secondary_screening": secondary_screening,
            "review_required": detection_result.get('review_required', False),
            "confirmed_detections": detection_result.get('confirmed_detections', 0),
            "has_disease": has_disease,
            "detection_type": "leaf",
            "model_used": MODEL_AVAILABLE,
            "reference_validated": ref_validated,
            "severity": severity,
            "recommendation": recommendation,
            "total_detected": total_detections,
            "emojis": DISEASE_EMOJIS,
            "severity_emoji": '🟢' if severity == 'healthy' else '🟡' if severity == 'low' else '🟠' if severity == 'moderate' else '🔴',
            "message": (
                f"Detected {total_detections} disease instance(s)"
                if total_detections > 0
                else "Visual screening found symptoms that need review"
                if secondary_screening
                else "No matching disease pattern was detected by the model"
            )
        }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Disease detection error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/predict_multiple', methods=['POST'])
def predict_multiple():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    if 'images' not in request.files:
        return jsonify({"error": "No images uploaded"}), 400
    files = request.files.getlist('images')
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400

    results = []
    rejected = []

    for file in files:
        if file.filename == '' or not allowed_image(file.filename):
            rejected.append(file.filename or 'unnamed')
            continue

        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            image = Image.open(filepath).convert('RGB')
            if not is_coffee_leaf_image(image):
                rejected.append(filename)
                continue

            detection_result = detect_diseases(image)
            secondary_screening = detection_result.get('secondary_screening')

            if detection_result['success'] and detection_result['has_disease']:
                result = detection_result['primary_disease']
                confidence = detection_result['avg_confidence']
                total_detections = detection_result['total_detections']
                class_counts = detection_result['class_counts']
                severity = detection_result.get('severity', 'healthy')
                recommendation = detection_result.get('recommendation', '')
            elif detection_result['success'] and secondary_screening:
                result = 'screening_required'
                confidence = secondary_screening['confidence']
                total_detections = 0
                class_counts = {"leaf_rust": 0, "brown_eye_spot": 0, "no_disease": 0, "leaf_miner": 0}
                severity = detection_result.get('severity', 'moderate')
                recommendation = detection_result.get('recommendation', '')
            else:
                result = "no_disease"
                confidence = 0
                total_detections = 0
                class_counts = {"leaf_rust": 0, "brown_eye_spot": 0, "no_disease": 0, "leaf_miner": 0}
                severity = 'healthy'
                recommendation = 'No matching disease pattern was detected by the model.'

            confidence = max(0, min(100, confidence))
            confidence_score = round(confidence / 100, 2)

            with open(filepath, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            with get_db() as conn:
                c = conn.cursor()
                c.execute("""
                    INSERT INTO predictions (email, filename, result, confidence, timestamp, image_data,
                                             class_counts, total_detections, severity, recommendation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session['email'], filename, result, confidence_score, datetime.now().isoformat(), image_data,
                      json.dumps(class_counts), total_detections, severity, recommendation))

            results.append({
                "success": True,
                "result": DISEASE_LABELS.get(result, result),
                "confidence": round(confidence, 2),
                "filename": filename,
                "disease_count": total_detections,
                "class_counts": class_counts,
                "severity": severity,
                "recommendation": recommendation,
                "secondary_screening": secondary_screening,
                "review_required": detection_result.get('review_required', False),
                "confirmed_detections": detection_result.get('confirmed_detections', 0),
            })
        except Exception as e:
            print(f"Prediction error for {filename}: {e}")
            results.append({"success": False, "error": str(e), "filename": filename})
            rejected.append(filename)

    return jsonify({
        "results": results,
        "rejected": rejected,
        "rejected_count": len(rejected),
        "model_used": MODEL_AVAILABLE
    })

# =========================
# HISTORY & DATA ROUTES - UPDATED
# =========================
@app.route('/history')
def get_history():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT id, filename, result, confidence, timestamp, image_data, 
                       disease_count, class_counts, total_detections, detection_type,
                       reference_validated, severity, recommendation
                FROM predictions
                WHERE email=?
                ORDER BY id DESC
                LIMIT 50
            """, (session['email'],))
            rows = c.fetchall()
        history = []
        for row in rows:
            result_key = row[2] or "no_disease"
            history.append({
                "id": row[0],
                "filename": row[1],
                "result": DISEASE_LABELS.get(result_key, result_key),
                "result_key": result_key,
                "confidence": round(row[3] * 100, 2) if row[3] else 0,
                "timestamp": row[4] or datetime.now().isoformat(),
                "image_data": row[5] if len(row) > 5 else None,
                "disease_count": row[6] if len(row) > 6 else 0,
                "class_counts": json.loads(row[7]) if len(row) > 7 and row[7] else {},
                "total_detections": row[8] if len(row) > 8 else 0,
                "detection_type": row[9] if len(row) > 9 else 'leaf',
                "reference_validated": bool(row[10]) if len(row) > 10 else False,
                "severity": row[11] if len(row) > 11 else 'healthy',
                "recommendation": row[12] if len(row) > 12 else '',
                "emoji": DISEASE_EMOJIS.get(result_key, '🍃')
            })
        return jsonify({"history": history})
    except Exception as e:
        print(f"History error: {e}")
        return jsonify({"history": []})

@app.route('/export_data')
def export_data():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT filename, result, confidence, timestamp, disease_count, total_detections, detection_type, severity
                FROM predictions
                WHERE email=?
                ORDER BY id DESC
            """, (session['email'],))
            rows = c.fetchall()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Filename', 'Disease', 'Confidence (%)', 'Timestamp', 'Disease Count', 'Total Detections', 'Detection Type', 'Severity'])
        for row in rows:
            disease_name = DISEASE_LABELS.get(row[1], row[1]) if row[1] else "No Disease"
            writer.writerow([row[0], disease_name, round(row[2] * 100, 2) if row[2] else 0,
                              row[3] or datetime.now().isoformat(), row[4] or 0, row[5] or 0, row[6] or 'leaf', row[7] or 'healthy'])
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename=disease_predictions_{datetime.now().strftime("%Y%m%d")}.csv'
        response.headers['Content-type'] = 'text/csv'
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# REPORT ROUTES - UPDATED
# =========================
@app.route('/generate_report', methods=['POST'])
def generate_report():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    try:
        stats = get_disease_stats(session['email'])
        report_data = {
            "generated": datetime.now().isoformat(),
            "total_predictions": stats['total'],
            "leaf_rust": stats['leaf_rust'],
            "brown_eye_spot": stats['brown_eye_spot'],
            "no_disease": stats.get('no_disease', 0),
            "leaf_miner": stats['leaf_miner'],
            "avg_confidence": stats['accuracy'],
            "total_detections": stats.get('total_detections', 0),
            "healthy": stats.get('healthy', 0),
            "low_risk": stats.get('low_risk', 0),
            "moderate_risk": stats.get('moderate_risk', 0),
            "high_risk": stats.get('high_risk', 0),
            "model_used": MODEL_AVAILABLE,
            "disease_labels": DISEASE_LABELS,
            "disease_emojis": DISEASE_EMOJIS
        }
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO reports (email, report_name, report_data, created_at)
                VALUES (?, ?, ?, ?)
            """, (session['email'], f"Disease_Report_{datetime.now().strftime('%Y%m%d_%H%M')}", json.dumps(report_data), datetime.now().isoformat()))
        return jsonify({"status": "success", "report": report_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reports')
def view_reports():
    if 'email' not in session:
        return redirect('/login')
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT id, report_name, created_at
                FROM reports
                WHERE email=?
                ORDER BY id DESC
            """, (session['email'],))
            reports = c.fetchall()
        return render_template("reports_list.html", reports=reports, fullname=session.get("fullname"))
    except Exception as e:
        print(f"Error loading reports: {e}")
        return render_template("reports_list.html", reports=[], fullname=session.get("fullname"))

@app.route('/report/<int:report_id>')
def view_report(report_id):
    if 'email' not in session:
        return redirect('/login')
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT report_name, report_data, created_at
                FROM reports
                WHERE id=? AND email=?
            """, (report_id, session['email']))
            row = c.fetchone()
        if not row:
            return "Report not found", 404
        report_data = json.loads(row[1])
        return render_template("report.html", report=report_data, report_name=row[0], fullname=session.get("fullname"))
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/reports_data')
def reports_data():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT id, report_name, report_data, created_at
                FROM reports
                WHERE email=?
                ORDER BY id DESC
            """, (session['email'],))
            rows = c.fetchall()
        reports = []
        for row in rows:
            data = json.loads(row[2])
            reports.append({
                "id": row[0],
                "name": row[1],
                "created_at": row[3],
                "leaf_rust": data.get("leaf_rust", 0),
                "brown_eye_spot": data.get("brown_eye_spot", 0),
                "no_disease": data.get("no_disease", 0),
                "leaf_miner": data.get("leaf_miner", 0),
                "total": data.get("total_predictions", 0),
                "total_detections": data.get("total_detections", 0),
                "healthy": data.get("healthy", 0),
                "low_risk": data.get("low_risk", 0),
                "moderate_risk": data.get("moderate_risk", 0),
                "high_risk": data.get("high_risk", 0)
            })
        return jsonify({"reports": reports})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# CLEAR DASHBOARD
# =========================
@app.route('/clear_dashboard', methods=['POST'])
def clear_dashboard():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM predictions WHERE email=?", (session['email'],))
            c.execute("DELETE FROM reports WHERE email=?", (session['email'],))
            c.execute("DELETE FROM payments WHERE email=?", (session['email'],))
            c.execute("DELETE FROM settings WHERE email=?", (session['email'],))
        return jsonify({"status": "cleared", "message": "All data deleted successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# PAYMENT ROUTES (Same as before)
# =========================
@app.route('/pay', methods=['POST'])
def pay():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    data = request.json or {}
    fullname = data.get("fullname", session.get("fullname", "")).strip()
    phone = data.get("phone", "").strip()
    amount = data.get("amount", 0)
    if not fullname or not phone or not amount or float(amount) <= 0:
        return jsonify({"error": "Please enter your name, phone number, and a valid amount"}), 400
    try:
        network = detect_network(phone)
        transaction_ref = "CG" + datetime.now().strftime("%Y%m%d%H%M%S") + ''.join(random.choices(string.digits, k=4))
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO payments (email, fullname, phone, network, amount, status, transaction_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session['email'], fullname, phone, network, float(amount), "awaiting_transfer", transaction_ref, datetime.now().isoformat()))
        return jsonify({
            "status": "awaiting_transfer",
            "reference": transaction_ref,
            "network": network,
            "merchant_phone": FARMER_PHONE,
            "message": (
                f"To complete your payment of UGX {amount}, open your {network if network != 'Unknown' else 'MTN or Airtel'} "
                f"Money app (or dial *185#), send UGX {amount} to {FARMER_PHONE}, then enter the transaction ID/reference "
                f"you receive below to confirm."
            )
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/confirm_payment', methods=['POST'])
def confirm_payment():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    data = request.json or {}
    reference = data.get("reference", "").strip()
    momo_transaction_id = data.get("momo_transaction_id", "").strip()
    if not reference or not momo_transaction_id:
        return jsonify({"error": "Please provide the payment reference and the MoMo/Airtel transaction ID"}), 400
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT fullname, phone, network, amount FROM payments
                WHERE email=? AND transaction_id=?
            """, (session['email'], reference))
            row = c.fetchone()
            if not row:
                return jsonify({"error": "We couldn't find that payment reference"}), 404
            fullname, phone, network, amount = row
            c.execute("""
                UPDATE payments
                SET status=?, transaction_id=?
                WHERE email=? AND transaction_id=?
            """, ("pending_verification", momo_transaction_id, session['email'], reference))
        subject = f"CoffeeGuard Payment to Verify - UGX {amount}"
        body = f"""
        A CoffeeGuard user has submitted a mobile money transaction for you to verify manually.

        Name: {fullname}
        Phone: {phone}
        Network: {network}
        Amount: UGX {amount}
        MoMo/Airtel Transaction ID: {momo_transaction_id}

        Please check your {network if network != 'Unknown' else 'MTN/Airtel'} Money statement on {FARMER_PHONE}
        for a matching deposit, then mark this payment as confirmed.

        ---
        Sent from CoffeeGuard Dashboard
        """
        send_email(EMAIL_USER, subject, body)
        return jsonify({"status": "pending_verification", "message": "Thanks! Your transaction ID has been submitted and is awaiting manual verification. You'll be notified once confirmed."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# PROFILE ROUTES (Same as before)
# =========================
@app.route('/save_profile', methods=['POST'])
def save_profile():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    data = request.json or {}
    fullname = data.get("fullname", session.get("fullname"))
    email = data.get("email", session['email'])
    phone = data.get("phone", session.get("phone", ""))
    location = data.get("location", session.get("location", "Uganda"))
    avatar_data = data.get("avatar_data", None)
    session['fullname'] = fullname
    session['phone'] = phone
    session['location'] = location
    try:
        with get_db() as conn:
            c = conn.cursor()
            if avatar_data is None:
                c.execute("""
                    UPDATE users
                    SET fullname = ?, phone = ?, location = ?, avatar_data = NULL
                    WHERE email = ?
                """, (fullname, phone, location, session['email']))
            else:
                c.execute("""
                    UPDATE users
                    SET fullname = ?, phone = ?, location = ?, avatar_data = ?
                    WHERE email = ?
                """, (fullname, phone, location, avatar_data, session['email']))
        return jsonify({"status": "success", "message": "Profile updated successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/save_settings', methods=['POST'])
def save_settings():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    data = request.json or {}
    if save_user_settings(session['email'], data):
        session['settings'] = data
        return jsonify({"status": "success", "message": "Settings saved successfully!"})
    else:
        return jsonify({"status": "error", "message": "Failed to save settings"}), 500

@app.route('/api/settings')
def get_settings():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    settings = get_user_settings(session['email'])
    return jsonify(settings)

@app.route('/api/profile')
def get_profile():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("SELECT fullname, email, phone, location, avatar_data FROM users WHERE email=?", (session['email'],))
            user = c.fetchone()
            if user:
                return jsonify({
                    "fullname": user[0],
                    "email": user[1],
                    "phone": user[2] or "",
                    "location": user[3] or "Uganda",
                    "avatar_data": user[4] if user[4] else None
                })
        return jsonify({"error": "user not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# SEND HELP (Same as before)
# =========================
@app.route('/send_help', methods=['POST'])
def send_help():
    data = request.json or {}
    name = data.get("name", "Anonymous")
    email = data.get("email", "no-reply@example.com")
    message = data.get("message", "")
    subject = f"CoffeeGuard Help Request from {name}"
    body = f"""
    New Help Request from CoffeeGuard Dashboard:

    Name: {name}
    Email: {email}

    Message:
    {message}

    ---
    Sent from CoffeeGuard Dashboard
    """
    email_sent = send_email(EMAIL_USER, subject, body)
    if email:
        send_email(email, "CoffeeGuard - We received your request",
                  f"Hello {name},\n\nThank you for contacting CoffeeGuard. We've received your request and will get back to you within 24 hours.\n\nBest regards,\nThe CoffeeGuard Team")
    return jsonify({"status": "sent", "message": "Help request sent!" if email_sent else "Help request logged."})

# =========================
# API STATUS ENDPOINTS - UPDATED
# =========================
@app.route('/api/model_status')
def model_status():
    model_path = os.path.join(BASE_DIR, 'best.pt')
    model_exists = os.path.exists(model_path)
    return jsonify({
        "model_available": MODEL_AVAILABLE,
        "model_exists": model_exists,
        "model_size": f"{os.path.getsize(model_path) / (1024*1024):.2f} MB" if model_exists else None,
        "classes": CLASS_MAP if MODEL_AVAILABLE else None,
        "disease_labels": DISEASE_LABELS,
        "model_path": model_path if model_exists else None,
        "detection_conf": DETECTION_CONF,
        "inference_image_size": INFERENCE_IMAGE_SIZE,
        "reference_count": len(disease_reference.reference_colors),
        "message": "✅ Disease detection model loaded successfully!" if MODEL_AVAILABLE else "❌ Model not loaded. Please check if best.pt exists."
    })

@app.route('/api/reference_status')
def reference_status():
    return jsonify({
        "classes": list(disease_reference.reference_colors.keys()),
        "reference_images": {k: v is not None for k, v in disease_reference.reference_images.items()},
        "total_references": len(disease_reference.reference_images),
        "disease_labels": DISEASE_LABELS,
        "disease_emojis": DISEASE_EMOJIS
    })

@app.route('/api/disease_info')
def disease_info():
    """Get information about all coffee leaf diseases and pests"""
    return jsonify({
        "diseases": [
            {
                "key": "leaf_rust",
                "name": "Leaf Rust",
                "emoji": "🔥",
                "description": "Pale yellow spots that develop orange/rust-coloured pustules, often on the lower leaf surface.",
                "severity": "severe",
                "treatment": disease_reference.get_recommendation("leaf_rust")
            },
            {
                "key": "brown_eye_spot",
                "name": "Brown Eye Spot",
                "emoji": "🍂",
                "description": "Brown leaf spots that enlarge and develop reddish-brown margins; repeated lesions can make leaves appear burnt.",
                "severity": "moderate",
                "treatment": disease_reference.get_recommendation("brown_eye_spot")
            },
            {
                "key": "no_disease",
                "name": "No Disease",
                "emoji": "✅",
                "description": "No disease symptoms detected in this image.",
                "severity": "healthy",
                "treatment": "Keep monitoring the crop and continue normal field hygiene."
            },
            {
                "key": "leaf_miner",
                "name": "Leaf Miner",
                "emoji": "🐛",
                "description": "Irregular brown or necrotic mines caused by larvae feeding between the upper and lower leaf surfaces.",
                "severity": "moderate",
                "treatment": disease_reference.get_recommendation("leaf_miner")
            }
        ],
        "severity_levels": {
            "healthy": "🟢 Healthy - No diseases detected",
            "low": "🟡 Low Risk - Minor disease presence",
            "moderate": "🟠 Moderate Risk - Take action",
            "severe": "🔴 Severe - Immediate action required"
        }
    })

# =========================
# LOGOUT
# =========================
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# =========================
# REFRESH REFERENCES
# =========================
@app.route('/refresh_references', methods=['POST'])
def refresh_references():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    try:
        disease_reference._load_reference_images()
        return jsonify({"status": "success", "message": "Disease references refreshed!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# MAIN - FIXED FOR RENDER DEPLOYMENT
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("☕ CoffeeGuard AI - Disease Detection Running...")
    print("=" * 60)
    
    # Get the port from Render's environment variable
    port = int(os.environ.get("PORT", 5000))
    
    print(f"📍 Running on port: {port}")
    print("=" * 60)
    print("🤖 AI STATUS:", "✅ AI MODEL LOADED!" if MODEL_AVAILABLE else "⚠️ AI model disabled")
    if MODEL_AVAILABLE:
        print("📊 Model classes:", CLASS_MAP)
        print("📁 Model path:", os.path.join(BASE_DIR, 'best.pt'))
        print(f"🎯 Detection confidence threshold: {DETECTION_CONF}")
        print("🔄 Reference validation: Enabled")
        print("📝 Detection: Coffee Leaf Disease Detection")
        print("🦠 Diseases: Leaf Rust, Brown Eye Spot, No Disease, Leaf Miner")
    else:
        print("💡 Please ensure best.pt is in the project directory")
        print("📁 Expected path:", os.path.join(BASE_DIR, 'best.pt'))
    print("=" * 60)
    
    # Production settings - bind to all interfaces and use Render's port
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=False  # IMPORTANT: Debug must be False in production
    )
