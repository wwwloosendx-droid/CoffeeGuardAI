from flask import Flask, render_template, request, redirect, session, jsonify, send_file, make_response
import os
import sqlite3
import logging
import smtplib
import numpy as np
import csv
import io
import json
import base64
import random
import string
from io import BytesIO
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
EMAIL_USER = "loosendx@gmail.com"
EMAIL_PASS = "frnn lzpn jstz oqqj"  # Gmail App Password

# Your phone number for receiving Mobile Money payments
FARMER_PHONE = "0759471328"

# News API configuration
NEWS_API_KEY = "19a58bf976a44e8689759ed06b302dc8"
NEWS_API_URL = "https://newsapi.org/v2/everything"

# =========================
# FLASK APP
# =========================
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.secret_key = "coffee_guard_ai_secret"

# Set session cookie limits
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['MAX_COOKIE_SIZE'] = 8192

app.config['UPLOAD_FOLDER'] = os.path.join(STATIC_DIR, 'uploads')
app.config['HEATMAP_FOLDER'] = os.path.join(STATIC_DIR, 'heatmaps')
app.config['REPORTS_FOLDER'] = os.path.join(STATIC_DIR, 'reports')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# =========================
# MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CoffeeLeafCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

CLASS_MAP = {0: "ripe", 1: "ripening", 2: "unripe"}

model = CoffeeLeafCNN(len(CLASS_MAP)).to(device)

model_path = os.path.join(BASE_DIR, "coffee_cnn_best.pth")
if os.path.exists(model_path):
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Model loading error: {e}")
        print("⚠️ Using untrained model (predictions will be random)")
else:
    print("⚠️ Model file not found. Using untrained model.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =========================
# DATABASE INIT
# =========================
def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("PRAGMA foreign_keys = ON")

        # Users table
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

        # Predictions table
        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                filename TEXT,
                result TEXT,
                confidence REAL,
                timestamp TEXT,
                image_data TEXT
            )
        """)

        # Reports table
        c.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                report_name TEXT,
                report_data TEXT,
                created_at TEXT
            )
        """)

        # Payments table
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

        # Settings table
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

        # Coffee news cache table
        c.execute("""
            CREATE TABLE IF NOT EXISTS coffee_news_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                news_data TEXT,
                fetched_at TEXT
            )
        """)

        conn.commit()
        conn.close()
        print("✅ Database initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        return False

# Initialize database
init_db()

# =========================
# HELPER FUNCTIONS
# =========================
def get_user_settings(email):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT notification, default_view, language, theme FROM settings WHERE email=?", (email,))
        row = c.fetchone()
        conn.close()
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
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO settings (email, notification, default_view, language, theme, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (email, data.get("notification", "All Notifications"), 
              data.get("default_view", "Overview"), data.get("language", "en"),
              data.get("theme", "light"), datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False

def detect_network(phone):
    """Detects MTN vs Airtel from a Ugandan phone number prefix."""
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
    """Send email using Gmail SMTP with proper authentication."""
    try:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = EMAIL_USER
        msg["To"] = to_email
        msg["Reply-To"] = EMAIL_USER
        
        msg.attach(MIMEText(body, "plain"))
        
        print(f"📧 Sending email to {to_email}...")
        
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.set_debuglevel(0)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"❌ Email error: {e}")
        return False

def get_predictions_stats(email):
    """Get prediction statistics for a user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT result, confidence FROM predictions WHERE email=?", (email,))
        rows = c.fetchall()
        conn.close()
        
        total = len(rows)
        ripe = sum(1 for r in rows if r[0] and r[0].lower() == "ripe")
        unripe = sum(1 for r in rows if r[0] and r[0].lower() == "unripe")
        ripening = sum(1 for r in rows if r[0] and r[0].lower() == "ripening")
        spoilt = sum(1 for r in rows if r[0] and r[0].lower() == "spoilt")
        
        confidences = [r[1] for r in rows if r[1] is not None]
        accuracy = round(np.mean(confidences) * 100, 2) if confidences else 0
        
        return {
            "total": total,
            "ripe": ripe,
            "unripe": unripe,
            "ripening": ripening,
            "spoilt": spoilt,
            "accuracy": accuracy
        }
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {"total": 0, "ripe": 0, "unripe": 0, "ripening": 0, "spoilt": 0, "accuracy": 0}

def get_cached_news():
    """Get cached coffee news from database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT news_data, fetched_at FROM coffee_news_cache ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        conn.close()
        
        if row:
            return {
                "news_data": json.loads(row[0]),
                "fetched_at": row[1]
            }
        return None
    except Exception as e:
        print(f"Error getting cached news: {e}")
        return None

def save_news_cache(news_data):
    """Save coffee news to cache."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM coffee_news_cache")
        c.execute("""
            INSERT INTO coffee_news_cache (news_data, fetched_at)
            VALUES (?, ?)
        """, (json.dumps(news_data), datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving news cache: {e}")
        return False

def fetch_coffee_news():
    """Fetch coffee news from NewsAPI or return cached data with images."""
    try:
        cached = get_cached_news()
        if cached:
            cache_age = datetime.now() - datetime.fromisoformat(cached["fetched_at"])
            if cache_age < timedelta(minutes=5):
                return cached["news_data"]
        
        params = {
            "q": "coffee Uganda OR Ugandan coffee",
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
                if "uganda" in combined or "ugandan" in combined:
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
    """Return a fallback coffee image URL."""
    coffee_images = [
        "https://images.unsplash.com/photo-1447933601403-0c6688de566e?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1559056199-641a0ac8b55e?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1511537190424-bbbab87ac5eb?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1517959100558-1b3bae50cd9f?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=400&h=200&fit=crop"
    ]
    return random.choice(coffee_images)

def get_mock_news_with_images():
    """Generate mock coffee news data with images as fallback."""
    now = datetime.now()
    images = [
        "https://images.unsplash.com/photo-1447933601403-0c6688de566e?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1559056199-641a0ac8b55e?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1511537190424-bbbab87ac5eb?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1517959100558-1b3bae50cd9f?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=400&h=200&fit=crop",
        "https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=400&h=200&fit=crop"
    ]
    return [
        {
            "title": "Uganda Coffee Exports Surge in 2026",
            "description": "Uganda's coffee exports have reached record levels, with the Uganda Coffee Development Authority reporting a 15% increase in export volumes.",
            "url": "https://www.monitor.co.ug/uganda/business/commodities/uganda-coffee-exports-surge-2026",
            "urlToImage": images[0],
            "publishedAt": (now - timedelta(hours=1)).isoformat(),
            "source": {"name": "Daily Monitor"}
        },
        {
            "title": "Climate-Smart Coffee Farming in Rwenzori",
            "description": "Farmers in the Rwenzori region are adopting climate-smart practices including shade-grown coffee and organic composting.",
            "url": "https://www.newvision.co.ug/agriculture/climate-smart-coffee-farming-rwenzori",
            "urlToImage": images[1],
            "publishedAt": (now - timedelta(hours=2)).isoformat(),
            "source": {"name": "New Vision"}
        },
        {
            "title": "AI Tool CoffeeGuard Transforming Harvest Decisions",
            "description": "Ugandan coffee farmers are using AI-powered tools like CoffeeGuard to accurately determine cherry ripeness.",
            "url": "https://www.ugandacoffee.org/ai-coffeeguard-transforming-harvest",
            "urlToImage": images[2],
            "publishedAt": (now - timedelta(hours=3)).isoformat(),
            "source": {"name": "Uganda Coffee Daily"}
        },
        {
            "title": "Modern Coffee Washing Stations in Uganda",
            "description": "New solar-powered coffee washing stations are being installed across Uganda, reducing water usage and energy costs.",
            "url": "https://www.businessfocus.co.ug/modern-coffee-washing-stations-uganda",
            "urlToImage": images[3],
            "publishedAt": (now - timedelta(hours=4)).isoformat(),
            "source": {"name": "Business Focus"}
        },
        {
            "title": "Youth Embrace Coffee Farming in Uganda",
            "description": "More young Ugandans are taking up coffee farming as a career, driven by access to technology and training programs.",
            "url": "https://www.theugandan.co.ug/youth-embrace-coffee-farming",
            "urlToImage": images[4],
            "publishedAt": (now - timedelta(hours=5)).isoformat(),
            "source": {"name": "The Ugandan"}
        }
    ]

# =========================
# ROUTES
# =========================

@app.route('/')
def home():
    if 'email' in session:
        return redirect('/dashboard')
    return redirect('/login')

# =========================
# REGISTER
# =========================
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
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()

            c.execute("""
                INSERT INTO users (fullname, email, password, phone, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (fullname, email, password, phone, datetime.now().isoformat()))
            conn.commit()
            conn.close()
            
            send_email(email, "Welcome to CoffeeGuard! ☕", 
                      f"Hello {fullname},\n\nWelcome to CoffeeGuard! 🎉\n\nYou've successfully created your account.\n\nBest regards,\nThe CoffeeGuard Team")
            
            return render_template("login.html", success="✅ Account created successfully! Please login.")
            
        except sqlite3.IntegrityError:
            return render_template("register.html", error="❌ Email already exists. Please use a different email.", fullname=fullname, phone=phone)
        except Exception as e:
            print(f"Registration error: {e}")
            return render_template("register.html", error="❌ An error occurred. Please try again.", fullname=fullname, phone=phone)

    return render_template("register.html")

# =========================
# LOGIN
# =========================
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
        conn = sqlite3.connect(DB_PATH)
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
            
            conn.close()
            return redirect('/dashboard')
        
        conn.close()
        return render_template("login.html", error="❌ Invalid email or password. Please try again.")
        
    except Exception as e:
        print(f"❌ Login error: {e}")
        return render_template("login.html", error="❌ An error occurred. Please try again.")

# =========================
# DASHBOARD - Main page with full HTML
# =========================
@app.route('/dashboard')
def dashboard():
    if 'email' not in session:
        return redirect('/login')

    try:
        stats = get_predictions_stats(session['email'])
        trees = max(1, stats['total'] // 5) if stats['total'] > 0 else 0
        
        # Get avatar from database
        avatar_data = None
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("SELECT avatar_data FROM users WHERE email=?", (session['email'],))
            row = c.fetchone()
            conn.close()
            if row and row[0]:
                avatar_data = row[0]
        except:
            pass

        # Render the dashboard template with all data
        return render_template(
            "dashboard.html",
            fullname=session.get('fullname', 'Coffee Farmer'),
            email=session.get('email'),
            phone=session.get('phone', ''),
            location=session.get('location', 'Uganda'),
            trees=trees,
            ripe=stats.get('ripe', 0),
            unripe=stats.get('unripe', 0),
            ripening=stats.get('ripening', 0),
            spoilt=stats.get('spoilt', 0),
            total=stats.get('total', 0),
            accuracy=stats.get('accuracy', 0),
            avatar_data=avatar_data,
            session=session
        )
    except Exception as e:
        print(f"Dashboard error: {e}")
        return render_template("dashboard.html", 
                             fullname=session.get("fullname", "Coffee Farmer"),
                             session=session)

# =========================
# API STATS
# =========================
@app.route('/api/dashboard_stats')
def dashboard_stats():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401

    try:
        stats = get_predictions_stats(session['email'])
        trees = max(1, stats['total'] // 5) if stats['total'] > 0 else 0

        return jsonify({
            "trees": trees,
            "ripe": stats['ripe'],
            "unripe": stats['unripe'],
            "ripening": stats['ripening'],
            "spoilt": stats.get('spoilt', 0),
            "total": stats['total'],
            "accuracy": stats['accuracy']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# COFFEE NEWS API
# =========================
@app.route('/api/coffee_news')
def coffee_news():
    """Get coffee news from Uganda with images and 5-minute caching."""
    try:
        news = fetch_coffee_news()
        for article in news:
            if not article.get('urlToImage'):
                article['urlToImage'] = get_fallback_image()
        # Return as list directly (not wrapped in articles) for frontend compatibility
        return jsonify(news)
    except Exception as e:
        print(f"Error fetching coffee news: {e}")
        mock_news = get_mock_news_with_images()
        return jsonify(mock_news)

# =========================
# PREDICT
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

    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        image = Image.open(filepath).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        result = CLASS_MAP.get(predicted.item(), "unripe")
        confidence_score = confidence.item()

        with open(filepath, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("""
            INSERT INTO predictions (email, filename, result, confidence, timestamp, image_data)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session['email'], filename, result, confidence_score, datetime.now().isoformat(), image_data))

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "result": result,
            "confidence": round(confidence_score * 100, 2),
            "filename": filename
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        # Return error instead of random result
        return jsonify({"success": False, "error": str(e)}), 500

# =========================
# PREDICT MULTIPLE
# =========================
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
        if file.filename == '':
            continue
            
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            image = Image.open(filepath).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            result = CLASS_MAP.get(predicted.item(), "unripe")
            confidence_score = confidence.item()

            with open(filepath, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("""
                INSERT INTO predictions (email, filename, result, confidence, timestamp, image_data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session['email'], filename, result, confidence_score, datetime.now().isoformat(), image_data))
            conn.commit()
            conn.close()

            results.append({
                "success": True,
                "result": result,
                "confidence": round(confidence_score * 100, 2),
                "filename": filename
            })

        except Exception as e:
            print(f"Prediction error for {filename}: {e}")
            # Return error instead of random result
            results.append({
                "success": False,
                "error": str(e),
                "filename": filename
            })
            rejected.append(filename)

    return jsonify({
        "results": results,
        "rejected": rejected,
        "rejected_count": len(rejected)
    })

# =========================
# GET HISTORY
# =========================
@app.route('/history')
def get_history():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("""
            SELECT id, filename, result, confidence, timestamp, image_data
            FROM predictions
            WHERE email=?
            ORDER BY id DESC
            LIMIT 50
        """, (session['email'],))
        rows = c.fetchall()
        conn.close()

        history = []
        for row in rows:
            history.append({
                "id": row[0],
                "filename": row[1],
                "result": row[2] or "Unknown",
                "confidence": round(row[3] * 100, 2) if row[3] else 0,
                "timestamp": row[4] or datetime.now().isoformat(),
                "image_data": row[5] if len(row) > 5 else None
            })

        return jsonify({"history": history})
    except Exception as e:
        print(f"History error: {e}")
        return jsonify({"history": []})

# =========================
# EXPORT DATA
# =========================
@app.route('/export_data')
def export_data():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("""
            SELECT filename, result, confidence, timestamp
            FROM predictions
            WHERE email=?
            ORDER BY id DESC
        """, (session['email'],))

        rows = c.fetchall()
        conn.close()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Filename', 'Result', 'Confidence (%)', 'Timestamp'])

        for row in rows:
            writer.writerow([
                row[0],
                row[1] or "Unknown",
                round(row[2] * 100, 2) if row[2] else 0,
                row[3] or datetime.now().isoformat()
            ])

        output.seek(0)

        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename=predictions_{datetime.now().strftime("%Y%m%d")}.csv'
        response.headers['Content-type'] = 'text/csv'

        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# GENERATE REPORT
# =========================
@app.route('/generate_report', methods=['POST'])
def generate_report():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401

    try:
        stats = get_predictions_stats(session['email'])

        report_data = {
            "generated": datetime.now().isoformat(),
            "total_predictions": stats['total'],
            "ripe": stats['ripe'],
            "ripening": stats['ripening'],
            "unripe": stats['unripe'],
            "spoilt": stats.get('spoilt', 0),
            "avg_confidence": stats['accuracy'],
        }

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO reports (email, report_name, report_data, created_at)
            VALUES (?, ?, ?, ?)
        """, (session['email'], f"Report_{datetime.now().strftime('%Y%m%d_%H%M')}", json.dumps(report_data), datetime.now().isoformat()))
        conn.commit()
        conn.close()

        return jsonify({"status": "success", "report": report_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# VIEW REPORTS
# =========================
@app.route('/reports')
def view_reports():
    if 'email' not in session:
        return redirect('/login')
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT id, report_name, created_at
            FROM reports
            WHERE email=?
            ORDER BY id DESC
        """, (session['email'],))
        reports = c.fetchall()
        conn.close()
        return render_template("reports_list.html", reports=reports, fullname=session.get("fullname"))
    except Exception as e:
        print(f"Error loading reports: {e}")
        return render_template("reports_list.html", reports=[], fullname=session.get("fullname"))

# =========================
# VIEW SINGLE REPORT
# =========================
@app.route('/report/<int:report_id>')
def view_report(report_id):
    if 'email' not in session:
        return redirect('/login')

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("""
            SELECT report_name, report_data, created_at
            FROM reports
            WHERE id=? AND email=?
        """, (report_id, session['email']))

        row = c.fetchone()
        conn.close()

        if not row:
            return "Report not found", 404

        report_data = json.loads(row[1])

        return render_template("report.html", report=report_data, report_name=row[0], fullname=session.get("fullname"))
    except Exception as e:
        return f"Error: {e}", 500

# =========================
# REPORTS DATA API
# =========================
@app.route('/reports_data')
def reports_data():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("""
            SELECT id, report_name, report_data, created_at
            FROM reports
            WHERE email=?
            ORDER BY id DESC
        """, (session['email'],))

        rows = c.fetchall()
        conn.close()

        reports = []
        for row in rows:
            data = json.loads(row[2])
            reports.append({
                "id": row[0],
                "name": row[1],
                "created_at": row[3],
                "ripe": data.get("ripe", 0),
                "ripening": data.get("ripening", 0),
                "unripe": data.get("unripe", 0),
                "spoilt": data.get("spoilt", 0),
                "total": data.get("total_predictions", 0)
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
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        c.execute("DELETE FROM predictions WHERE email=?", (session['email'],))
        c.execute("DELETE FROM reports WHERE email=?", (session['email'],))
        c.execute("DELETE FROM payments WHERE email=?", (session['email'],))
        c.execute("DELETE FROM settings WHERE email=?", (session['email'],))
        
        conn.commit()
        conn.close()

        return jsonify({"status": "cleared", "message": "All data deleted successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# MOBILE MONEY PAYMENT
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

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            INSERT INTO payments (email, fullname, phone, network, amount, status, transaction_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session['email'], fullname, phone, network, float(amount), "awaiting_transfer", transaction_ref, datetime.now().isoformat()))
        conn.commit()
        conn.close()

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

# =========================
# CONFIRM PAYMENT
# =========================
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
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("""
            SELECT fullname, phone, network, amount FROM payments
            WHERE email=? AND transaction_id=?
        """, (session['email'], reference))
        row = c.fetchone()

        if not row:
            conn.close()
            return jsonify({"error": "We couldn't find that payment reference"}), 404

        fullname, phone, network, amount = row

        c.execute("""
            UPDATE payments
            SET status=?, transaction_id=?
            WHERE email=? AND transaction_id=?
        """, ("pending_verification", momo_transaction_id, session['email'], reference))
        conn.commit()
        conn.close()

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

        return jsonify({
            "status": "pending_verification",
            "message": "Thanks! Your transaction ID has been submitted and is awaiting manual verification. You'll be notified once confirmed."
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# SEND HELP EMAIL
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
    
    return jsonify({
        "status": "sent", 
        "message": "Help request sent to loosendx@gmail.com!" if email_sent else "Help request logged. We'll get back to you soon."
    })

# =========================
# SAVE PROFILE - FIXED: Handles null avatar properly
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
    avatar_data = data.get("avatar_data", None)  # Can be null for deletion
    
    # Update session with text data only
    session['fullname'] = fullname
    session['phone'] = phone
    session['location'] = location
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # If avatar_data is None, set to NULL in database
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
        
        conn.commit()
        conn.close()
        
        return jsonify({"status": "success", "message": "Profile updated successfully!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# =========================
# SAVE SETTINGS
# =========================
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

# =========================
# GET USER SETTINGS
# =========================
@app.route('/api/settings')
def get_settings():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    
    settings = get_user_settings(session['email'])
    return jsonify(settings)

# =========================
# GET USER PROFILE
# =========================
@app.route('/api/profile')
def get_profile():
    if 'email' not in session:
        return jsonify({"error": "unauthorized"}), 401
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT fullname, email, phone, location, avatar_data FROM users WHERE email=?", (session['email'],))
        user = c.fetchone()
        conn.close()
        
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
# ESTIMATOR PRICES API
# =========================
@app.route('/api/estimator_prices')
def estimator_prices():
    """Return pricing data for the estimator with Fresh and Kiboko options."""
    return jsonify({
        "fresh": {
            "ripe_price": 5250,
            "ripening_price": 2500,
            "unripe_price": 0,
            "drying_ratio": 2.2,
            "ripe_per_kg": 500,
            "ripening_per_kg": 600,
            "unripe_per_kg": 1000,
            "note": "Wet mill gate prices - UCDA 2026"
        },
        "kiboko": {
            "ripe_price": 11550,
            "ripening_price": 5500,
            "unripe_price": 0,
            "drying_ratio": 2.2,
            "ripe_per_kg": 500,
            "ripening_per_kg": 600,
            "unripe_per_kg": 1000,
            "note": "Dried Kiboko prices - adjusted for drying loss (2.2:1 ratio)"
        },
        "source": "Uganda Coffee Development Authority (UCDA) - 2026 Indicative Farmgate Prices",
        "disclaimer": "Only ripe cherries have a sellable market value. Unripe cherries have no market."
    })

# =========================
# LOGOUT
# =========================
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("☕ CoffeeGuard Running...")
    print("📍 http://127.0.0.1:5000")
    print("📧 Help emails will be sent to loosendx@gmail.com")
    print("📱 Mobile Money payments (manual verification) go to 0759471328")
    print("🔄 CoffeeScope auto-refreshes every 5 minutes")
    print("📸 AI-powered coffee image validation enabled")
    print("💰 UCDA-verified estimator prices: Ripe=5,250 UGX/kg fresh")
    print("📊 Fresh/Kiboko toggle with 2.2:1 drying ratio")
    print("👤 Profile avatar persistence and delete functionality enabled")
    app.run(host='0.0.0.0', port=5000, debug=True)