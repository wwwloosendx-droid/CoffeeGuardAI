# reset_database.py
import sqlite3
import os

DB_PATH = "predictions.db"

def reset_database():
    """Complete database reset with correct schema"""
    print("🔄 Resetting Database...")
    print("=" * 60)
    
    # Delete old database
    if os.path.exists(DB_PATH):
        print("🗑️ Deleting old database...")
        os.remove(DB_PATH)
    
    # Create new database with correct schema
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
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
    
    # Create predictions table with all columns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            filename TEXT,
            result TEXT,
            confidence REAL,
            timestamp TEXT,
            image_data TEXT,
            cherry_count INTEGER DEFAULT 0,
            class_counts TEXT,
            total_cherries INTEGER DEFAULT 0,
            detection_type TEXT DEFAULT 'bean'
        )
    """)
    
    # Create aerial_detections table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS aerial_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            filename TEXT,
            total_cherries INTEGER,
            clusters_count INTEGER,
            confidence REAL,
            timestamp TEXT,
            image_data TEXT,
            detection_details TEXT
        )
    """)
    
    # Create reports table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            report_name TEXT,
            report_data TEXT,
            created_at TEXT
        )
    """)
    
    # Create payments table
    cursor.execute("""
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
    
    # Create settings table
    cursor.execute("""
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
    
    # Create coffee_news_cache table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS coffee_news_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_data TEXT,
            fetched_at TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    
    print("✅ Database reset complete with correct schema!")
    print("=" * 60)

if __name__ == "__main__":
    reset_database()