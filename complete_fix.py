import sqlite3
import os
import shutil
from datetime import datetime

DB_PATH = "predictions.db"

def complete_fix():
    print("🔧 Starting complete database fix...")
    
    # Backup existing database
    if os.path.exists(DB_PATH):
        backup_path = DB_PATH + ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy(DB_PATH, backup_path)
        print(f"📦 Database backed up to: {backup_path}")
    
    # Delete old database to start fresh
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("🗑️ Old database deleted")
    
    # Create new database with correct schema
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create users table with ALL required columns
    c.execute("""
        CREATE TABLE users (
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
    print("✅ Users table created with all columns")
    
    # Create predictions table
    c.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            filename TEXT,
            result TEXT,
            confidence REAL,
            timestamp TEXT,
            image_data TEXT
        )
    """)
    print("✅ Predictions table created")
    
    # Create reports table
    c.execute("""
        CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            report_name TEXT,
            report_data TEXT,
            created_at TEXT
        )
    """)
    print("✅ Reports table created")
    
    # Create payments table
    c.execute("""
        CREATE TABLE payments (
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
    print("✅ Payments table created")
    
    # Create settings table
    c.execute("""
        CREATE TABLE settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            notification TEXT,
            default_view TEXT,
            language TEXT,
            theme TEXT,
            updated_at TEXT
        )
    """)
    print("✅ Settings table created")
    
    # Insert a test user for debugging
    c.execute("""
        INSERT INTO users (fullname, email, password, phone, location, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, ("Test User", "test@test.com", "password123", "0700000000", "Uganda", datetime.now().isoformat()))
    print("✅ Test user created (email: test@test.com, password: password123)")
    
    conn.commit()
    
    # Verify the table structure
    c.execute("PRAGMA table_info(users)")
    columns = c.fetchall()
    print("\n📋 Users table columns:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    conn.close()
    print("\n✅ Database created successfully!")
    print("📝 You can now register a new account or use test@test.com / password123")

if __name__ == "__main__":
    complete_fix()