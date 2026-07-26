# fix_database.py
import sqlite3
import os

DB_PATH = "predictions.db"

def fix_database():
    """Add missing columns to the predictions table"""
    print("🔧 Fixing Database Schema...")
    print("=" * 60)
    
    # Check if database exists
    if not os.path.exists(DB_PATH):
        print("❌ Database not found. Run the app first to create it.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check current columns
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [col[1] for col in cursor.fetchall()]
    print(f"📋 Current columns: {columns}")
    
    # Add missing columns
    columns_to_add = [
        ("cherry_count", "INTEGER DEFAULT 0"),
        ("class_counts", "TEXT"),
        ("total_cherries", "INTEGER DEFAULT 0"),
        ("detection_type", "TEXT DEFAULT 'bean'")
    ]
    
    for col_name, col_type in columns_to_add:
        if col_name not in columns:
            print(f"📝 Adding column: {col_name}")
            cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
    
    conn.commit()
    conn.close()
    
    print("✅ Database fixed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    fix_database()