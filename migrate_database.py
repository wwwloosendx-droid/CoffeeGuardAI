# migrate_database.py
import sqlite3
import json
import os

DB_PATH = "predictions.db"

def migrate_database():
    """Migrate database with safe column additions"""
    print("🔄 Migrating Database...")
    print("=" * 60)
    
    if not os.path.exists(DB_PATH):
        print("❌ Database not found.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get existing columns
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [col[1] for col in cursor.fetchall()]
    print(f"📋 Existing columns: {columns}")
    
    # Add columns if they don't exist
    changes_made = False
    
    if 'cherry_count' not in columns:
        print("➕ Adding cherry_count column...")
        cursor.execute("ALTER TABLE predictions ADD COLUMN cherry_count INTEGER DEFAULT 0")
        changes_made = True
    
    if 'class_counts' not in columns:
        print("➕ Adding class_counts column...")
        cursor.execute("ALTER TABLE predictions ADD COLUMN class_counts TEXT")
        changes_made = True
    
    if 'total_cherries' not in columns:
        print("➕ Adding total_cherries column...")
        cursor.execute("ALTER TABLE predictions ADD COLUMN total_cherries INTEGER DEFAULT 0")
        changes_made = True
    
    if 'detection_type' not in columns:
        print("➕ Adding detection_type column...")
        cursor.execute("ALTER TABLE predictions ADD COLUMN detection_type TEXT DEFAULT 'bean'")
        changes_made = True
    
    if changes_made:
        conn.commit()
        print("✅ Database migration complete!")
    else:
        print("✅ Database is already up to date!")
    
    conn.close()
    print("=" * 60)

if __name__ == "__main__":
    migrate_database()