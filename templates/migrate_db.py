import sqlite3
import os

DB_PATH = "predictions.db"

def migrate_database():
    """Add missing columns to the database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get existing columns
    c.execute("PRAGMA table_info(predictions)")
    columns = [col[1] for col in c.fetchall()]
    
    # Add missing columns
    if 'image_data' not in columns:
        print("📝 Adding image_data column...")
        c.execute("ALTER TABLE predictions ADD COLUMN image_data TEXT")
        conn.commit()
        print("✅ image_data column added!")
    else:
        print("✅ image_data column already exists.")
    
    # Check payments table
    c.execute("PRAGMA table_info(payments)")
    columns = [col[1] for col in c.fetchall()]
    
    if 'fullname' not in columns:
        print("📝 Adding fullname column to payments...")
        c.execute("ALTER TABLE payments ADD COLUMN fullname TEXT")
        conn.commit()
        print("✅ fullname column added!")
    
    if 'network' not in columns:
        print("📝 Adding network column to payments...")
        c.execute("ALTER TABLE payments ADD COLUMN network TEXT")
        conn.commit()
        print("✅ network column added!")
    
    conn.close()
    print("🎉 Database migration complete!")

if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        migrate_database()
    else:
        print(f"❌ Database file {DB_PATH} not found!")