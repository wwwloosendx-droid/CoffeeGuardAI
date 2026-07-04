import sqlite3
import os

DB_PATH = "predictions.db"

def fix_database():
    print("🔧 Fixing database...")
    
    # Backup existing database if it exists
    if os.path.exists(DB_PATH):
        import shutil
        backup_path = DB_PATH + ".backup"
        shutil.copy(DB_PATH, backup_path)
        print(f"📦 Database backed up to: {backup_path}")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # Check if users table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if c.fetchone():
            print("✅ Users table exists")
            
            # Get existing columns
            c.execute("PRAGMA table_info(users)")
            columns = [col[1] for col in c.fetchall()]
            print(f"📋 Current columns: {columns}")
            
            # Add missing columns
            missing = []
            if 'avatar_data' not in columns:
                missing.append('avatar_data')
            if 'phone' not in columns:
                missing.append('phone')
            if 'location' not in columns:
                missing.append('location')
                
            for col in missing:
                print(f"➕ Adding column: {col}")
                try:
                    c.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT")
                    print(f"✅ Added {col} successfully")
                except Exception as e:
                    print(f"⚠️ Could not add {col}: {e}")
            
            # Check if there are any users
            c.execute("SELECT COUNT(*) FROM users")
            count = c.fetchone()[0]
            print(f"👤 Users in database: {count}")
            
            if count == 0:
                print("📝 No users found. Please register first.")
        else:
            print("❌ Users table not found. Creating tables...")
            from app import init_db
            init_db()
            print("✅ Tables created successfully!")
            
        # Check predictions table
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        if c.fetchone():
            c.execute("PRAGMA table_info(predictions)")
            pred_cols = [col[1] for col in c.fetchall()]
            if 'image_data' not in pred_cols:
                print("➕ Adding image_data to predictions...")
                try:
                    c.execute("ALTER TABLE predictions ADD COLUMN image_data TEXT")
                    print("✅ Added image_data successfully")
                except Exception as e:
                    print(f"⚠️ Could not add image_data: {e}")
        
        conn.commit()
        print("✅ Database fix completed successfully!")
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    fix_database()
    
    # Test the connection
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT fullname, email FROM users")
        users = c.fetchall()
        print(f"\n👤 Users in database: {len(users)}")
        for user in users:
            print(f"  - {user[0]} ({user[1]})")
        conn.close()
        print("\n✅ Database is working correctly!")
    except Exception as e:
        print(f"\n❌ Database test failed: {e}")