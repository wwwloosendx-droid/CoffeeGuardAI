import sqlite3

conn = sqlite3.connect("predictions.db")
c = conn.cursor()

# Add missing columns safely
try:
    c.execute("ALTER TABLE predictions ADD COLUMN confidence REAL")
except:
    pass

try:
    c.execute("ALTER TABLE predictions ADD COLUMN image_path TEXT")
except:
    pass

try:
    c.execute("ALTER TABLE predictions ADD COLUMN timestamp DATETIME DEFAULT CURRENT_TIMESTAMP")
except:
    pass

conn.commit()
conn.close()

print("Database updated successfully!")