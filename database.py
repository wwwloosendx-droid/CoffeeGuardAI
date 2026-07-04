import sqlite3

conn = sqlite3.connect("predictions.db")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    result TEXT
)
""")

conn.commit()
conn.close()

print("Database created successfully!")