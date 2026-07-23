import sqlite3

conn = sqlite3.connect("trafficDetector.db")

cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password TEXT
)
""")

conn.commit()
conn.close()

print("Users table created successfully")