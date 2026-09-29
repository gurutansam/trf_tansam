import sqlite3

conn = sqlite3.connect("trafficDetector.db")
cur = conn.cursor()

# 1. Users table
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password TEXT
)
""")

# 2. Files table
cur.execute("""
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fileName TEXT,
    camera TEXT,
    startDATE TEXT,
    status TEXT,
    phase TEXT,
    location TEXT
)
""")

# 3. Traffic classification table
cur.execute("""
CREATE TABLE IF NOT EXISTS trafficClassification (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT,
    camera TEXT,
    vehicle TEXT,
    phase TEXT,
    location TEXT
)
""")

# Default accounts
default_users = [
    ("Admin", "admin@example.com", "admin123"),
    ("Operator", "operator@example.com", "operator123"),
    ("Admin", "admin", "admin"),
    ("Operator", "operator", "operator"),
]

for name, email, password in default_users:
    cur.execute("""
        INSERT OR IGNORE INTO users (name, email, password)
        VALUES (?, ?, ?)
    """, (name, email, password))

conn.commit()
conn.close()

print("Database initialized successfully with default users and required tables.")