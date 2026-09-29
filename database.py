import sqlite3
from config import Config

def get_db():
    """
    Returns a unified SQLite connection configured with:
    - 30s timeout
    - Row factory (supports row['col'], row[0], and tuple unpacking)
    - WAL journal mode for parallel reads/writes without table lock errors
    """
    conn = sqlite3.connect(Config.DB_PATH, timeout=Config.DB_TIMEOUT)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn

# Alias for backwards compatibility
db = get_db
