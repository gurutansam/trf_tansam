# Compatibility alias pointing to database.py
from database import get_db, db

__all__ = ["get_db", "db"]
