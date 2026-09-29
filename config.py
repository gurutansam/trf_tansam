import os

class Config:
    """Central configuration for TN Highways Traffic Detection System."""
    SECRET_KEY = os.environ.get("SECRET_KEY", "highway_secret_key")
    DB_PATH = os.environ.get("DB_PATH", "trafficDetector.db")
    VIDEOS_DIR = os.environ.get("VIDEOS_DIR", "videos")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
    
    # Server configuration
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 5000))
    DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "t")
    
    # Pagination & polling
    PER_PAGE = 20
    DB_TIMEOUT = 30.0
