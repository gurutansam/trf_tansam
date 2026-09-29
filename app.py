import os
from flask import Flask
from config import Config
from blueprints.auth import auth_bp
from blueprints.dashboard import dashboard_bp
from blueprints.files import files_bp
from blueprints.detection import detection_bp
from blueprints.reports import reports_bp

def create_app(config_class=Config):
    """Application factory for TN Highways Traffic Detection System."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.secret_key = config_class.SECRET_KEY

    # Ensure required runtime storage directories exist
    os.makedirs(config_class.VIDEOS_DIR, exist_ok=True)
    os.makedirs(config_class.OUTPUT_DIR, exist_ok=True)

    # Register modular blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(files_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(reports_bp)

    return app

app = create_app()

if __name__ == "__main__":
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )