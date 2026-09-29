from flask import Blueprint, render_template, redirect, session
from database import get_db
from services.auth_service import login_required
from services.detection_service import detection_manager

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/")
def home():
    if session.get("user"):
        return redirect("/site")
    return redirect("/login")

@dashboard_bp.route("/site")
@login_required
def site():
    total_files = 0
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM files")
        row = cur.fetchone()
        total_files = row[0] if row else 0
        conn.close()
    except Exception as e:
        print("[dashboard_bp /site] error:", e)

    status = detection_manager.get_status()
    return render_template(
        "site.html",
        total_files=total_files,
        is_running=status.get("running", False),
        current_video=status.get("current_video"),
        current_camera=status.get("current_camera"),
        current_location=status.get("current_location"),
    )

@dashboard_bp.route("/lookup")
@login_required
def lookup():
    return render_template("traffic_detection.html")
