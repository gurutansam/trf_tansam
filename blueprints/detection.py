import os
from flask import Blueprint, render_template, request, redirect, jsonify
from config import Config
from database import get_db
from services.auth_service import login_required
from services.detection_service import detection_manager

detection_bp = Blueprint("detection", __name__)

@detection_bp.route("/upload", methods=["GET"])
@login_required
def upload_page():
    live_payload = detection_manager.build_live_payload()
    return render_template("upload.html", live=live_payload)


@detection_bp.route("/upload", methods=["POST"])
@login_required
def upload_post():
    video_name = request.form.get("video_name", "").strip()
    camera = request.form.get("camera", "").strip()
    location = request.form.get("location", "").strip()
    phase = request.form.get("phase") or None
    start_dt = request.form.get("startDateTime", "").strip()

    if not all([video_name, camera, location, start_dt]):
        return "Missing required fields", 400

    video_path = os.path.join(Config.VIDEOS_DIR, video_name)
    if not os.path.exists(video_path):
        # Also check root if video was placed directly in root
        if os.path.exists(video_name):
            video_path = video_name
        else:
            return f"Video not found: {video_path}", 400

    # Remove any previous UNPROCESSED duplicate entry for same file+camera+location
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM files
        WHERE fileName=? AND camera=? AND location=? AND status='PROCESSING'
    """, (video_name, camera, location))
    cur.execute("""
        INSERT INTO files (fileName, camera, startDATE, status, phase, location)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (video_name, camera, start_dt, "PROCESSING", phase, location))
    new_file_id = cur.lastrowid
    conn.commit()
    conn.close()

    detection_manager.start_detection(
        video_path=video_path,
        file_id=new_file_id,
        video_name=video_name,
        camera=camera,
        location=location,
        start_dt=start_dt
    )

    return redirect("/progress")


@detection_bp.route("/progress")
@login_required
def progress_page():
    status = detection_manager.get_status()
    return render_template(
        "progress.html",
        camera=status.get("current_camera"),
        location=status.get("current_location"),
        video=status.get("current_video"),
        file_id=status.get("current_file_id"),
    )


@detection_bp.route("/progress/status")
@login_required
def progress_status():
    return jsonify(detection_manager.build_live_payload())
