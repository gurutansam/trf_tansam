import os
from flask import Blueprint, render_template, redirect, send_from_directory, abort
from config import Config
from database import get_db
from services.auth_service import login_required
from services.analytics_service import get_file_window, count_window

files_bp = Blueprint("files", __name__)

@files_bp.route("/files")
@login_required
def files_page():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, fileName, camera, startDATE, status, phase, location
        FROM files ORDER BY id DESC
    """)
    files_raw = cur.fetchall()
    conn.close()

    enriched = []
    for f in files_raw:
        fid, fname, camera, start_dt, status, phase, location = f
        window = get_file_window(fid)
        count = count_window(*window[0:4]) if window else 0
        enriched.append({
            "id": fid,
            "fileName": fname,
            "camera": camera,
            "startDATE": start_dt,
            "status": status,
            "phase": phase,
            "location": location,
            "total_count": count,
        })
    return render_template("uploaded_files.html", files=enriched)


@files_bp.route("/delete_file/<int:file_id>", methods=["POST"])
@login_required
def delete_file(file_id):
    """Delete a file entry AND all its trafficClassification rows within its time window."""
    window = get_file_window(file_id)

    conn = get_db()
    cur = conn.cursor()

    if window:
        camera, location, sdt, edt = window
        cur.execute("""
            DELETE FROM trafficClassification
            WHERE camera=? AND location=? AND time >= ? AND time < ?
        """, (camera, location, sdt, edt))

    cur.execute("DELETE FROM files WHERE id=?", (file_id,))
    conn.commit()
    conn.close()
    return redirect("/files")


@files_bp.route("/video/<path:filename>")
@login_required
def stream_video(filename):
    """Serve processed or raw video file."""
    for folder in [Config.OUTPUT_DIR, Config.VIDEOS_DIR]:
        full_path = os.path.join(folder, filename)
        if os.path.exists(full_path):
            return send_from_directory(folder, filename)
    abort(404)
