from flask import Flask, render_template, request, redirect, jsonify, Response
import sqlite3
import os
import threading
import io
from datetime import datetime, timedelta
import sqlite3
from flask import session

from worker import run_detection
from reportlab.platypus import TableStyle
from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from openpyxl import Workbook

app = Flask(__name__)
app.secret_key = "highway_secret_key"
# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        conn = get_db()
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        )

        user = cur.fetchone()

        conn.close()

        # EMAIL EXISTS
        if user:

            # PASSWORD MATCH
            if user["password"] == password:

                session["user"] = user["email"]
                session["name"] = user["name"]

                return redirect("/site")

            else:

                return render_template(
                    "Login.html",
                    error="Wrong password"
                )

        else:

            return render_template(
                "Login.html",
                error="Account does not exist"
            )

    return render_template("Login.html")


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")


# ============================================================
# REGISTER
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")

        conn = get_db()
        cur = conn.cursor()

        # CHECK EMAIL EXISTS
        cur.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        )

        existing_user = cur.fetchone()

        if existing_user:

            conn.close()

            return render_template(
                "register.html",
                error="Email already exists"
            )

        # INSERT NEW USER
        cur.execute(
            "INSERT INTO users(name, email, password) VALUES(?, ?, ?)",
            (name, email, password)
        )

        conn.commit()
        conn.close()

        return redirect("/login")

    return render_template("register.html")


# ============================================================
# FORGOT PASSWORD
# ============================================================

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():

    if request.method == "POST":

        email = request.form.get("email")
        new_password = request.form.get("password")

        conn = get_db()
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM users WHERE email=?",
            (email,)
        )

        user = cur.fetchone()

        if user:

            cur.execute(
                "UPDATE users SET password=? WHERE email=?",
                (new_password, email)
            )

            conn.commit()
            conn.close()

            return redirect("/login")

        else:

            conn.close()

            return render_template(
                "forgot_password.html",
                error="Email not found"
            )

    return render_template("forgot_password.html")


DB_PATH = "trafficDetector.db"
PER_PAGE = 20
def get_db():

    conn = sqlite3.connect("trafficDetector.db")
    conn.row_factory = sqlite3.Row

    return conn
PROCESS_STATUS = {
    "running": False,
    "message": "Idle",
    "current_file_id": None,
    "current_video": None,
    "current_camera": None,
    "current_location": None,
    "current_start_dt": None,
}


def db():
    return sqlite3.connect(DB_PATH)


# ============================================================
# HELPERS
# ============================================================
def _norm_dt(raw):
    """'2026-01-01T21:00' → '2026-01-01 21:00:00'"""
    if not raw:
        return None
    s = raw.replace("T", " ")
    if len(s) == 16:
        s += ":00"
    return s


def _get_file_window(file_id):
    """Return (camera, location, start_dt, end_dt) for a single file."""
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT id, camera, location, startDATE FROM files WHERE id=?", (file_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return None
    fid, camera, location, start_dt_raw = row
    start_dt = _norm_dt(start_dt_raw)

    # Next file for the same camera+location (strictly after this one's startDATE)
    cur.execute("""
        SELECT MIN(startDATE) FROM files
        WHERE camera=? AND location=? AND startDATE > ? AND id != ?
    """, (camera, location, start_dt_raw, fid))
    next_raw = cur.fetchone()[0]
    conn.close()

    if next_raw:
        end_dt = _norm_dt(next_raw)
    else:
        try:
            d = datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S")
            end_dt = (d + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            end_dt = "2999-12-31 23:59:59"

    return camera, location, start_dt, end_dt


def _count_window(camera, location, sdt, edt):
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM trafficClassification
        WHERE camera=? AND location=? AND time >= ? AND time < ?
    """, (camera, location, sdt, edt))
    n = cur.fetchone()[0]
    conn.close()
    return n


def _summary_window(camera, location, sdt, edt):
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT vehicle, COUNT(*) FROM trafficClassification
        WHERE camera=? AND location=? AND time >= ? AND time < ?
        GROUP BY vehicle ORDER BY 2 DESC
    """, (camera, location, sdt, edt))
    rows = cur.fetchall()
    conn.close()
    return rows


def _details_window(camera, location, sdt, edt):
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT time, camera, vehicle, location FROM trafficClassification
        WHERE camera=? AND location=? AND time >= ? AND time < ?
        ORDER BY time
    """, (camera, location, sdt, edt))
    rows = cur.fetchall()
    conn.close()
    return rows


# ============================================================
# HOME
# ============================================================
@app.route("/")
def home():
    return redirect("/login")


# ============================================================
# DASHBOARD
# ============================================================
@app.route("/site")
def site():
    total_files = 0
    try:
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM files")
        total_files = cur.fetchone()[0]
        conn.close()
    except Exception as e:
        print("[/site] error:", e)

    is_running = PROCESS_STATUS.get("running", False)
    return render_template(
        "site.html",
        total_files=total_files,
        is_running=is_running,
        current_video=PROCESS_STATUS.get("current_video"),
        current_camera=PROCESS_STATUS.get("current_camera"),
        current_location=PROCESS_STATUS.get("current_location"),
    )
@app.route("/lookup")
def lookup():
    return render_template("traffic_detection.html")


# ============================================================
# FILES — per-file totals, delete
# ============================================================
@app.route("/files")
def files_page():
    """Real schema: id | fileName | camera | startDATE | status | phase | location"""
    conn = db()
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
        window = _get_file_window(fid)
        count = _count_window(*window[0:4]) if window else 0
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


@app.route("/delete_file/<int:file_id>", methods=["POST"])
def delete_file(file_id):
    """Delete a file entry AND all its trafficClassification rows."""
    window = _get_file_window(file_id)

    conn = db()
    cur = conn.cursor()

    # Delete detections for this file's time window if we can determine it
    if window:
        camera, location, sdt, edt = window
        cur.execute("""
            DELETE FROM trafficClassification
            WHERE camera=? AND location=? AND time >= ? AND time < ?
        """, (camera, location, sdt, edt))

    # Delete the file row itself
    cur.execute("DELETE FROM files WHERE id=?", (file_id,))
    conn.commit()
    conn.close()
    return redirect("/files")


# ============================================================
# UPLOAD
# ============================================================
@app.route("/upload", methods=["GET"])
def upload_page():
    live_payload = _build_live_payload()
    return render_template("upload.html", live=live_payload)


@app.route("/upload", methods=["POST"])
def upload_post():
    global PROCESS_STATUS

    video_name = request.form.get("video_name", "").strip()
    camera = request.form.get("camera", "").strip()
    location = request.form.get("location", "").strip()
    phase = request.form.get("phase") or None
    start_dt = request.form.get("startDateTime", "").strip()

    if not all([video_name, camera, location, start_dt]):
        return "Missing required fields", 400

    video_path = os.path.join("videos", video_name)
    if not os.path.exists(video_path):
        return f"Video not found: {video_path}", 400

    # Remove any previous UNPROCESSED duplicate entry for same file+camera+location
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM files
        WHERE fileName=? AND camera=? AND location=? AND status='PROCESSING'
    """, (video_name, camera, location))
    cur.execute("""
        INSERT INTO files (fileName, camera, startDATE, status, phase, location)
        VALUES (?,?,?,?,?,?)
    """, (video_name, camera, start_dt, "PROCESSING", phase, location))
    new_file_id = cur.lastrowid
    conn.commit()
    conn.close()

    PROCESS_STATUS.update({
        "running": True,
        "message": "Detection started",
        "current_file_id": new_file_id,
        "current_video": video_name,
        "current_camera": camera,
        "current_location": location,
        "current_start_dt": start_dt,
    })

    threading.Thread(
        target=_run_and_finalise,
        args=(video_path, new_file_id),
        daemon=True
    ).start()

    return redirect("/progress")


def _run_and_finalise(video_path, file_id):
    """
    Launch parallel chunked detection.
    worker.py marks COMPLETED when all chunks finish.
    This thread polls the DB until COMPLETED then updates PROCESS_STATUS.
    """
    global PROCESS_STATUS
    import time as _time
    try:
        run_detection(video_path, file_id)
    except Exception as e:
        print("[detection] error:", e)

    # Poll DB until worker marks COMPLETED
    print("[finalise] waiting for all chunks to complete...")
    for _ in range(7200):   # max 2 hours
        try:
            conn = db()
            cur = conn.cursor()
            cur.execute("SELECT status FROM files WHERE id=?", (file_id,))
            row = cur.fetchone()
            conn.close()
            if row and row[0] == "COMPLETED":
                print(f"[finalise] file {file_id} COMPLETED")
                break
        except Exception:
            pass
        _time.sleep(5)

    PROCESS_STATUS.update({"running": False, "message": "Detection completed"})


# ============================================================
# PROGRESS
# ============================================================
@app.route("/progress")
def progress_page():
    return render_template(
        "progress.html",
        camera=PROCESS_STATUS.get("current_camera"),
        location=PROCESS_STATUS.get("current_location"),
        video=PROCESS_STATUS.get("current_video"),
        file_id=PROCESS_STATUS.get("current_file_id"),
    )


def _build_live_payload():
    # If PROCESS_STATUS lost state (Flask restart), recover from DB
    is_running = PROCESS_STATUS.get("running", False)
    cam        = PROCESS_STATUS.get("current_camera")
    loc        = PROCESS_STATUS.get("current_location")
    start_raw  = PROCESS_STATUS.get("current_start_dt")
    video      = PROCESS_STATUS.get("current_video")
    file_id    = PROCESS_STATUS.get("current_file_id")

    # If not running in memory, check DB for any PROCESSING file
    if not is_running:
        try:
            conn = db()
            cur  = conn.cursor()
            cur.execute("""
                SELECT id, fileName, camera, location, startDATE
                FROM files WHERE status='PROCESSING'
                ORDER BY id DESC LIMIT 1
            """)
            row = cur.fetchone()
            conn.close()
            if row:
                file_id, video, cam, loc, start_raw = row
                is_running = True
                # Restore PROCESS_STATUS
                PROCESS_STATUS.update({
                    "running": True,
                    "message": "Detection running",
                    "current_file_id": file_id,
                    "current_video": video,
                    "current_camera": cam,
                    "current_location": loc,
                    "current_start_dt": start_raw,
                })
        except Exception as e:
            print("[live] DB recovery error:", e)

    payload = {
        "running": is_running,
        "message": PROCESS_STATUS.get("message", "Idle"),
        "video": video,
        "camera": cam,
        "location": loc,
        "file_id": file_id,
        "total_count": 0,
        "vehicle_counts": {},
    }

    if cam and loc and start_raw:
        try:
            sdt  = _norm_dt(start_raw)
            rows = _summary_window(cam, loc, sdt, "2999-12-31 23:59:59")
            payload["vehicle_counts"] = {v: c for v, c in rows}
            payload["total_count"]    = sum(payload["vehicle_counts"].values())
        except Exception as e:
            print("[live] error:", e)
    return payload


@app.route("/progress/status")
def progress_status():
    return jsonify(_build_live_payload())


# ============================================================
# RESULTS
# ============================================================
@app.route("/results", methods=["POST"])
def results():
    start_date = request.form.get("start_date")
    end_date = request.form.get("end_date")
    from_time = request.form.get("from_time")
    to_time = request.form.get("to_time")
    camera = request.form.get("camera")
    location = request.form.get("location")

    if not all([start_date, end_date, from_time, to_time, camera, location]):
        return "Missing required filters", 400

    start_dt = f"{start_date} {from_time}"
    end_dt = f"{end_date} {to_time}"

    conn = db()
    cur = conn.cursor()
    cur.execute("""
        SELECT vehicle, COUNT(*) FROM trafficClassification
        WHERE time BETWEEN ? AND ? AND camera=? AND location=?
        GROUP BY vehicle ORDER BY 2 DESC
    """, (start_dt, end_dt, camera, location))
    vehicle_counts = cur.fetchall()   # keep as ordered list for chart
    vehicle_dict = dict(vehicle_counts)

    conn.close()

    return render_template(
        "results.html",
        vehicle_counts=vehicle_dict,
        vehicle_list=vehicle_counts,  # list for chart order
        total_count=sum(vehicle_dict.values()),
        duration=f"{start_dt} — {end_dt}",
        camera=camera, location=location,
        start_date=start_date, end_date=end_date,
        from_time=from_time, to_time=to_time
    )


# ============================================================
# DOWNLOADS — range-based
# ============================================================
@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    sf = request.form.get("start_date") + " " + request.form.get("from_time")
    ef = request.form.get("end_date") + " " + request.form.get("to_time")
    cam = request.form.get("camera")
    loc = request.form.get("location")
    if not all([sf, ef, cam, loc]):
        return "Missing parameters", 400
    summary = _summary_window(cam, loc, sf, ef)
    details = _details_window(cam, loc, sf, ef)
    total = sum(c for _, c in summary)
    buf = _build_pdf(cam, loc, sf, ef, summary, details, total)
    return Response(buf, mimetype="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=traffic_report.pdf"})


@app.route("/download_excel", methods=["POST"])
def download_excel():
    sf = request.form.get("start_date") + " " + request.form.get("from_time")
    ef = request.form.get("end_date") + " " + request.form.get("to_time")
    cam = request.form.get("camera")
    loc = request.form.get("location")
    if not all([sf, ef, cam, loc]):
        return "Missing parameters", 400
    summary = _summary_window(cam, loc, sf, ef)
    details = _details_window(cam, loc, sf, ef)
    total = sum(c for _, c in summary)
    out = _build_excel(cam, loc, sf, ef, summary, details, total)
    safe = loc.replace(" ", "_").replace("/", "-")
    return Response(out,
                    mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f"attachment; filename=report_{safe}.xlsx"})


# ============================================================
# DOWNLOADS — per-file
# ============================================================
@app.route("/file_pdf/<int:file_id>")
def file_pdf(file_id):
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT fileName, status FROM files WHERE id=?", (file_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return "File not found", 404
    fname, status = row
    if status == "PROCESSING":
        return "Still processing — download available after completion.", 409

    window = _get_file_window(file_id)
    if not window:
        return "No data", 404
    cam, loc, sdt, edt = window
    summary = _summary_window(cam, loc, sdt, edt)
    details = _details_window(cam, loc, sdt, edt)
    total = sum(c for _, c in summary)
    buf = _build_pdf(cam, loc, sdt, edt, summary, details, total, fname)
    safe = fname.replace(".mp4", "").replace(" ", "_").replace("/", "-")
    return Response(buf, mimetype="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename={safe}.pdf"})


@app.route("/file_excel/<int:file_id>")
def file_excel(file_id):
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT fileName, status FROM files WHERE id=?", (file_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return "File not found", 404
    fname, status = row
    if status == "PROCESSING":
        return "Still processing — download available after completion.", 409

    window = _get_file_window(file_id)
    if not window:
        return "No data", 404
    cam, loc, sdt, edt = window
    summary = _summary_window(cam, loc, sdt, edt)
    details = _details_window(cam, loc, sdt, edt)
    total = sum(c for _, c in summary)
    out = _build_excel(cam, loc, sdt, edt, summary, details, total, fname)
    safe = fname.replace(".mp4", "").replace(" ", "_").replace("/", "-")
    return Response(out,
                    mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f"attachment; filename={safe}.xlsx"})


# ============================================================
# PDF / EXCEL builders
# ============================================================
def _build_pdf(camera, location, sf, ef, summary, details, total, video_name=None):
    buffer = io.BytesIO()
    PW, PH = landscape(A4)
    doc = SimpleDocTemplate(buffer, pagesize=(PW, PH),
                            leftMargin=30, rightMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    el = [Paragraph("<b>Traffic Detection Report</b>", styles["Title"])]
    if video_name:
        el.append(Paragraph(f"<b>Video:</b> {video_name}", styles["Normal"]))
    el += [
        Paragraph(f"<b>Camera:</b> {camera}", styles["Normal"]),
        Paragraph(f"<b>Location:</b> {location}", styles["Normal"]),
        Paragraph(f"<b>Duration:</b> {sf} → {ef}", styles["Normal"]),
        Paragraph(f"<b>Total Vehicle Count:</b> {total}", styles["Normal"]),
        Paragraph("<br/>", styles["Normal"]),
    ]
    st = Table([["Vehicle", "Count"]] + list(summary),
               colWidths=[PW * 0.6, PW * 0.2], hAlign="LEFT")
    st.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
        ("FONT", (0,0), (-1,0), "Helvetica-Bold")
    ]))
    el.append(st)
    el.append(PageBreak())
    dt = Table([["Time", "Camera", "Vehicle", "Location"]] + list(details),
               colWidths=[PW*0.25, PW*0.15, PW*0.30, PW*0.20], repeatRows=1, hAlign="LEFT")
    dt.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONT", (0,0), (-1,0), "Helvetica-Bold"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE")
    ]))
    el.append(dt)
    doc.build(el)
    buffer.seek(0)
    return buffer


def _build_excel(camera, location, sf, ef, summary, details, total, video_name=None):
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Summary"
    if video_name:
        ws1.append(["Video", video_name])
    ws1.append(["Camera", camera])
    ws1.append(["Location", location])
    ws1.append(["Duration", f"{sf} → {ef}"])
    ws1.append(["Total Vehicle Count", total])
    ws1.append([])
    ws1.append(["Vehicle", "Count"])
    for row in summary:
        ws1.append(row)

    ws2 = wb.create_sheet(title="Details")
    if video_name:
        ws2.append(["Video", video_name])
    ws2.append(["Camera", camera])
    ws2.append(["Location", location])
    ws2.append(["Duration", f"{sf} → {ef}"])
    ws2.append(["Total Vehicle Count", total])
    ws2.append([])
    ws2.append(["Time", "Camera", "Vehicle", "Location"])
    for row in details:
        ws2.append(row)

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output


if __name__ == "__main__":
    app.run(host="192.168.1.117", port=5000, debug=True)