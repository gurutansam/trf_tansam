from flask import Blueprint, render_template, request, Response
from database import get_db
from services.auth_service import login_required
from services.analytics_service import (
    get_file_window,
    summary_window,
    details_window,
    get_results_data
)
from services.report_service import build_pdf, build_excel

reports_bp = Blueprint("reports", __name__)

@reports_bp.route("/results", methods=["POST"])
@login_required
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

    vehicle_list, vehicle_dict, total_count = get_results_data(
        start_dt, end_dt, camera, location
    )

    return render_template(
        "results.html",
        vehicle_counts=vehicle_dict,
        vehicle_list=vehicle_list,
        total_count=total_count,
        duration=f"{start_dt} — {end_dt}",
        camera=camera,
        location=location,
        start_date=start_date,
        end_date=end_date,
        from_time=from_time,
        to_time=to_time
    )


@reports_bp.route("/download_pdf", methods=["POST"])
@login_required
def download_pdf():
    start_date = request.form.get("start_date", "")
    from_time = request.form.get("from_time", "")
    end_date = request.form.get("end_date", "")
    to_time = request.form.get("to_time", "")
    cam = request.form.get("camera")
    loc = request.form.get("location")

    if not all([start_date, from_time, end_date, to_time, cam, loc]):
        return "Missing parameters", 400

    sf = f"{start_date} {from_time}"
    ef = f"{end_date} {to_time}"

    summary = summary_window(cam, loc, sf, ef)
    details = details_window(cam, loc, sf, ef)
    total = sum(c for _, c in summary)
    buf = build_pdf(cam, loc, sf, ef, summary, details, total)

    return Response(
        buf,
        mimetype="application/pdf",
        headers={"Content-Disposition": "attachment; filename=traffic_report.pdf"}
    )


@reports_bp.route("/download_excel", methods=["POST"])
@login_required
def download_excel():
    start_date = request.form.get("start_date", "")
    from_time = request.form.get("from_time", "")
    end_date = request.form.get("end_date", "")
    to_time = request.form.get("to_time", "")
    cam = request.form.get("camera")
    loc = request.form.get("location")

    if not all([start_date, from_time, end_date, to_time, cam, loc]):
        return "Missing parameters", 400

    sf = f"{start_date} {from_time}"
    ef = f"{end_date} {to_time}"

    summary = summary_window(cam, loc, sf, ef)
    details = details_window(cam, loc, sf, ef)
    total = sum(c for _, c in summary)
    out = build_excel(cam, loc, sf, ef, summary, details, total)
    safe = loc.replace(" ", "_").replace("/", "-")

    return Response(
        out,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=report_{safe}.xlsx"}
    )


@reports_bp.route("/file_pdf/<int:file_id>")
@login_required
def file_pdf(file_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT fileName, status FROM files WHERE id=?", (file_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return "File not found", 404

    fname, status = row
    if status == "PROCESSING":
        return "Still processing — download available after completion.", 409

    window = get_file_window(file_id)
    if not window:
        return "No data", 404

    cam, loc, sdt, edt = window
    summary = summary_window(cam, loc, sdt, edt)
    details = details_window(cam, loc, sdt, edt)
    total = sum(c for _, c in summary)
    buf = build_pdf(cam, loc, sdt, edt, summary, details, total, fname)
    safe = fname.replace(".mp4", "").replace(" ", "_").replace("/", "-")

    return Response(
        buf,
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={safe}.pdf"}
    )


@reports_bp.route("/file_excel/<int:file_id>")
@login_required
def file_excel(file_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT fileName, status FROM files WHERE id=?", (file_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return "File not found", 404

    fname, status = row
    if status == "PROCESSING":
        return "Still processing — download available after completion.", 409

    window = get_file_window(file_id)
    if not window:
        return "No data", 404

    cam, loc, sdt, edt = window
    summary = summary_window(cam, loc, sdt, edt)
    details = details_window(cam, loc, sdt, edt)
    total = sum(c for _, c in summary)
    out = build_excel(cam, loc, sdt, edt, summary, details, total, fname)
    safe = fname.replace(".mp4", "").replace(" ", "_").replace("/", "-")

    return Response(
        out,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={safe}.xlsx"}
    )
