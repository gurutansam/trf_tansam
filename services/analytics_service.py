from datetime import datetime, timedelta
from database import get_db

def norm_dt(raw: str | None) -> str | None:
    """Normalize datetime format: '2026-01-01T21:00' -> '2026-01-01 21:00:00'"""
    if not raw:
        return None
    s = raw.replace("T", " ")
    if len(s) == 16:
        s += ":00"
    return s

def get_file_window(file_id: int):
    """
    Return (camera, location, start_dt, end_dt) for a single file.
    End date is either the start of the next video for the same camera/location
    or 24 hours later.
    """
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, camera, location, startDATE FROM files WHERE id=?", (file_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return None
    fid, camera, location, start_dt_raw = row
    start_dt = norm_dt(start_dt_raw)

    # Next file for the same camera+location (strictly after this one's startDATE)
    cur.execute("""
        SELECT MIN(startDATE) FROM files
        WHERE camera=? AND location=? AND startDATE > ? AND id != ?
    """, (camera, location, start_dt_raw, fid))
    next_raw = cur.fetchone()[0]
    conn.close()

    if next_raw:
        end_dt = norm_dt(next_raw)
    else:
        try:
            d = datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S")
            end_dt = (d + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            end_dt = "2999-12-31 23:59:59"

    return camera, location, start_dt, end_dt

def count_window(camera: str, location: str, sdt: str, edt: str) -> int:
    """Count vehicle detections within a time window for a camera and location."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM trafficClassification
        WHERE camera=? AND location=? AND time >= ? AND time < ?
    """, (camera, location, sdt, edt))
    n = cur.fetchone()[0]
    conn.close()
    return n

def summary_window(camera: str, location: str, sdt: str, edt: str):
    """Summarize vehicle class counts within a time window."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT vehicle, COUNT(*) FROM trafficClassification
        WHERE camera=? AND location=? AND time >= ? AND time < ?
        GROUP BY vehicle ORDER BY 2 DESC
    """, (camera, location, sdt, edt))
    rows = cur.fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]

def details_window(camera: str, location: str, sdt: str, edt: str):
    """Retrieve detailed timestamped records within a time window."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT time, camera, vehicle, location FROM trafficClassification
        WHERE camera=? AND location=? AND time >= ? AND time < ?
        ORDER BY time
    """, (camera, location, sdt, edt))
    rows = cur.fetchall()
    conn.close()
    return [(r[0], r[1], r[2], r[3]) for r in rows]

def get_results_data(start_dt: str, end_dt: str, camera: str, location: str):
    """Retrieve grouped vehicle counts for range query."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT vehicle, COUNT(*) FROM trafficClassification
        WHERE time BETWEEN ? AND ? AND camera=? AND location=?
        GROUP BY vehicle ORDER BY 2 DESC
    """, (start_dt, end_dt, camera, location))
    vehicle_counts = cur.fetchall()
    conn.close()
    
    vehicle_list = [(r[0], r[1]) for r in vehicle_counts]
    vehicle_dict = dict(vehicle_list)
    total_count = sum(vehicle_dict.values())
    return vehicle_list, vehicle_dict, total_count
