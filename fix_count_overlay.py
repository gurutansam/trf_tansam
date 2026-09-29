"""
fix_count_overlay.py — Merge chunks + add cumulative count overlay.

Chunk videos have bounding boxes and ID tracking.
This script:
  1. Merges chunk videos using FFmpeg (or OpenCV fallback)
  2. Reads detections timeline from SQLite
  3. Renders a clean, high-contrast cumulative count card overlay
  4. Saves final video: output/<name>_FINAL.mp4
"""
import sys
import os
import cv2
import sqlite3
import subprocess
import time
from datetime import datetime
from collections import defaultdict

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(SCRIPT_DIR, "trafficDetector.db")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "output")
NUM_CHUNKS  = 3


def have_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


def merge_ffmpeg(base: str, chunk_files: list, merged_path: str):
    list_file = os.path.join(OUTPUT_DIR, f"_concat_list_{base}.txt")
    with open(list_file, "w") as f:
        for cf in chunk_files:
            abs_path = os.path.abspath(cf).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", list_file, "-c", "copy", merged_path]
    print("[merge] FFmpeg concat (fast)...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(list_file):
        os.remove(list_file)
    if result.returncode != 0:
        print(f"[merge] FFmpeg error: {result.stderr[-300:]}", flush=True)
        return False
    return True


def merge_opencv(chunk_files: list, merged_path: str):
    print("[merge] OpenCV concat fallback...", flush=True)
    cap = cv2.VideoCapture(chunk_files[0])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    writer = cv2.VideoWriter(merged_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        return False
    for i, cf in enumerate(chunk_files):
        cap = cv2.VideoCapture(cf)
        n = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            n += 1
        cap.release()
        print(f"[merge] chunk {i+1}: {n} frames", flush=True)
    writer.release()
    return True


def get_metadata(video_name: str):
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cur  = conn.cursor()
    cur.execute("""
        SELECT camera, location, startDATE FROM files
        WHERE fileName=? ORDER BY id DESC LIMIT 1
    """, (video_name,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise RuntimeError(f"No metadata found in DB for {video_name}")
    camera, location, sdt = row
    sdt = sdt.replace("T", " ")
    if len(sdt) == 16:
        sdt += ":00"
    return camera, location, datetime.strptime(sdt, "%Y-%m-%d %H:%M:%S")


def load_timeline(camera: str, location: str, video_start: datetime):
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cur  = conn.cursor()
    cur.execute("""
        SELECT time, vehicle FROM trafficClassification
        WHERE camera=? AND location=? ORDER BY time
    """, (camera, location))
    rows = cur.fetchall()
    conn.close()
    tl = []
    for ts, veh in rows:
        t   = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        sec = (t - video_start).total_seconds()
        if sec >= 0:
            tl.append((sec, veh))
    tl.sort(key=lambda x: x[0])
    return tl


def add_overlay(merged_path: str, final_path: str, timeline: list):
    cap          = cv2.VideoCapture(merged_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[overlay] Dimensions: {w}x{h} @ {fps:.1f}fps ({total_frames} frames)", flush=True)

    writer = cv2.VideoWriter(
        final_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        print(f"[overlay] ERROR: Cannot create {final_path}", flush=True)
        cap.release()
        return False

    cumulative = defaultdict(int)
    tl_idx     = 0
    frame_idx  = 0
    prog_step  = max(1, total_frames // 20)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        sec_now = frame_idx / fps

        while tl_idx < len(timeline) and timeline[tl_idx][0] <= sec_now:
            cumulative[timeline[tl_idx][1]] += 1
            tl_idx += 1

        active_classes = [(k, v) for k, v in sorted(cumulative.items(), key=lambda x: -x[1]) if v > 0]
        
        # Semi-transparent dark overlay card behind vehicle counts for readability
        card_w = 260
        card_h = 75 + len(active_classes) * 26
        overlay_mask = frame.copy()
        cv2.rectangle(overlay_mask, (12, 12), (12 + card_w, 12 + card_h), (18, 18, 18), -1)
        cv2.addWeighted(overlay_mask, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (12, 12), (12 + card_w, 12 + card_h), (0, 220, 255), 1)

        # Header
        y = 38
        cv2.putText(frame, "VEHICLE COUNTS", (22, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        
        # Class list
        y += 28
        for k, v in active_classes:
            cv2.putText(frame, f"{k}:", (22, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, str(v), (190, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2, cv2.LINE_AA)
            y += 26

        # Total line
        cv2.putText(frame, f"TOTAL: {sum(cumulative.values())}",
                    (22, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)

        if frame_idx % prog_step == 0:
            pct = (frame_idx / max(1, total_frames)) * 100
            print(f"[overlay] {pct:5.1f}%  total={sum(cumulative.values())}", flush=True)

    cap.release()
    writer.release()
    return True


def run_overlay(video_name: str):
    """Execute video merging and overlay rendering for given video name."""
    video_name = os.path.basename(video_name)
    base = os.path.splitext(video_name)[0]
    chunk_files = [os.path.join(OUTPUT_DIR, f"{base}_chunk{i+1}.mp4") for i in range(NUM_CHUNKS)]
    merged_path = os.path.join(OUTPUT_DIR, f"{base}_MERGED.mp4")
    final_path  = os.path.join(OUTPUT_DIR, f"{base}_FINAL.mp4")

    print("=" * 65, flush=True)
    print("MERGE + CUMULATIVE COUNT OVERLAY", flush=True)
    print("=" * 65, flush=True)

    # Check chunks exist
    for cf in chunk_files:
        if not os.path.exists(cf):
            print(f"[ERROR] Missing chunk: {cf}", flush=True)
            return False

    t0 = time.time()

    # Step 1: Merge
    ok = merge_ffmpeg(base, chunk_files, merged_path) if have_ffmpeg() else merge_opencv(chunk_files, merged_path)
    if not ok or not os.path.exists(merged_path):
        print("ERROR: Merge failed", flush=True)
        return False
    print(f"[merge] Done in {time.time()-t0:.1f}s", flush=True)

    # Step 2: Load DB
    camera, location, video_start = get_metadata(video_name)
    timeline = load_timeline(camera, location, video_start)
    print(f"[db] {len(timeline)} detections loaded for {camera} @ {location}", flush=True)

    # Step 3: Add overlay
    t1 = time.time()
    ok = add_overlay(merged_path, final_path, timeline)
    if not ok:
        return False
    print(f"[overlay] Done in {(time.time()-t1)/60:.1f} min", flush=True)

    # Cleanup temporary intermediates
    for cf in chunk_files:
        try: os.remove(cf)
        except: pass
    try: os.remove(merged_path)
    except: pass

    size_mb = os.path.getsize(final_path) / 1e6
    print("=" * 65, flush=True)
    print(f"DONE in {(time.time()-t0)/60:.1f} min | Output: {final_path} ({size_mb:.1f} MB)", flush=True)
    print("=" * 65, flush=True)
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_count_overlay.py <video_name>")
        sys.exit(1)
    run_overlay(sys.argv[1])