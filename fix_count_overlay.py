"""
fix_count_overlay.py — Merge chunks + add cumulative count overlay.

Chunk videos have bounding boxes but NO count text.
This script:
  1. Merges chunk videos using FFmpeg (5-10 sec, no re-encoding)
  2. Reads merged video (has bounding boxes)
  3. Adds ONE cumulative count overlay from DB timestamps
  4. Saves final video, cleans up intermediates

Usage:
    python fix_count_overlay.py vandalur.mp4
    python fix_count_overlay.py forest_1.mp4

Output:
    output/<name>_FINAL.mp4  — bounding boxes + continuous cumulative count
"""
import sys
import os
import cv2
import sqlite3
import subprocess
import time
from datetime import datetime
from collections import defaultdict

if len(sys.argv) < 2:
    print("Usage: python fix_count_overlay.py <video_name>")
    sys.exit(1)

VIDEO_NAME  = sys.argv[1]
DB_PATH     = "trafficDetector.db"
OUTPUT_DIR  = "output"
NUM_CHUNKS  = 3

base         = os.path.splitext(VIDEO_NAME)[0]
chunk_files  = [os.path.join(OUTPUT_DIR, f"{base}_chunk{i+1}.mp4") for i in range(NUM_CHUNKS)]
merged_path  = os.path.join(OUTPUT_DIR, f"{base}_MERGED.mp4")
final_path   = os.path.join(OUTPUT_DIR, f"{base}_FINAL.mp4")


def have_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        return True
    except Exception:
        return False


def merge_ffmpeg():
    list_file = os.path.join(OUTPUT_DIR, "_concat_list.txt")
    with open(list_file, "w") as f:
        for cf in chunk_files:
            abs_path = os.path.abspath(cf).replace("\\", "/")
            f.write(f"file '{abs_path}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", list_file, "-c", "copy", merged_path]
    print("[merge] FFmpeg concat (fast)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(list_file)
    if result.returncode != 0:
        print(f"[merge] FFmpeg error: {result.stderr[-300:]}")
        return False
    return True


def merge_opencv():
    print("[merge] OpenCV (slower)...")
    cap = cv2.VideoCapture(chunk_files[0])
    fps = cap.get(cv2.CAP_PROP_FPS)
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
            if not ret: break
            writer.write(frame)
            n += 1
        cap.release()
        print(f"[merge] chunk {i+1}: {n} frames")
    writer.release()
    return True


def get_metadata():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        SELECT camera, location, startDATE FROM files
        WHERE fileName=? ORDER BY id DESC LIMIT 1
    """, (VIDEO_NAME,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise RuntimeError(f"No metadata for {VIDEO_NAME}")
    camera, location, sdt = row
    sdt = sdt.replace("T", " ")
    if len(sdt) == 16: sdt += ":00"
    return camera, location, datetime.strptime(sdt, "%Y-%m-%d %H:%M:%S")


def load_timeline(camera, location, video_start):
    conn = sqlite3.connect(DB_PATH)
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
    tl.sort()
    return tl


def add_overlay(timeline):
    cap          = cv2.VideoCapture(merged_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[overlay] {w}x{h} @ {fps:.1f}fps  {total_frames} frames")

    writer = cv2.VideoWriter(
        final_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        print(f"[overlay] ERROR: cannot create {final_path}")
        cap.release()
        return False

    cumulative = defaultdict(int)
    tl_idx     = 0
    frame_idx  = 0
    prog_step  = max(1, total_frames // 20)

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        sec_now = frame_idx / fps

        while tl_idx < len(timeline) and timeline[tl_idx][0] <= sec_now:
            cumulative[timeline[tl_idx][1]] += 1
            tl_idx += 1

        # Draw cumulative count overlay — top left
        y = 45
        cv2.putText(frame, "VEHICLE COUNT", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        y += 42
        for k, v in sorted(cumulative.items(), key=lambda x: -x[1]):
            if v > 0:
                cv2.putText(frame, f"{k}: {v}", (15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                y += 30
        y += 5
        cv2.putText(frame, f"TOTAL: {sum(cumulative.values())}",
                    (15, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        writer.write(frame)

        if frame_idx % prog_step == 0:
            pct = frame_idx / total_frames * 100
            print(f"[overlay] {pct:5.1f}%  total={sum(cumulative.values())}", flush=True)

    cap.release()
    writer.release()
    return True


def main():
    print("=" * 65)
    print("MERGE + CUMULATIVE COUNT OVERLAY")
    print("=" * 65)

    # Check chunks
    print("Chunk videos:")
    for cf in chunk_files:
        if os.path.exists(cf):
            print(f"  [FOUND]   {cf}  ({os.path.getsize(cf)/1e6:.0f} MB)")
        else:
            print(f"  [MISSING] {cf}")
            sys.exit(1)

    t0 = time.time()

    # Step 1: Merge
    print()
    ok = merge_ffmpeg() if have_ffmpeg() else merge_opencv()
    if not ok or not os.path.exists(merged_path):
        print("ERROR: merge failed"); sys.exit(1)
    print(f"[merge] Done in {time.time()-t0:.1f}s")

    # Step 2: Load DB
    print()
    camera, location, video_start = get_metadata()
    timeline = load_timeline(camera, location, video_start)
    print(f"[db] {len(timeline)} detections  camera={camera}  location={location}")

    # Step 3: Add overlay
    print()
    t1 = time.time()
    ok = add_overlay(timeline)
    if not ok: sys.exit(1)
    print(f"[overlay] Done in {(time.time()-t1)/60:.1f} min")

    # Cleanup
    print()
    for cf in chunk_files:
        try: os.remove(cf); print(f"[cleanup] deleted {cf}")
        except: pass
    try: os.remove(merged_path); print(f"[cleanup] deleted {merged_path}")
    except: pass

    size_mb = os.path.getsize(final_path) / 1e6
    print()
    print("=" * 65)
    print(f"DONE  total time: {(time.time()-t0)/60:.1f} min")
    print(f"FINAL VIDEO: {final_path}  ({size_mb:.0f} MB)")
    print("Bounding boxes + continuous cumulative count 0 to total.")
    print("=" * 65)


if __name__ == "__main__":
    main()