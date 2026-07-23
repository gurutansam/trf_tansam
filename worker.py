"""
worker.py — Parallel chunked detection worker.

Called from app.py when a video is uploaded.
Splits the video into 3 chunks, runs them in parallel.
Each chunk:
  - Writes detections to trafficClassification table (real camera/location)
  - Saves an annotated chunk video: output/1.mp4, output/2.mp4, output/3.mp4

No merging. 3 separate chunk videos are saved in output/.

When all chunks finish, marks the file status as COMPLETED in the DB.
"""
import subprocess
import sys
import os
import threading
import time
import sqlite3
import cv2

NUM_CHUNKS = 3
DB_PATH = "trafficDetector.db"
OUTPUT_DIR = "output"


def _get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def _wait_and_finalise(video_path, file_id, processes):
    """Wait for all chunk processes to finish, then mark COMPLETED in DB."""
    for i, p in enumerate(processes):
        p.wait()
        print(f"[worker] chunk {i} finished (code {p.returncode})", flush=True)

    # Mark COMPLETED
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("UPDATE files SET status='COMPLETED' WHERE id=?", (file_id,))
        conn.commit()
        conn.close()
        print(f"[worker] file {file_id} marked COMPLETED", flush=True)
    except Exception as e:
        print(f"[worker] DB update error: {e}", flush=True)


def run_detection(video_path, file_id=None):
    """
    Launch 3 parallel detection chunks.
    Called from app.py after inserting the file row.

    Args:
        video_path: path to video file (e.g. 'videos/111.mp4')
        file_id: the id of the row inserted in files table
                 (used to mark COMPLETED when done)
    """
    if not os.path.exists(video_path):
        print(f"[worker] ERROR: video not found: {video_path}")
        return

    script_path = os.path.join(os.path.dirname(__file__), "test4_chunk.py")
    if not os.path.exists(script_path):
        print(f"[worker] ERROR: test4_chunk.py not found at {script_path}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_frames = _get_total_frames(video_path)
    if total_frames <= 0:
        print(f"[worker] ERROR: cannot read frames from {video_path}")
        return

    print(f"[worker] {os.path.basename(video_path)}: {total_frames} frames, {NUM_CHUNKS} chunks")

    chunk_size = total_frames // NUM_CHUNKS
    ranges = []
    for i in range(NUM_CHUNKS):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < NUM_CHUNKS - 1 else total_frames
        ranges.append((start, end))
        print(f"[worker] chunk {i}: frames {start} -> {end}")

    python_exe = sys.executable
    processes = []

    for i, (start, end) in enumerate(ranges):
        cmd = [
            python_exe, script_path,
            video_path,          # arg 1: video path
            str(i),              # arg 2: chunk id (0, 1, 2)
            str(start),          # arg 3: start frame
            str(end)             # arg 4: end frame
        ]
        print(f"[worker] launching chunk {i}", flush=True)
        p = subprocess.Popen(cmd)
        processes.append(p)
        time.sleep(2)  # stagger launches to avoid GPU memory collision

    # Watch for completion in background thread (doesn't block Flask)
    threading.Thread(
        target=_wait_and_finalise,
        args=(video_path, file_id, processes),
        daemon=True
    ).start()