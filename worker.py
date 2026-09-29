"""
worker.py — Parallel chunked detection worker.

Called from detection_service / app.py when a video is registered.
Splits the video into 3 chunks, runs them in parallel.
Each chunk:
  - Writes detections to trafficClassification table (real camera/location)
  - Saves an annotated chunk video in output/

When all chunks finish:
  - Merges chunks and generates cumulative count overlay via fix_count_overlay.py
  - Marks file status as COMPLETED in DB
"""
import subprocess
import sys
import os
import threading
import time
import sqlite3
import cv2

NUM_CHUNKS = 3
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "trafficDetector.db")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def _get_total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def _wait_and_finalise(video_path: str, file_id: int, processes: list):
    """Wait for all chunk processes to finish, run final overlay merge, and update DB status."""
    failed_chunks = []
    for i, p in enumerate(processes):
        p.wait()
        print(f"[worker] Chunk {i} finished (code {p.returncode})", flush=True)
        if p.returncode != 0:
            failed_chunks.append(i)

    if not failed_chunks:
        # Automatically generate merged final video with cumulative overlay
        try:
            fix_script = os.path.join(SCRIPT_DIR, "fix_count_overlay.py")
            if os.path.exists(fix_script):
                video_name = os.path.basename(video_path)
                print(f"[worker] Launching automatic merge and cumulative overlay for {video_name}...", flush=True)
                res = subprocess.run([sys.executable, fix_script, video_name], capture_output=True, text=True)
                if res.returncode == 0:
                    print(f"[worker] Final video generated successfully for {video_name}.", flush=True)
                else:
                    print(f"[worker] Overlay warning: {res.stderr[-300:]}", flush=True)
        except Exception as e:
            print(f"[worker] Overlay generation error: {e}", flush=True)

        final_status = "COMPLETED"
    else:
        print(f"[worker] Error: Chunks {failed_chunks} encountered failures.", flush=True)
        final_status = "FAILED"

    # Update status in DB
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        cur = conn.cursor()
        cur.execute("UPDATE files SET status=? WHERE id=?", (final_status, file_id))
        conn.commit()
        conn.close()
        print(f"[worker] File {file_id} marked {final_status} in database.", flush=True)
    except Exception as e:
        print(f"[worker] DB update error: {e}", flush=True)


def run_detection(video_path: str, file_id: int = None):
    """
    Launch 3 parallel detection chunks.

    Args:
        video_path: path to video file (e.g. 'videos/sample.mp4')
        file_id: row ID inserted in files table
    """
    if not os.path.exists(video_path):
        print(f"[worker] ERROR: Video not found: {video_path}")
        return

    script_path = os.path.join(SCRIPT_DIR, "test4_chunk.py")
    if not os.path.exists(script_path):
        print(f"[worker] ERROR: test4_chunk.py not found at {script_path}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_frames = _get_total_frames(video_path)
    if total_frames <= 0:
        print(f"[worker] ERROR: Cannot read frames from {video_path}")
        return

    print(f"[worker] {os.path.basename(video_path)}: {total_frames} frames, {NUM_CHUNKS} parallel chunks")

    chunk_size = total_frames // NUM_CHUNKS
    ranges = []
    for i in range(NUM_CHUNKS):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < NUM_CHUNKS - 1 else total_frames
        ranges.append((start, end))
        print(f"[worker] Chunk {i}: frames {start} -> {end}")

    python_exe = sys.executable
    processes = []

    for i, (start, end) in enumerate(ranges):
        cmd = [
            python_exe, script_path,
            video_path,
            str(i),
            str(start),
            str(end)
        ]
        print(f"[worker] Launching chunk {i}", flush=True)
        p = subprocess.Popen(cmd)
        processes.append(p)
        time.sleep(2)  # Stagger GPU allocations

    # Monitor in background thread
    threading.Thread(
        target=_wait_and_finalise,
        args=(video_path, file_id, processes),
        daemon=True
    ).start()