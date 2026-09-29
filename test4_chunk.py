"""
test4_chunk.py — Chunked traffic detection engine.
Processes a frame segment of a video using YOLO11 + BotSort.
Annotates bounding boxes on the chunk video.
Cumulative count overlay is cleanly rendered during post-processing by fix_count_overlay.py.
"""
import os
import sys
import cv2
import torch
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from ultralytics import YOLO
from collections import defaultdict, deque

# CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Directories & DB
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
DB_PATH    = os.path.join(SCRIPT_DIR, "trafficDetector.db")

# Model Loading with resilient fallback
ENGINE_PATH = os.path.join(SCRIPT_DIR, "best.engine")
PT_PATH     = os.path.join(SCRIPT_DIR, "best.pt")

MODEL_PATH = None
MODEL_TYPE = "None"
if os.path.exists(ENGINE_PATH):
    MODEL_PATH = ENGINE_PATH
    MODEL_TYPE = "TensorRT"
elif os.path.exists(PT_PATH):
    MODEL_PATH = PT_PATH
    MODEL_TYPE = "PyTorch"
else:
    print("ERROR: No YOLO model found (neither best.engine nor best.pt exist).")
    sys.exit(1)

# Detection & Tracking Parameters
CONF_THRES        = 0.15   # Filter background noise for faster, accurate tracking
HIGH_CONF_THRES   = 0.30   # Minimum peak confidence for qualification
CLASS_DOMINANCE   = 0.70   # Dominant class vote ratio required
MIN_CLASS_VOTES   = 6      # Minimum detection votes before counting
MIN_STABLE_FRAMES = 5      # Centroid history frames for displacement check
MIN_DISPLACEMENT  = 8      # Minimum pixel movement to filter stationary artifacts
COUNT_Y_MIN       = 0.40   # Counting zone upper boundary (40% from top)
COUNT_Y_MAX       = 0.95   # Counting zone lower boundary (95% to avoid edge cutoff)

TRACKER           = "botsort.yaml"
IMG_SIZE          = 640
DB_BATCH_SIZE     = 20
MAX_OUT_WIDTH     = 1280
MAX_OUT_HEIGHT    = 720


def in_count_zone(cy: float, h: int) -> bool:
    """Check if centroid y-coordinate falls within the counting zone."""
    return COUNT_Y_MIN <= (cy / h) <= COUNT_Y_MAX


def draw_zone(img, w: int, h: int):
    """Draw a clean, semi-transparent counting zone overlay with boundary markers."""
    y1 = int(COUNT_Y_MIN * h)
    y2 = int(COUNT_Y_MAX * h)
    
    # Boundary guideline lines
    cv2.line(img, (0, y1), (w, y1), (0, 220, 255), 2)
    cv2.line(img, (0, y2), (w, y2), (0, 165, 255), 2)
    
    # Zone label
    cv2.putText(img, "DETECTION ZONE", (w - 180, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)


def get_metadata(video_path: str):
    video_name = os.path.basename(video_path)
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cur  = conn.cursor()
    cur.execute("""
        SELECT camera, phase, location, startDATE
        FROM files WHERE fileName = ?
        ORDER BY id DESC LIMIT 1
    """, (video_name,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise RuntimeError(f"Metadata not found in DB for: {video_name}")
    camera, phase, location, start_dt = row
    start_dt = start_dt.replace("T", " ")
    if len(start_dt) == 16:
        start_dt += ":00"
    return camera, phase, location, datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S")


class DBBatcher:
    """Thread-safe and locked-resilient batch inserter into SQLite."""
    def __init__(self, db_path: str, batch_size: int = 20):
        self.conn = sqlite3.connect(db_path, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.cur = self.conn.cursor()
        self.batch = []
        self.batch_size = batch_size

    def add(self, event_time, vehicle, camera, phase, location):
        self.batch.append((
            event_time.strftime("%Y-%m-%d %H:%M:%S"),
            camera, vehicle, phase, location
        ))
        if len(self.batch) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.batch:
            return
        import time as _t
        for attempt in range(5):
            try:
                self.cur.executemany("""
                    INSERT INTO trafficClassification
                    (time, camera, vehicle, phase, location)
                    VALUES (?, ?, ?, ?, ?)
                """, self.batch)
                self.conn.commit()
                self.batch.clear()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < 4:
                    _t.sleep(0.5)
                    continue
                raise

    def close(self):
        self.flush()
        self.conn.close()


def process_chunk(input_video: str, chunk_id: int, start_frame: int, end_frame: int):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _base = os.path.splitext(os.path.basename(input_video))[0]
    chunk_video_name = f"{_base}_chunk{chunk_id + 1}"
    print(f"[CHUNK {chunk_id}] Range: frames {start_frame} -> {end_frame} | Model: {MODEL_TYPE}", flush=True)

    camera, phase, location, video_start_time = get_metadata(input_video)

    cap   = cv2.VideoCapture(input_video)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Maintain true aspect ratio
    if src_w > MAX_OUT_WIDTH or src_h > MAX_OUT_HEIGHT:
        scale = min(MAX_OUT_WIDTH / src_w, MAX_OUT_HEIGHT / src_h)
        out_w = int(src_w * scale)
        out_h = int(src_h * scale)
        # Ensure dimensions are even numbers for MP4 encoders
        out_w = out_w if out_w % 2 == 0 else out_w - 1
        out_h = out_h if out_h % 2 == 0 else out_h - 1
        need_resize = True
    else:
        out_w, out_h = src_w, src_h
        need_resize = False

    chunk_video_path = os.path.join(OUTPUT_DIR, f"{chunk_video_name}.mp4")
    writer = cv2.VideoWriter(
        chunk_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (out_w, out_h)
    )
    if not writer.isOpened():
        print(f"[CHUNK {chunk_id}] ERROR: VideoWriter failed for {chunk_video_path}", flush=True)
        writer = None
    else:
        print(f"[CHUNK {chunk_id}] Output: {chunk_video_path} ({out_w}x{out_h} @ {fps:.1f} fps)", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CHUNK {chunk_id}] Device: {device}", flush=True)

    # Initialize and validate YOLO model with resilient fallback
    model = None
    active_model_path = MODEL_PATH

    for candidate_path in [MODEL_PATH, PT_PATH]:
        if not candidate_path or not os.path.exists(candidate_path):
            continue
        try:
            m = YOLO(candidate_path)
            # Warm-up / validation test to verify TensorRT deserialization or PyTorch weights on GPU
            dummy_frame = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            m.predict(dummy_frame, device=device, verbose=False)
            model = m
            active_model_path = candidate_path
            break
        except Exception as e:
            print(f"[CHUNK {chunk_id}] Notice: {os.path.basename(candidate_path)} cannot be loaded ({e}). Trying fallback...", flush=True)

    if model is None:
        print(f"[CHUNK {chunk_id}] ERROR: Could not initialize any YOLO model.", flush=True)
        sys.exit(1)

    print(f"[CHUNK {chunk_id}] Active Model: {os.path.basename(active_model_path)} on {device}", flush=True)

    # Track states
    track_history   = defaultdict(lambda: deque(maxlen=MIN_STABLE_FRAMES))
    track_votes     = defaultdict(lambda: defaultdict(int))
    track_max_conf  = defaultdict(float)
    track_last_seen = {}
    permanent_ids   = {}
    total_counts    = defaultdict(int)
    next_id         = 1

    db_batcher  = DBBatcher(DB_PATH, DB_BATCH_SIZE)
    chunk_total = max(1, end_frame - start_frame)
    prog_step   = max(1, chunk_total // 10)
    frame_index = start_frame
    processed   = 0

    try:
        while frame_index < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            processed   += 1

            if need_resize:
                frame = cv2.resize(frame, (out_w, out_h))

            if writer:
                draw_zone(frame, out_w, out_h)

            results = model.track(
                frame,
                persist=True,
                conf=CONF_THRES,
                imgsz=IMG_SIZE,
                tracker=TRACKER,
                device=device,
                verbose=False
            )[0]

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                clss  = results.boxes.cls.cpu().numpy()
                tids  = results.boxes.id.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()

                for box, cls_id, tid, conf in zip(boxes, clss, tids, confs):
                    tid = int(tid)
                    cls_name = model.names[int(cls_id)]
                    if cls_name.lower() == "person":
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    track_votes[tid][cls_name] += 1
                    total_votes = sum(track_votes[tid].values())
                    best_class  = max(track_votes[tid], key=track_votes[tid].get)
                    dominance   = track_votes[tid][best_class] / total_votes
                    track_max_conf[tid] = max(track_max_conf[tid], float(conf))
                    track_last_seen[tid] = frame_index

                    track_history[tid].append((cx, cy))
                    if len(track_history[tid]) >= MIN_STABLE_FRAMES:
                        dx   = track_history[tid][-1][0] - track_history[tid][0][0]
                        dy   = track_history[tid][-1][1] - track_history[tid][0][1]
                        disp = np.sqrt(dx * dx + dy * dy)
                    else:
                        disp = 0.0

                    # Qualification for counting
                    if tid not in permanent_ids:
                        if (disp >= MIN_DISPLACEMENT
                                and total_votes >= MIN_CLASS_VOTES
                                and dominance >= CLASS_DOMINANCE
                                and track_max_conf[tid] >= HIGH_CONF_THRES
                                and in_count_zone(cy, out_h)):
                            permanent_ids[tid] = next_id
                            total_counts[best_class] += 1
                            event_time = video_start_time + timedelta(seconds=frame_index / fps)
                            db_batcher.add(event_time, best_class, camera, phase, location)
                            next_id += 1

                    # Clean visual annotations
                    if writer:
                        if tid in permanent_ids:
                            # Confirmed counted vehicle: Solid Green Box + Label Badge
                            label = f"ID:{permanent_ids[tid]} {best_class}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                            badge_y1 = max(0, y1 - th - 6)
                            cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 6, badge_y1 + th + 6), (0, 200, 0), -1)
                            cv2.putText(frame, label, (x1 + 3, badge_y1 + th + 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                        elif track_max_conf[tid] >= HIGH_CONF_THRES:
                            # Tracked vehicle before qualification: Thin Cyan Box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)

            # Write annotated frame (NO duplicate chunk counts burned into video)
            if writer:
                writer.write(frame)

            # Periodic memory cleanup every 500 frames for inactive tracks
            if processed % 500 == 0:
                stale_tids = [t for t, last_f in track_last_seen.items() if frame_index - last_f > 150]
                for st in stale_tids:
                    track_votes.pop(st, None)
                    track_history.pop(st, None)
                    track_max_conf.pop(st, None)
                    track_last_seen.pop(st, None)

            if processed % prog_step == 0:
                pct = (processed / chunk_total) * 100
                print(f"[CHUNK {chunk_id}] {pct:5.1f}%  counted={sum(total_counts.values())}", flush=True)

    finally:
        db_batcher.close()
        cap.release()
        if writer:
            writer.release()

    print(f"[CHUNK {chunk_id}] Completed! Total vehicles counted: {sum(total_counts.values())}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python test4_chunk.py <video_path> <chunk_id> <start_frame> <end_frame>")
        sys.exit(1)
    process_chunk(
        input_video=sys.argv[1],
        chunk_id=int(sys.argv[2]),
        start_frame=int(sys.argv[3]),
        end_frame=int(sys.argv[4])
    )