"""
test4_chunk.py — Chunked detection.
Writes bounding boxes on chunk video BUT NO count overlay text.
The count overlay is added later by fix_count_overlay.py (cumulative).
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

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

INPUT_VIDEO = sys.argv[1]
CHUNK_ID    = int(sys.argv[2])
START_FRAME = int(sys.argv[3])
END_FRAME   = int(sys.argv[4])

_base            = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
CHUNK_VIDEO_NAME = f"{_base}_chunk{CHUNK_ID + 1}"

if os.path.exists("best.engine"):
    MODEL_PATH = r"best.engine"
    MODEL_TYPE = "TensorRT"
elif os.path.exists("best.pt"):
    MODEL_PATH = r"best.pt"
    MODEL_TYPE = "PyTorch"
else:
    print("ERROR: no model found"); sys.exit(1)

OUTPUT_DIR = r"output"
DB_PATH    = r"trafficDetector.db"

WRITE_OUTPUT_VIDEO = True
DB_BATCH_SIZE      = 20
TRACKER            = "botsort.yaml"
IMG_SIZE           = 640

NEW_WIDTH, NEW_HEIGHT = 1080, 720
CONF_THRES        = 0.05
HIGH_CONF_THRES   = 0.20
CLASS_DOMINANCE   = 0.7
MIN_CLASS_VOTES   = 8
MIN_STABLE_FRAMES = 5
MIN_DISPLACEMENT  = 7
COUNT_Y_MIN       = 0.50
COUNT_Y_MAX       = 0.99


def in_count_zone(cx, cy, h):
    return COUNT_Y_MIN <= (cy / h) <= COUNT_Y_MAX


def draw_zone(img, w, h):
    y1 = int(COUNT_Y_MIN * h)
    y2 = int(COUNT_Y_MAX * h)
    cv2.rectangle(img, (0, y1), (w, y2), (0, 255, 255), 3)


def get_metadata(video_path):
    video_name = os.path.basename(video_path)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        SELECT camera, phase, location, startDATE
        FROM files WHERE fileName = ?
        ORDER BY id DESC LIMIT 1
    """, (video_name,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise RuntimeError(f"Metadata not found: {video_name}")
    camera, phase, location, start_dt = row
    start_dt = start_dt.replace("T", " ")
    if len(start_dt) == 16:
        start_dt += ":00"
    return camera, phase, location, datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S")


class DBBatcher:
    def __init__(self, db_path, batch_size=20):
        self.conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.cur   = self.conn.cursor()
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
        for attempt in range(3):
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
                if "locked" in str(e) and attempt < 2:
                    import time as _t; _t.sleep(0.5)
                    continue
                raise

    def close(self):
        self.flush()
        self.conn.close()


def process_chunk():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[CHUNK {CHUNK_ID}] frames {START_FRAME}->{END_FRAME}  model={MODEL_TYPE}", flush=True)

    camera, phase, location, video_start_time = get_metadata(INPUT_VIDEO)

    cap   = cv2.VideoCapture(INPUT_VIDEO)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    need_resize = (src_w > NEW_WIDTH or src_h > NEW_HEIGHT)
    out_w = NEW_WIDTH if need_resize else src_w
    out_h = NEW_HEIGHT if need_resize else src_h

    chunk_video_path = os.path.join(OUTPUT_DIR, f"{CHUNK_VIDEO_NAME}.mp4")
    writer = cv2.VideoWriter(
        chunk_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (out_w, out_h)
    )
    if not writer.isOpened():
        print(f"[CHUNK {CHUNK_ID}] ERROR: writer failed", flush=True)
        writer = None
    else:
        print(f"[CHUNK {CHUNK_ID}] writing to {chunk_video_path}", flush=True)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CHUNK {CHUNK_ID}] device={device}", flush=True)

    model    = YOLO(MODEL_PATH)
    use_fp16 = device == "cuda"
    if MODEL_PATH.endswith(".pt"):
        model.fuse(); model.to(device)
        if use_fp16: model.model.half()

    track_history  = defaultdict(lambda: deque(maxlen=MIN_STABLE_FRAMES))
    track_votes    = defaultdict(lambda: defaultdict(int))
    track_max_conf = defaultdict(float)
    permanent_ids  = {}
    total_counts   = defaultdict(int)
    next_id        = 1

    db_batcher  = DBBatcher(DB_PATH, DB_BATCH_SIZE)
    chunk_total = END_FRAME - START_FRAME
    prog_step   = max(1, chunk_total // 10)
    frame_index = START_FRAME
    processed   = 0

    try:
        while frame_index < END_FRAME:
            ret, frame = cap.read()
            if not ret: break
            frame_index += 1
            processed   += 1

            if need_resize:
                frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT))

            if writer:
                draw_zone(frame, out_w, out_h)

            results = model.track(
                frame, persist=True, conf=CONF_THRES, imgsz=IMG_SIZE,
                tracker=TRACKER, device=device, half=use_fp16, verbose=False
            )[0]

            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                clss  = results.boxes.cls.cpu().numpy()
                tids  = results.boxes.id.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()

                for box, cls_id, tid, conf in zip(boxes, clss, tids, confs):
                    tid      = int(tid)
                    cls_name = model.names[int(cls_id)]
                    if cls_name.lower() == "person": continue

                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    track_votes[tid][cls_name] += 1
                    total_votes = sum(track_votes[tid].values())
                    best_class  = max(track_votes[tid], key=track_votes[tid].get)
                    dominance   = track_votes[tid][best_class] / total_votes
                    track_max_conf[tid] = max(track_max_conf[tid], float(conf))

                    track_history[tid].append((cx, cy))
                    if len(track_history[tid]) >= MIN_STABLE_FRAMES:
                        dx   = track_history[tid][-1][0] - track_history[tid][0][0]
                        dy   = track_history[tid][-1][1] - track_history[tid][0][1]
                        disp = np.sqrt(dx*dx + dy*dy)
                    else:
                        disp = 0

                    if tid not in permanent_ids:
                        if (disp > MIN_DISPLACEMENT
                                and total_votes >= MIN_CLASS_VOTES
                                and dominance   >= CLASS_DOMINANCE
                                and track_max_conf[tid] >= HIGH_CONF_THRES
                                and in_count_zone(cx, cy, out_h)):
                            permanent_ids[tid] = next_id
                            total_counts[best_class] += 1
                            event_time = video_start_time + timedelta(seconds=frame_index / fps)
                            db_batcher.add(event_time, best_class, camera, phase, location)
                            next_id += 1

                    if writer and tid in permanent_ids and track_max_conf[tid] >= HIGH_CONF_THRES:
                        label = f"ID:{permanent_ids[tid]} {best_class}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, max(20, y1-8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # ===== CHUNK VEHICLE COUNT =====
            if writer:
                y = 30

                for k, v in total_counts.items():
                    if v > 0:
                        cv2.putText(
                            frame,
                            f"{k} - {v}",
                            (15, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
                        y += 25

                writer.write(frame)

            if processed % prog_step == 0:
                pct = (processed / chunk_total) * 100
                print(f"[CHUNK {CHUNK_ID}] {pct:5.1f}%  vehicles={sum(total_counts.values())}", flush=True)

    finally:
        db_batcher.close()
        cap.release()
        if writer: writer.release()

    print(f"[CHUNK {CHUNK_ID}] DONE  vehicles={sum(total_counts.values())}", flush=True)


if __name__ == "__main__":
    process_chunk()