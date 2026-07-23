import os
import sys
import cv2
import torch
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from ultralytics import YOLO
from collections import defaultdict, deque

# ================== GPU OPTIMIZATION ==================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# =====================================================

# ================== CONFIG ==================
MODEL_PATH = r"best.pt"
INPUT_VIDEO = sys.argv[1]
OUTPUT_DIR = r"output"
DB_PATH = r"trafficDetector.db"

NEW_WIDTH, NEW_HEIGHT = 1080, 720

CONF_THRES = 0.10
HIGH_CONF_THRES = 0.30

CLASS_DOMINANCE = 0.7
MIN_CLASS_VOTES = 8

MIN_STABLE_FRAMES = 5
MIN_DISPLACEMENT = 7

COUNT_Y_MIN = 0.20
COUNT_Y_MAX = 0.99
# ============================================


def in_count_zone(cx, cy, h):
    return COUNT_Y_MIN <= (cy / h) <= COUNT_Y_MAX


def draw_zone(img, w, h):
    y1 = int(COUNT_Y_MIN * h)
    y2 = int(COUNT_Y_MAX * h)
    cv2.rectangle(img, (0, y1), (w, y2), (0, 255, 255), 3)


# ================== DB LOGIC (FROM test2) ==================
def get_metadata(video_path):
    video_name = os.path.basename(video_path)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT camera, phase, location, startDATE
        FROM files
        WHERE fileName = ?
        ORDER BY id DESC
        LIMIT 1
    """, (video_name,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise RuntimeError("Metadata not found for video")

    camera, phase, location, start_dt = row
    video_start_time = datetime.strptime(start_dt, "%Y-%m-%dT%H:%M")
    return camera, phase, location, video_start_time


def save_to_db(event_time, vehicle, camera, phase, location):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO trafficClassification
        (time, camera, vehicle, phase, location)
        VALUES (?, ?, ?, ?, ?)
    """, (
        event_time.strftime("%Y-%m-%d %H:%M:%S"),
        camera, vehicle, phase, location
    ))
    conn.commit()
    conn.close()
# =========================================================


def process():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    camera, phase, location, video_start_time = get_metadata(INPUT_VIDEO)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    base_name = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_counted.mp4")

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (NEW_WIDTH, NEW_HEIGHT)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    model = YOLO(MODEL_PATH)
    model.fuse()
    model.to(device)

    use_fp16 = device == "cuda"
    if use_fp16:
        model.model.half()

    track_history = defaultdict(lambda: deque(maxlen=MIN_STABLE_FRAMES))
    track_votes = defaultdict(lambda: defaultdict(int))
    track_max_conf = defaultdict(float)

    permanent_ids = {}
    total_counts = defaultdict(int)
    next_id = 1

    frame_index = 0
    print("[INFO] Detection started")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT))
        draw_zone(frame, NEW_WIDTH, NEW_HEIGHT)

        results = model.track(
            frame,
            persist=True,
            conf=CONF_THRES,
            imgsz=640,
            tracker="botsort.yaml",
            device=device,
            half=use_fp16,
            verbose=False
        )[0]

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()
            tids = results.boxes.id.cpu().numpy()
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
                best_class = max(track_votes[tid], key=track_votes[tid].get)
                dominance = track_votes[tid][best_class] / total_votes

                track_max_conf[tid] = max(track_max_conf[tid], float(conf))

                track_history[tid].append((cx, cy))
                if len(track_history[tid]) >= MIN_STABLE_FRAMES:
                    dx = track_history[tid][-1][0] - track_history[tid][0][0]
                    dy = track_history[tid][-1][1] - track_history[tid][0][1]
                    disp = np.sqrt(dx * dx + dy * dy)
                else:
                    disp = 0

                if tid not in permanent_ids:
                    if (
                        disp > MIN_DISPLACEMENT
                        and total_votes >= MIN_CLASS_VOTES
                        and dominance >= CLASS_DOMINANCE
                        and track_max_conf[tid] >= HIGH_CONF_THRES
                        and in_count_zone(cx, cy, NEW_HEIGHT)
                    ):
                        permanent_ids[tid] = next_id
                        total_counts[best_class] += 1

                        event_time = video_start_time + timedelta(seconds=frame_index / fps)
                        save_to_db(event_time, best_class, camera, phase, location)

                        next_id += 1

                if tid in permanent_ids and track_max_conf[tid] >= HIGH_CONF_THRES:
                    label = f"ID:{permanent_ids[tid]} {best_class}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 255), 2)

        y = 40
        cv2.putText(frame, "VEHICLE COUNT", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        y += 40

        for k, v in total_counts.items():
            if v > 0:
                cv2.putText(frame, f"{k}: {v}", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
                y += 30

        cv2.putText(frame, f"TOTAL: {sum(total_counts.values())}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 255), 3)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[DONE] Saved → {out_path}")


if __name__ == "__main__":
    process()
