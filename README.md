# 🚦 Tamil Nadu Highways — Automated Traffic Detection System

An AI-powered traffic detection and vehicle classification system built for the Tamil Nadu Highways Department. Operators upload highway surveillance videos; the system detects, tracks, and counts vehicles by class using a fine-tuned YOLO model accelerated with TensorRT, then exports reports for department use.

---

## ✨ Features

- 🚗 Detects and classifies multiple vehicle types (2-wheelers to 5+ axle trucks)
- ⚡ **3× parallel chunk processing** — 1-hour video processed in ~25 minutes
- 📊 Live dashboard with real-time count chart during processing
- 📁 Export results as **PDF or Excel** reports
- 🎬 Annotated output video with bounding boxes + cumulative count overlay
- 🔐 Session-based login (admin / operator roles)
- 🗄️ SQLite database with full detection history

---

## 🏗️ System Architecture

```
User uploads video
        ↓
Flask (app.py) → inserts DB row (status=PROCESSING)
        ↓
worker.py splits video into 3 time-based chunks
        ↓
3 parallel Python processes (test4_chunk.py)
  Chunk 1 → frames 0–20 min   → output/video_chunk1.mp4
  Chunk 2 → frames 20–40 min  → output/video_chunk2.mp4
  Chunk 3 → frames 40–60 min  → output/video_chunk3.mp4
  All 3 write detections to SQLite simultaneously
        ↓
All chunks done → worker marks status=COMPLETED
        ↓
fix_count_overlay.py (post-processing):
  FFmpeg merges 3 chunks (no re-encode, ~5–10 sec)
  Renders cumulative count overlay (0 → total)
  → output/video_FINAL.mp4
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Web UI | Flask + Bootstrap 5 + Chart.js |
| Database | SQLite (WAL mode) |
| AI Detection | YOLO11s + BotSort tracker |
| GPU Acceleration | TensorRT (RTX 3060 12GB) |
| Video Processing | OpenCV + FFmpeg |

---

## ⚙️ Detection Configuration

These values are tuned for accuracy and should not be changed without re-validation:

```python
CONF_THRES        = 0.10   # Detection confidence threshold
HIGH_CONF_THRES   = 0.30   # High-confidence classification threshold
CLASS_DOMINANCE   = 0.7    # Minimum vote share for class assignment
MIN_CLASS_VOTES   = 8      # Minimum votes to assign a class
MIN_STABLE_FRAMES = 5      # Frames required before counting
MIN_DISPLACEMENT  = 7      # Minimum pixel movement to confirm motion
COUNT_Y_MIN       = 0.20   # Top of counting zone (% of frame height)
COUNT_Y_MAX       = 0.99   # Bottom of counting zone
TRACKER           = botsort.yaml
```

---

## 📁 Project Structure

```
trf/
├── app.py                  # Flask web server — all routes and UI logic
├── worker.py               # Launches 3 parallel detection subprocesses
├── test4_chunk.py          # Detection script — processes one chunk
├── fix_count_overlay.py    # Merges chunks + renders cumulative overlay
├── best.engine             # TensorRT compiled YOLO model (fast) ← not in repo
├── best.pt                 # PyTorch model fallback ← not in repo
├── trafficDetector.db      # SQLite database ← not in repo
├── templates/
│   ├── dashboard.html
│   ├── lookup.html
│   ├── upload.html
│   ├── progress.html
│   ├── files.html
│   └── results.html
├── static/
│   └── ...
└── output/                 # Generated videos ← not in repo
```

---

## 🖥️ UI Pages

| Page | Route | Description |
|---|---|---|
| Dashboard | `/site` | Live vehicle counts + chart during processing |
| Lookup | `/` | Search by date, time, camera, location |
| Upload | `/upload` | Register a video for detection |
| Progress | `/progress` | Live detection progress with chart |
| Files | `/files` | All videos, counts, PDF/Excel download, delete |
| Results | `/results` | Bar chart + breakdown per vehicle class |

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- TensorRT installed
- FFmpeg installed (`winget install ffmpeg` on Windows)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/tn-highways-traffic-detection.git
cd tn-highways-traffic-detection
```

### 2. Install Python Dependencies

```bash
pip install flask opencv-python ultralytics torch torchvision \
            tensorrt pycuda reportlab openpyxl
```

### 3. Add Model Files

Download `best.engine` (TensorRT) and `best.pt` (PyTorch fallback) and place them in the project root.

> Model files are not included in this repository due to size. Contact the project maintainer or see [Releases](#).

### 4. Initialize the Database

The database is created automatically on first run.

### 5. Run the Flask App

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## 📹 Processing a Video

### Step 1 — Upload
Go to `/upload`, enter the video filename, camera ID, location, and date/time.

### Step 2 — Detection runs automatically
Three parallel processes handle 0–20 min, 20–40 min, and 40–60 min chunks simultaneously. Monitor live on the `/progress` or `/site` dashboard.

### Step 3 — Merge and overlay (manual)
After detection completes, run:

```bash
python fix_count_overlay.py your_video.mp4
```

This merges the 3 chunks and renders the cumulative count overlay into `output/your_video_FINAL.mp4`.

### Step 4 — View results
Go to `/results` or `/files` to view counts, download reports, or play the annotated video.

---

## ⏱️ Performance

| Configuration | 1-hour video processing time |
|---|---|
| Original sequential | ~1 hr 30 min |
| 3 parallel chunks | ~20–25 min |
| + overlay render | +2–3 min |
| **Total** | **~25 min** |

**Speedup: ~3–4× faster than sequential processing**

---

## 📤 Output Files

| File | Description |
|---|---|
| `output/video_chunk1.mp4` | Chunk 1 annotated with bounding boxes |
| `output/video_chunk2.mp4` | Chunk 2 annotated with bounding boxes |
| `output/video_chunk3.mp4` | Chunk 3 annotated with bounding boxes |
| `output/video_FINAL.mp4` | Merged full video with cumulative count overlay |
| `trafficDetector.db` | All detections with timestamps |

---

## 🔧 Troubleshooting

**Processing stuck at PROCESSING status after a crash:**

```sql
UPDATE videos SET status='FAILED' WHERE status='PROCESSING';
```

Run this in DB Browser for SQLite or via Python's `sqlite3` module.

**TensorRT engine not loading:**
The system automatically falls back to `best.pt` (PyTorch). Re-compile the engine with:
```bash
yolo export model=best.pt format=engine device=0
```

---

## 🔐 Login

Default credentials (change before production deployment):

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | _(set in app.py)_ |
| Operator | `operator` | _(set in app.py)_ |

---

## 📄 License

This project was developed for the **Tamil Nadu Highways Department**. All rights reserved.

---

## 👤 Author

Developed by ** Roshan Joel K**
Tamil Nadu Highways Department — Traffic Monitoring Division

> For queries, contact: joelrj700@gmail.com
