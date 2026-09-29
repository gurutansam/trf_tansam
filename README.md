# 🚦 Tamil Nadu Highways — Automated Traffic Detection System

An AI-powered traffic detection and vehicle classification system built for the **Tamil Nadu Highways Department**. Operators upload highway surveillance videos; the system detects, tracks, and counts vehicles by class using a fine-tuned YOLO model accelerated with **TensorRT** on an NVIDIA RTX 3060, then exports reports as PDF or Excel for department analytics.

---

## ✨ Features

- 🚗 **Multi-Class Vehicle Classification**: Detects and classifies 27 vehicle categories (2-wheelers, cars, buses, rigid trucks, trailer trucks, LCVs, etc.).
- ⚡ **High-Speed TensorRT Acceleration**: Runs at **~4.3 ms per frame (~230 FPS)** on an NVIDIA RTX 3060 GPU.
- 🔀 **Parallel Chunk Processing**: Automatically partitions videos into parallel chunks for 3–5× speedups.
- 🎬 **End-to-End Automated Pipeline**: Chunk detection, SQLite logging, video concatenation, and cumulative counter overlay execute end-to-end automatically.
- 📐 **Aspect-Ratio Preservation**: Automatically preserves native 16:9 surveillance aspect ratios without vertical stretching or distortion.
- 📊 **Live Dashboard & Progress**: Real-time vehicle count analytics and live charts during detection.
- 📁 **Instant Exports**: One-click **PDF** (ReportLab) and **Excel** (OpenPyXL) summary and detailed logs.
- 🔐 **Secure Authentication**: Session-based login with hashed passwords and protected routes.
- 🧱 **Modular Architecture**: Built with Flask Blueprints, application factory, and dedicated service layers.

---

## 🏗️ System Architecture

```text
User uploads video to videos/
        ↓
Flask UI (/upload) → inserts DB row (status=PROCESSING)
        ↓
worker.py splits video into parallel chunks
        ↓
Parallel Python processes (test4_chunk.py)
  Chunk 1 → frames 0–33%   → output/video_chunk1.mp4
  Chunk 2 → frames 33–66%  → output/video_chunk2.mp4
  Chunk 3 → frames 66–100% → output/video_chunk3.mp4
  All chunks stream detections to SQLite (WAL mode)
        ↓
All chunks finish successfully
        ↓
worker.py automatically triggers fix_count_overlay.py:
  1. Fast FFmpeg stream concat (no re-encoding)
  2. Renders high-contrast cumulative count card overlay (0 → total)
  → output/video_FINAL.mp4
        ↓
worker.py marks DB status as COMPLETED
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Web UI** | Flask 3.1 + Bootstrap 5 + Chart.js + Jinja2 |
| **Architecture** | Flask Blueprints + Application Factory Pattern |
| **Database** | SQLite (WAL mode, busy timeout, thread-safe batching) |
| **AI Detection** | YOLO11s + BotSort Tracker |
| **GPU Acceleration** | TensorRT 10.16 (FP16 Tensor Cores, RTX 3060 12GB) |
| **Video Processing** | OpenCV + FFmpeg |
| **Reports** | ReportLab (PDF) + OpenPyXL (Excel) |

---

## ⚙️ Detection Configuration

Tuned in `test4_chunk.py` for maximum accuracy, noise reduction, and high throughput:

```python
CONF_THRES        = 0.15   # Detection threshold (filters background noise, speeds up tracker)
HIGH_CONF_THRES   = 0.30   # High-confidence threshold required before counting
CLASS_DOMINANCE   = 0.70   # Dominant class vote ratio required (70% consensus)
MIN_CLASS_VOTES   = 6      # Minimum detection votes before counting
MIN_STABLE_FRAMES = 5      # History frames for trajectory confirmation
MIN_DISPLACEMENT  = 8      # Minimum pixel movement to filter stationary artifacts
COUNT_Y_MIN       = 0.40   # Upper boundary of counting zone (% of frame height)
COUNT_Y_MAX       = 0.95   # Lower boundary of counting zone
MAX_OUT_WIDTH     = 1280   # Max video output width (preserves aspect ratio)
MAX_OUT_HEIGHT    = 720    # Max video output height
TRACKER           = "botsort.yaml"
```

---

## 📁 Project Structure

```text
trf_tansam/
├── app.py                      # Flask Application Factory (~34 lines)
├── config.py                   # Central system configuration
├── database.py                 # Unified SQLite manager with WAL mode
├── init_db.py                  # Database initialization & default account seeding
├── requirements.txt            # Pinned dependencies with CUDA 12.4 index
│
├── services/                   # Business Logic & Heavy Processing
│   ├── auth_service.py         # Password hashing & @login_required decorator
│   ├── analytics_service.py    # Query aggregations, metrics & time-window filters
│   ├── detection_service.py    # Thread-safe background execution & live state tracker
│   └── report_service.py       # In-memory PDF & Excel export engines
│
├── blueprints/                 # HTTP Presentation Layer (Routes & Controllers)
│   ├── auth.py                 # /login, /logout, /register, /forgot-password
│   ├── dashboard.py            # /, /site, /lookup
│   ├── files.py                # /files, /delete_file, /video/<filename>
│   ├── detection.py            # /upload, /progress, /progress/status
│   └── reports.py              # /results, /download_pdf, /download_excel, /file_pdf, /file_excel
│
├── worker.py                   # Parallel chunk detection & automated post-processor
├── test4_chunk.py              # YOLO / TensorRT chunk detection engine
├── fix_count_overlay.py        # Video stitching & cumulative count card overlay
│
├── best.engine                 # TensorRT compiled YOLO model (fastest)
├── best.pt                     # PyTorch model weights (resilient fallback)
├── trafficDetector.db          # SQLite database (history & detections)
│
├── videos/                     # Storage folder for uploaded raw videos
├── output/                     # Generated chunk & final annotated videos
├── templates/                  # Bootstrap 5 HTML templates
└── static/                     # CSS, JS, and branding images
```

---

## 🖥️ UI Pages & Routes

| Page | Route | Description |
|---|---|---|
| **Login** | `/login` | Secure session authentication |
| **Register** | `/register` | New user onboarding |
| **Dashboard** | `/site` | Overview metrics, live status, and camera stats |
| **Lookup** | `/lookup` | Filter detections by camera, location, date, and time |
| **Upload** | `/upload` | Register a surveillance video for automated detection |
| **Progress** | `/progress` | Real-time progress bar, vehicle breakdown, and chart |
| **Files** | `/files` | Upload history, per-video counts, PDF/Excel downloads, deletion |
| **Results** | `/results` | Aggregated bar charts and tabular vehicle breakdown |

---

## 🚀 Setup & Installation

### Prerequisites

- **OS**: Windows 10/11 or Linux
- **Python**: 3.10 or 3.11
- **GPU**: NVIDIA GPU with CUDA support (e.g., RTX 3060 12GB)
- **FFmpeg**: Installed and available in system PATH (`winget install ffmpeg` on Windows)

### 1. Clone & Enter Repository

```bash
git clone https://github.com/yourusername/tn-highways-traffic-detection.git
cd tn-highways-traffic-detection
```

### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv .venv
.\.venv\Scripts\activate

# Install all dependencies (includes PyTorch CUDA and TensorRT)
pip install -r requirements.txt
```

### 3. Initialize Database & Default Accounts

```bash
python init_db.py
```

### 4. Run the Application

```bash
python app.py
```

Open **`http://localhost:5000`** in your browser.

---

## 🔐 Default Login Credentials

| Role | Username / Email | Password |
|---|---|---|
| **Admin** | `admin` *(or `admin@example.com`)* | `admin123` *(or `admin`)* |
| **Operator** | `operator` *(or `operator@example.com`)* | `operator123` *(or `operator`)* |

*(You can also register custom accounts at `/register`).*

---

## 📹 Processing a Video

1. **Place Video**: Copy your video file (e.g. `highway_test.mp4`) into the **`videos/`** folder.
2. **Start Detection**: Navigate to **`/upload`**, enter the filename (`highway_test.mp4`), Camera ID, Location, and Start Date/Time, and click **Start Detection**.
3. **Monitor Live**: Watch detection progress and live count charts on **`/progress`** or **`/site`**.
4. **Automatic Final Video**: Once all parallel chunks complete, the system **automatically** stitches them together and burns in the cumulative count card at `output/highway_test_FINAL.mp4`.
5. **Download Reports**: Go to **`/files`** to view vehicle totals and download **PDF** or **Excel** reports.

---

## ⏱️ Benchmark Performance (RTX 3060 12GB)

| Inference Backend | Latency | Frame Rate (FPS) | 1-Hour Video (90k frames) |
|---|---|---|---|
| **PyTorch (`best.pt`)** | ~28.5 ms | ~35 FPS | ~20–25 min (3 chunks) |
| **TensorRT (`best.engine`)** | **4.34 ms** | **230.3 FPS** | **~6–8 min** (3 chunks) |

*Speedup: **~6.5× faster inference** using TensorRT FP16 Tensor Cores.*

---

## 📤 Output Files

| File | Description |
|---|---|
| `output/<name>_chunk1.mp4` | Chunk 1 annotated with bounding boxes and tracking IDs |
| `output/<name>_chunk2.mp4` | Chunk 2 annotated with bounding boxes and tracking IDs |
| `output/<name>_chunk3.mp4` | Chunk 3 annotated with bounding boxes and tracking IDs |
| `output/<name>_FINAL.mp4` | Stitched full video with high-contrast cumulative count card |
| `trafficDetector.db` | SQLite database storing all vehicle timestamp detections |

---

## 📄 License

This project was developed for the **Tamil Nadu Highways Department**. All rights reserved.

---

## 👤 Author

Developed by **Joel Roshan**  
Upgraded by **Gurucharan**  
**TANSAM**
