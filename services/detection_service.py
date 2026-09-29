import os
import threading
import time
from database import get_db
from services.analytics_service import norm_dt, summary_window
from worker import run_detection

class DetectionManager:
    """Thread-safe manager for traffic detection background tasks and live state."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._status = {
            "running": False,
            "message": "Idle",
            "current_file_id": None,
            "current_video": None,
            "current_camera": None,
            "current_location": None,
            "current_start_dt": None,
        }

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def update_status(self, **kwargs):
        with self._lock:
            self._status.update(kwargs)

    def start_detection(self, video_path: str, file_id: int, video_name: str,
                        camera: str, location: str, start_dt: str):
        """Register job and spawn background execution thread."""
        self.update_status(
            running=True,
            message="Detection started",
            current_file_id=file_id,
            current_video=video_name,
            current_camera=camera,
            current_location=location,
            current_start_dt=start_dt,
        )

        worker_thread = threading.Thread(
            target=self._run_and_finalise,
            args=(video_path, file_id),
            daemon=True
        )
        worker_thread.start()

    def _run_and_finalise(self, video_path: str, file_id: int):
        """Worker thread that executes detection and waits for worker completion."""
        try:
            run_detection(video_path, file_id)
        except Exception as e:
            print(f"[detection_service] Execution error for file {file_id}:", e)

        print(f"[detection_service] Waiting for chunks to complete for file {file_id}...")
        for _ in range(7200):  # max 2 hours
            try:
                conn = get_db()
                cur = conn.cursor()
                cur.execute("SELECT status FROM files WHERE id=?", (file_id,))
                row = cur.fetchone()
                conn.close()
                if row and row[0] == "COMPLETED":
                    print(f"[detection_service] File {file_id} marked COMPLETED.")
                    break
            except Exception:
                pass
            time.sleep(5)

        self.update_status(running=False, message="Detection completed")

    def build_live_payload(self) -> dict:
        """Construct the live payload for dashboard and progress monitoring."""
        status = self.get_status()
        is_running = status.get("running", False)
        cam = status.get("current_camera")
        loc = status.get("current_location")
        start_raw = status.get("current_start_dt")
        video = status.get("current_video")
        file_id = status.get("current_file_id")

        # If not running in memory (e.g. after server reload), check DB for active PROCESSING file
        if not is_running:
            try:
                conn = get_db()
                cur = conn.cursor()
                cur.execute("""
                    SELECT id, fileName, camera, location, startDATE
                    FROM files WHERE status='PROCESSING'
                    ORDER BY id DESC LIMIT 1
                """)
                row = cur.fetchone()
                conn.close()
                if row:
                    file_id, video, cam, loc, start_raw = row
                    is_running = True
                    self.update_status(
                        running=True,
                        message="Detection running",
                        current_file_id=file_id,
                        current_video=video,
                        current_camera=cam,
                        current_location=loc,
                        current_start_dt=start_raw,
                    )
            except Exception as e:
                print("[detection_service] DB recovery error:", e)

        payload = {
            "running": is_running,
            "message": status.get("message", "Idle"),
            "video": video,
            "camera": cam,
            "location": loc,
            "file_id": file_id,
            "total_count": 0,
            "vehicle_counts": {},
        }

        if cam and loc and start_raw:
            try:
                sdt = norm_dt(start_raw)
                rows = summary_window(cam, loc, sdt, "2999-12-31 23:59:59")
                payload["vehicle_counts"] = {v: c for v, c in rows}
                payload["total_count"] = sum(payload["vehicle_counts"].values())
            except Exception as e:
                print("[detection_service] Live counts error:", e)

        return payload

# Global singleton instance
detection_manager = DetectionManager()
