from cv2.utils import logging as cv2_logging  # silence OpenCV logs
import os
import uuid
import threading
from flask import Flask, render_template, redirect, url_for, request, jsonify, Response, send_file
from werkzeug.utils import secure_filename
import logging
import time
from faceblur_services import process_raw_video
from dotenv import load_dotenv
load_dotenv()
port = os.getenv("PORT")
app = Flask(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR      = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
MAX_VIDEO_BYTES = 100 * 1024 * 1024   # 100 MB

ALLOWED_VIDEO = {"mp4", "mov", "avi", "mkv"}
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── In-memory job store ───────────────────────────────────────────────────────
# { job_id: { "status": "queued|processing|done|error",
#             "progress": 0-100,
#             "message": str,
#             "output_path": str | None,
#             "error": str | None } }
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


def _run_job(job_id: str, video_path: str, face_paths: list, output_path: str, temp_path: str):
    """Background thread: runs processing and updates job state."""
    _update_job(job_id, status="processing", progress=5, message="Loading face database…")
    try:
        result = process_raw_video(
            video_path,
            face_paths,
            output_path,
            temp_path,
            progress_callback=lambda pct, msg: _update_job(
                job_id, progress=pct, message=msg
            ),
        )
        if result is None:
            _update_job(job_id, status="error", error="Processing returned no output.")
        else:
            _update_job(job_id, status="done", progress=100,
                        message="Done!", output_path=result)
    except Exception as exc:
        logger.exception(f"[job {job_id}] Failed")
        _update_job(job_id, status="error", error=str(exc))
    finally:
        # Always clean up input uploads
        for p in [video_path] + face_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    

# ── Helpers ──────────────────────────────────────────────────────────────────
def allowed(filename: str, allowed_set: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def root():
    return redirect(url_for("home"))


@app.route("/home")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """Validate uploads, save them, kick off background job, return job_id immediately."""
    video_file = request.files.get("video")
    face_files = request.files.getlist("reference_faces")

    if not video_file or not allowed(video_file.filename, ALLOWED_VIDEO):
        return jsonify({"error": "Invalid or missing video file."}), 400

    if not face_files or not all(allowed(f.filename, ALLOWED_IMAGE) for f in face_files):
        return jsonify({"error": "Please upload at least one valid face image (JPG/PNG/WebP)."}), 400

    video_file.seek(0, 2)
    video_size = video_file.tell()
    video_file.seek(0)
    if video_size > MAX_VIDEO_BYTES:
        return jsonify({"error": f"Video exceeds 100 MB ({video_size // (1024*1024)} MB uploaded)."}), 400

    uid        = uuid.uuid4().hex
    video_ext  = secure_filename(video_file.filename).rsplit(".", 1)[1].lower()
    video_path = os.path.join(UPLOAD_DIR, f"{uid}_input.{video_ext}")
    out_path   = os.path.join(OUTPUT_DIR,  f"{uid}_processed.mp4")
    temp_path  = os.path.join(OUTPUT_DIR,  f"{uid}_temp.mp4")
    face_paths: list[str] = []
    for i, ff in enumerate(face_files):
        ext = secure_filename(ff.filename).rsplit(".", 1)[1].lower()
        fp  = os.path.join(UPLOAD_DIR, f"{uid}_face_{i}.{ext}")
        ff.save(fp)
        face_paths.append(fp)
    video_file.save(video_path)

    # Register job
    with _jobs_lock:
        _jobs[uid] = {
            "status": "queued",
            "progress": 0,
            "message": "Queued…",
            "output_path": None,
            "error": None,
        }

    # Start background thread
    t = threading.Thread(
        target=_run_job,
        args=(uid, video_path, face_paths, out_path, temp_path),
        daemon=True,
    )
    t.start()
    logger.info(f"[job {uid}] started — video={video_path}, faces={len(face_paths)}")

    return jsonify({"job_id": uid}), 202


@app.route("/status/<job_id>")
def status(job_id: str):
    """Return current job status and progress."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Job not found."}), 404
    return jsonify({
        "status":   job["status"],
        "progress": job["progress"],
        "message":  job["message"],
        "error":    job["error"],
    })


@app.route("/download/<job_id>")
def download(job_id: str):
    """Serve the processed video and clean up the job."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        return jsonify({"error": "Job not found."}), 404
    if job["status"] != "done":
        return jsonify({"error": "Job not ready yet."}), 400

    output_path = job["output_path"]
    if not output_path or not os.path.exists(output_path):
        return jsonify({"error": "Output file missing."}), 500

    directory = os.path.dirname(output_path)
    filename = os.path.basename(output_path)

    try:
        # Use send_from_directory for robust file serving
        response = send_file(
            output_path,
            mimetype="video/mp4",
            as_attachment=True,
            download_name="vlogguard_processed.mp4"
        )
        
        # Add CORS headers in case the browser is strict about IP origins
        response.headers["Access-Control-Allow-Origin"] = "*"
            
        return response
        
    except Exception as exc:
        logger.error(f"Download route error: {exc}")
        return jsonify({"error": f"Download failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, threaded=True)
    # app.run()
