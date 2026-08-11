# Thin frontend.
# Upload -> queue -> results page. No model/data logic here -- everything real lives in
# src/foretrack/demo_pipeline.py, run by worker.py, not this process.

import json
import uuid
from pathlib import Path

import config
from flask import Flask, redirect, render_template, request, send_file, url_for

app = Flask(__name__)
Path(config.JOBS_DIR).mkdir(parents=True, exist_ok=True)


@app.route("/")
def upload_form():
    return render_template("upload.html", default_frac=config.DEFAULT_CONDITIONING_FRAC, max_duration_s=15)


@app.route("/upload", methods=["POST"])
def upload():
    video = request.files.get("video")
    if not video or video.filename == "":
        return render_template("upload.html", default_frac=config.DEFAULT_CONDITIONING_FRAC, max_duration_s=15, error="choose a video file first"), 400

    try:
        conditioning_frac = float(request.form.get("conditioning_frac", config.DEFAULT_CONDITIONING_FRAC))
    except ValueError:
        conditioning_frac = config.DEFAULT_CONDITIONING_FRAC

    job_id = uuid.uuid4().hex[:12]
    job_dir = Path(config.JOBS_DIR) / job_id
    job_dir.mkdir(parents=True)
    video.save(job_dir / "input.mp4")
    (job_dir / "status.json").write_text(json.dumps({"state": "queued", "conditioning_frac": conditioning_frac}))

    return redirect(url_for("job_status", job_id=job_id))


@app.route("/jobs/<job_id>")
def job_status(job_id):
    job_dir = Path(config.JOBS_DIR) / job_id
    status_path = job_dir / "status.json"
    if not status_path.exists():
        return "job not found", 404
    status = json.loads(status_path.read_text())
    return render_template("job.html", job_id=job_id, status=status)


@app.route("/jobs/<job_id>/comparison.jpg")
def job_image(job_id):
    render_path = Path(config.JOBS_DIR) / job_id / "comparison.jpg"
    if not render_path.exists():
        return "not ready", 404
    return send_file(render_path, mimetype="image/jpeg")


@app.route("/jobs/<job_id>/comparison.mp4")
def job_video(job_id):
    video_path = Path(config.JOBS_DIR) / job_id / "comparison.mp4"
    if not video_path.exists():
        return "not ready", 404
    return send_file(video_path, mimetype="video/mp4", conditional=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
