import json
import sys
import time
from pathlib import Path

import config

sys.path.insert(0, str(Path(config.REPO_ROOT) / "src"))
from foretrack.demo_pipeline import run_demo_job  # noqa: E402

POLL_INTERVAL_S = 2.0


def _read_status(job_dir: Path) -> dict:
    return json.loads((job_dir / "status.json").read_text())


def _write_status(job_dir: Path, status: dict) -> None:
    (job_dir / "status.json").write_text(json.dumps(status))


def process_job(job_dir: Path) -> None:
    status = _read_status(job_dir)
    status["state"] = "running"
    _write_status(job_dir, status)

    try:
        result = run_demo_job(
            str(job_dir / "input.mp4"),
            status["conditioning_frac"],
            str(job_dir),
            hand_model_path=config.HAND_MODEL_PATH,
            sam2_checkpoint=config.SAM2_CHECKPOINT,
            sam2_model_cfg=config.SAM2_MODEL_CFG,
            tapip3d_python=config.TAPIP3D_PYTHON,
            tapip3d_repo=config.TAPIP3D_REPO,
            tapip3d_checkpoint=config.TAPIP3D_CHECKPOINT,
            forecast_python=config.FORECAST_PYTHON,
            forecast_script=config.FORECAST_SCRIPT,
            forecast_config=config.FORECAST_CONFIG,
            forecast_checkpoint=config.FORECAST_CHECKPOINT,
            k_samples=config.K_SAMPLES,
        )
    except Exception as e:
        result = {"status": "error", "reason": f"{type(e).__name__}: {e}"}

    status["state"] = result.pop("status")
    status.update(result)
    _write_status(job_dir, status)


def main():
    jobs_dir = Path(config.JOBS_DIR)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    print(f"worker watching {jobs_dir}", flush=True)
    while True:
        for job_dir in sorted(jobs_dir.iterdir()):
            status_path = job_dir / "status.json"
            if not status_path.exists():
                continue
            status = _read_status(job_dir)
            if status.get("state") == "queued":
                print(f"processing job {job_dir.name}", flush=True)
                process_job(job_dir)
                print(f"job {job_dir.name} -> {_read_status(job_dir)['state']}", flush=True)
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
