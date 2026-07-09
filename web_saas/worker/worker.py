import os
import sys
import time
import socket
from pathlib import Path

import requests
from dotenv import load_dotenv

from renderers import render_lipsync_job, render_voice_job


load_dotenv(Path(__file__).resolve().parents[1] / ".env")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8080").rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "change-this-long-random-token")
POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "5"))
WORK_DIR = Path(os.environ.get("WORK_DIR", "work")).resolve()
WORKER_ID = os.environ.get("WORKER_ID", socket.gethostname())
RENDER_BACKEND = os.environ.get("RENDER_BACKEND", "mock")


def headers() -> dict[str, str]:
    return {"X-Worker-Token": WORKER_TOKEN}


def post_log(job_id: str, message: str, progress: int = 10) -> None:
    try:
        requests.post(
            f"{BACKEND_URL}/api/worker/jobs/{job_id}/log",
            headers=headers(),
            json={"message": message, "log": message, "progress": progress},
            timeout=20,
        )
    except requests.RequestException:
        pass


def heartbeat(status: str = "online", current_job_id: str | None = None, message: str = "") -> None:
    try:
        requests.post(
            f"{BACKEND_URL}/api/worker/heartbeat",
            headers=headers(),
            json={
                "worker_id": WORKER_ID,
                "backend": RENDER_BACKEND,
                "status": status,
                "current_job_id": current_job_id,
                "message": message,
            },
            timeout=15,
        )
    except requests.RequestException:
        pass


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=headers(), stream=True, timeout=120) as res:
        res.raise_for_status()
        with destination.open("wb") as f:
            for chunk in res.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def complete(job_id: str, result_path: Path) -> None:
    content_type = "audio/wav" if result_path.suffix.lower() in {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"} else "video/mp4"
    with result_path.open("rb") as f:
        res = requests.post(
            f"{BACKEND_URL}/api/worker/jobs/{job_id}/complete",
            headers=headers(),
            files={"result": (result_path.name, f, content_type)},
            timeout=300,
        )
    res.raise_for_status()


def fail(job_id: str, message: str) -> None:
    requests.post(
        f"{BACKEND_URL}/api/worker/jobs/{job_id}/fail",
        headers=headers(),
        json={"message": message},
        timeout=30,
    )


def run_once() -> bool:
    res = requests.get(f"{BACKEND_URL}/api/worker/next", headers=headers(), timeout=30)
    res.raise_for_status()
    job = res.json().get("job")
    if not job:
        return False

    job_id = job["id"]
    job_dir = WORK_DIR / job_id
    is_voice = job.get("kind") == "voice"
    is_voice_lipsync = job.get("kind") == "voice_lipsync"
    video_path = job_dir / f"input_video{job.get('video_ext') or '.mp4'}"
    audio_path = job_dir / f"input_audio{job.get('audio_ext') or '.wav'}"
    voice_sample_path = job_dir / f"voice_sample{job.get('voice_sample_ext') or '.wav'}"
    generated_audio_path = job_dir / "generated_voice.wav"
    result_path = job_dir / ("voice_result.wav" if is_voice else "result.mp4")
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        heartbeat("processing", job_id, "Downloading inputs")
        post_log(job_id, "Downloading inputs", 8)
        if is_voice:
            if job.get("voice_sample_url"):
                download(job["voice_sample_url"], voice_sample_path)
            else:
                voice_sample_path = None
        elif is_voice_lipsync:
            download(job["video_url"], video_path)
            if job.get("voice_sample_url"):
                download(job["voice_sample_url"], voice_sample_path)
            else:
                voice_sample_path = None
        else:
            download(job["video_url"], video_path)
            download(job["audio_url"], audio_path)

        heartbeat("processing", job_id, "Rendering")
        post_log(job_id, "Rendering", 20)
        if is_voice:
            render_voice_job(
                voice_sample_path,
                job.get("voice_text") or "",
                result_path,
                job.get("settings", {}),
                lambda msg, pct: post_log(job_id, msg, pct),
            )
        elif is_voice_lipsync:
            post_log(job_id, "Generating voice audio", 18)
            render_voice_job(
                voice_sample_path,
                job.get("voice_text") or "",
                generated_audio_path,
                job.get("settings", {}),
                lambda msg, pct: post_log(job_id, msg, min(45, 18 + int(pct * 0.27))),
            )
            post_log(job_id, "Voice ready; starting lipsync render", 46)
            render_lipsync_job(
                video_path,
                generated_audio_path,
                result_path,
                job.get("settings", {}),
                lambda msg, pct: post_log(job_id, msg, min(94, 46 + int(pct * 0.48))),
            )
        else:
            render_lipsync_job(
                video_path,
                audio_path,
                result_path,
                job.get("settings", {}),
                lambda msg, pct: post_log(job_id, msg, pct),
            )

        heartbeat("processing", job_id, "Uploading result")
        post_log(job_id, "Uploading result", 95)
        complete(job_id, result_path)
        heartbeat("online", None, "Last job completed")
        print(f"Completed job {job_id}")
        return True
    except Exception as exc:
        message = f"Worker failed: {exc}"
        print(message, file=sys.stderr)
        heartbeat("error", job_id, message)
        try:
            fail(job_id, message)
        except Exception:
            pass
        return True


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Worker {WORKER_ID} connected to {BACKEND_URL} ({RENDER_BACKEND})")
    while True:
        try:
            heartbeat("online", None, "Polling")
            had_job = run_once()
        except Exception as exc:
            print(f"Poll error: {exc}", file=sys.stderr)
            had_job = False
        if not had_job:
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
