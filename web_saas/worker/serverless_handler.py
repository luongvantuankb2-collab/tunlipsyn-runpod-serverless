import base64
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import requests
import runpod

from renderers import render_lipsync_job, render_voice_job


WORK_DIR = Path(os.environ.get("WORK_DIR", "/workspace/work")).resolve()
RENDER_BACKEND = os.environ.get("RENDER_BACKEND", "runpod-serverless")


def _headers(worker_token: str) -> dict[str, str]:
    return {"X-Worker-Token": worker_token}


def _post_log(backend_url: str, worker_token: str, job_id: str, message: str, progress: int = 10) -> None:
    try:
        requests.post(
            f"{backend_url}/api/worker/jobs/{job_id}/log",
            headers=_headers(worker_token),
            json={"message": message, "log": message, "progress": progress},
            timeout=30,
        )
    except requests.RequestException:
        pass


def _heartbeat(backend_url: str, worker_token: str, status: str, job_id: str | None = None, message: str = "") -> None:
    try:
        requests.post(
            f"{backend_url}/api/worker/heartbeat",
            headers=_headers(worker_token),
            json={
                "worker_id": os.environ.get("RUNPOD_POD_ID") or socket.gethostname(),
                "backend": RENDER_BACKEND,
                "status": status,
                "current_job_id": job_id,
                "message": message,
            },
            timeout=20,
        )
    except requests.RequestException:
        pass


def _download(url: str, destination: Path, worker_token: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=_headers(worker_token), stream=True, timeout=300) as response:
        response.raise_for_status()
        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _download_plain(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _complete(backend_url: str, worker_token: str, job_id: str, result_path: Path) -> None:
    content_type = "audio/wav" if result_path.suffix.lower() in {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"} else "video/mp4"
    with result_path.open("rb") as f:
        response = requests.post(
            f"{backend_url}/api/worker/jobs/{job_id}/complete",
            headers=_headers(worker_token),
            files={"result": (result_path.name, f, content_type)},
            timeout=600,
        )
    response.raise_for_status()


def _fail(backend_url: str, worker_token: str, job_id: str, message: str) -> None:
    try:
        requests.post(
            f"{backend_url}/api/worker/jobs/{job_id}/fail",
            headers=_headers(worker_token),
            json={"message": message[:1000]},
            timeout=30,
        )
    except requests.RequestException:
        pass


def _claim_job(backend_url: str, worker_token: str, job_id: str) -> dict[str, Any] | None:
    response = requests.post(
        f"{backend_url}/api/worker/jobs/{job_id}/claim",
        headers=_headers(worker_token),
        timeout=60,
    )
    response.raise_for_status()
    return response.json().get("job")


def _collect_logs() -> tuple[list[str], Any]:
    logs: list[str] = []

    def progress(message: str, percent: int) -> None:
        line = f"[{percent}] {message}"
        logs.append(line)
        print(line, flush=True)

    return logs, progress


def _demo_asset(name: str) -> Path:
    latentsync_dir = Path(os.environ.get("LATENTSYNC_DIR", "/workspace/tunlipsyn/engines/LatentSync"))
    candidates = [
        latentsync_dir / "assets" / name,
        latentsync_dir / "assets" / name.replace("demo1_", "demo_"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing demo asset {name}; checked: {', '.join(str(c) for c in candidates)}")


def _result_payload(result_path: Path, logs: list[str], started: float) -> dict[str, Any]:
    size = result_path.stat().st_size if result_path.exists() else 0
    payload: dict[str, Any] = {
        "ok": True,
        "mode": "direct_test",
        "result_size": size,
        "seconds": round(time.time() - started, 3),
        "logs": logs[-200:],
    }
    inline_limit = int(os.environ.get("RUNPOD_INLINE_RESULT_MAX_BYTES", str(18 * 1024 * 1024)))
    if size and size <= inline_limit:
        payload["result_base64"] = base64.b64encode(result_path.read_bytes()).decode("ascii")
        payload["result_mime"] = "video/mp4"
        payload["result_name"] = result_path.name
    else:
        payload["message"] = "Render completed, but result is too large to return inline. Use a backend/storage callback for production."
    return payload


def _direct_lipsync_test(payload: dict[str, Any], started: float) -> dict[str, Any]:
    job_id = str(payload.get("job_id") or f"direct_{int(started)}")
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    logs, progress = _collect_logs()

    video_url = str(payload.get("video_url") or "").strip()
    audio_url = str(payload.get("audio_url") or "").strip()
    video_path = job_dir / "input_video.mp4"
    audio_path = job_dir / "input_audio.wav"
    result_path = job_dir / "result.mp4"

    if video_url:
        progress("Downloading video_url", 5)
        _download_plain(video_url, video_path)
    else:
        progress("Using bundled LatentSync demo video", 5)
        video_path = _demo_asset("demo1_video.mp4")

    if audio_url:
        progress("Downloading audio_url", 8)
        _download_plain(audio_url, audio_path)
    else:
        progress("Using bundled LatentSync demo audio", 8)
        audio_path = _demo_asset("demo1_audio.wav")

    settings = dict(payload.get("settings") or {})
    settings.setdefault("preset", payload.get("preset") or "v15")
    settings.setdefault("lipsync_model", payload.get("lipsync_model") or "v15")
    settings.setdefault("latentsync_steps", str(payload.get("steps") or payload.get("latentsync_steps") or "40"))
    settings.setdefault("latentsync_guidance", str(payload.get("guidance") or payload.get("latentsync_guidance") or "1.8"))
    settings.setdefault("crop_scale", str(payload.get("crop_scale") or "0.75"))
    settings.setdefault("keep_original_frame", "on")
    settings.setdefault("latentsync_deepcache", "off")
    settings.setdefault("enhance_video", "off")

    render_lipsync_job(video_path, audio_path, result_path, settings, progress)
    return _result_payload(result_path, logs, started)


def _render_claimed_job(backend_url: str, worker_token: str, job: dict[str, Any]) -> Path:
    job_id = job["id"]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    is_voice = job.get("kind") == "voice"
    is_voice_lipsync = job.get("kind") == "voice_lipsync"
    video_path = job_dir / f"input_video{job.get('video_ext') or '.mp4'}"
    audio_path = job_dir / f"input_audio{job.get('audio_ext') or '.wav'}"
    voice_sample_path: Path | None = job_dir / f"voice_sample{job.get('voice_sample_ext') or '.wav'}"
    generated_audio_path = job_dir / "generated_voice.wav"
    result_path = job_dir / ("voice_result.wav" if is_voice else "result.mp4")

    _heartbeat(backend_url, worker_token, "processing", job_id, "Downloading inputs")
    _post_log(backend_url, worker_token, job_id, "Downloading inputs", 8)
    if is_voice:
        if job.get("voice_sample_url"):
            _download(job["voice_sample_url"], voice_sample_path, worker_token)
        else:
            voice_sample_path = None
    elif is_voice_lipsync:
        _download(job["video_url"], video_path, worker_token)
        if job.get("voice_sample_url"):
            _download(job["voice_sample_url"], voice_sample_path, worker_token)
        else:
            voice_sample_path = None
    else:
        _download(job["video_url"], video_path, worker_token)
        _download(job["audio_url"], audio_path, worker_token)

    _heartbeat(backend_url, worker_token, "processing", job_id, "Rendering")
    _post_log(backend_url, worker_token, job_id, "Rendering", 20)
    if is_voice:
        render_voice_job(
            voice_sample_path,
            job.get("voice_text") or "",
            result_path,
            job.get("settings", {}),
            lambda msg, pct: _post_log(backend_url, worker_token, job_id, msg, pct),
        )
    elif is_voice_lipsync:
        _post_log(backend_url, worker_token, job_id, "Generating voice audio", 18)
        render_voice_job(
            voice_sample_path,
            job.get("voice_text") or "",
            generated_audio_path,
            job.get("settings", {}),
            lambda msg, pct: _post_log(backend_url, worker_token, job_id, msg, min(45, 18 + int(pct * 0.27))),
        )
        _post_log(backend_url, worker_token, job_id, "Voice ready; starting lipsync render", 46)
        render_lipsync_job(
            video_path,
            generated_audio_path,
            result_path,
            job.get("settings", {}),
            lambda msg, pct: _post_log(backend_url, worker_token, job_id, msg, min(94, 46 + int(pct * 0.48))),
        )
    else:
        render_lipsync_job(
            video_path,
            audio_path,
            result_path,
            job.get("settings", {}),
            lambda msg, pct: _post_log(backend_url, worker_token, job_id, msg, pct),
        )
    return result_path


def handler(event: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    payload = event.get("input") or {}
    if payload.get("self_test") or payload.get("mode") in {"self_test", "direct_lipsync", "direct_test"}:
        try:
            return _direct_lipsync_test(payload, started)
        except Exception as exc:
            message = f"RunPod direct test failed: {exc}"
            print(message, file=sys.stderr)
            return {"ok": False, "mode": "direct_test", "error": message, "seconds": round(time.time() - started, 3)}

    job_id = str(payload.get("job_id") or "").strip()
    backend_url = str(payload.get("backend_url") or os.environ.get("BACKEND_URL") or "").rstrip("/")
    worker_token = str(payload.get("worker_token") or os.environ.get("WORKER_TOKEN") or "")
    if not job_id:
        return {"ok": False, "error": "Missing input.job_id"}
    if not backend_url:
        return {"ok": False, "job_id": job_id, "error": "Missing input.backend_url or BACKEND_URL"}
    if not worker_token:
        return {"ok": False, "job_id": job_id, "error": "Missing input.worker_token or WORKER_TOKEN"}

    try:
        _heartbeat(backend_url, worker_token, "online", job_id, "RunPod handler started")
        job = _claim_job(backend_url, worker_token, job_id)
        if not job:
            return {"ok": True, "job_id": job_id, "skipped": True, "message": "Job was not queued/processing"}
        result_path = _render_claimed_job(backend_url, worker_token, job)
        _heartbeat(backend_url, worker_token, "processing", job_id, "Uploading result")
        _post_log(backend_url, worker_token, job_id, "Uploading result", 95)
        _complete(backend_url, worker_token, job_id, result_path)
        _heartbeat(backend_url, worker_token, "online", None, "Last job completed")
        return {
            "ok": True,
            "job_id": job_id,
            "kind": job.get("kind"),
            "result_size": result_path.stat().st_size if result_path.exists() else 0,
            "seconds": round(time.time() - started, 3),
        }
    except Exception as exc:
        message = f"RunPod worker failed: {exc}"
        print(message, file=sys.stderr)
        _heartbeat(backend_url, worker_token, "error", job_id, message)
        _fail(backend_url, worker_token, job_id, message)
        return {"ok": False, "job_id": job_id, "error": message, "seconds": round(time.time() - started, 3)}


runpod.serverless.start({"handler": handler})