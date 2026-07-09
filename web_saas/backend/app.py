import json
import os
import shutil
import sqlite3
import subprocess
import time
import unicodedata
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
load_dotenv(PROJECT_ROOT / ".env")
DATABASE_PATH = Path(os.environ.get("DATABASE_PATH", PROJECT_ROOT / "storage" / "app.db"))
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", PROJECT_ROOT / "storage" / "jobs"))
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
WORKER_BASE_URL = os.environ.get("WORKER_BASE_URL", PUBLIC_BASE_URL).rstrip("/")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "change-this-long-random-token")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "1024"))
FORCE_STABLE_LIPSYNC_PROFILE = os.environ.get("LATENTSYNC_FORCE_STABLE_PROFILE", "1").lower() not in {"0", "false", "no"}
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "").strip()
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
RUNPOD_API_BASE = os.environ.get("RUNPOD_API_BASE", "https://api.runpod.ai/v2").rstrip("/")
RUNPOD_DISPATCH_ENABLED = os.environ.get("RUNPOD_DISPATCH_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
RUNPOD_CALLBACK_BASE_URL = os.environ.get("RUNPOD_CALLBACK_BASE_URL", WORKER_BASE_URL).rstrip("/")


def normalize_base_path(value: str) -> str:
    value = (value or "").strip()
    if not value or value == "/":
        return ""
    return "/" + value.strip("/")


WEB_BASE_PATH = normalize_base_path(os.environ.get("WEB_BASE_PATH", ""))

ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}
ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}
JOB_KIND_LIPSYNC = "lipsync"
JOB_KIND_VOICE = "voice"
JOB_KIND_VOICE_LIPSYNC = "voice_lipsync"

VOICE_NAME_MAP = {
    "ngoc_lan": "Ngọc Lan",
    "gia_bao": "Gia Bảo",
    "thai_son": "Thái Sơn",
    "duc_tri": "Đức Trí",
    "my_duyen": "Mỹ Duyên",
    "truc_ly": "Trúc Ly",
    "xuan_vinh": "Xuân Vĩnh",
    "trong_huu": "Trọng Hữu",
    "binh_an": "Bình An",
    "ngoc_linh": "Ngọc Linh",
}

VOICE_NAME_MAP.update({
    "ngoc_lan": "Ngọc Lan",
    "gia_bao": "Gia Bảo",
    "thai_son": "Thái Sơn",
    "duc_tri": "Đức Trí",
    "my_duyen": "Mỹ Duyên",
    "truc_ly": "Trúc Ly",
    "xuan_vinh": "Xuân Vĩnh",
    "trong_huu": "Trọng Hữu",
    "binh_an": "Bình An",
    "ngoc_linh": "Ngọc Linh",
})

ERROR_CATALOG: dict[str, dict[str, str]] = {
    "UNSUPPORTED_FILE": {
        "message": "Định dạng file chưa được hỗ trợ.",
        "action": "Dùng MP4/MOV/MKV/WEBM cho video hoặc WAV/MP3/M4A/AAC/OGG/FLAC cho audio.",
    },
    "UPLOAD_TOO_LARGE": {
        "message": "File tải lên vượt quá giới hạn dung lượng.",
        "action": f"Giảm dung lượng file hoặc tăng MAX_UPLOAD_MB hiện tại ({MAX_UPLOAD_MB} MB).",
    },
    "TEXT_REQUIRED": {
        "message": "Bạn chưa nhập nội dung cần đọc.",
        "action": "Nhập lời thoại trước khi tạo giọng nói.",
    },
    "VOICE_SAMPLE_REQUIRED": {
        "message": "Chưa có mẫu giọng để clone.",
        "action": "Chọn file audio mẫu rõ tiếng, ít nhiễu, dài khoảng 10-30 giây.",
    },
    "VOICE_JOB_NOT_FOUND": {
        "message": "Không tìm thấy job tạo giọng.",
        "action": "Làm mới danh sách job rồi thử lại.",
    },
    "VOICE_NOT_READY": {
        "message": "Audio tạo giọng chưa sẵn sàng.",
        "action": "Chờ job audio hoàn tất rồi hãy tạo video lipsync.",
    },
    "VOICE_RESULT_MISSING": {
        "message": "File audio kết quả không còn tồn tại trên máy chủ.",
        "action": "Chạy lại job tạo giọng hoặc kiểm tra thư mục storage.",
    },
    "JOB_NOT_FOUND": {
        "message": "Không tìm thấy job.",
        "action": "Job có thể đã bị xóa. Làm mới danh sách rồi thử lại.",
    },
    "RESULT_NOT_READY": {
        "message": "Kết quả chưa sẵn sàng.",
        "action": "Chờ worker xử lý xong rồi tải lại.",
    },
    "RESULT_MISSING": {
        "message": "File kết quả không còn tồn tại trên máy chủ.",
        "action": "Chạy lại job hoặc kiểm tra thư mục storage.",
    },
    "INPUT_MISSING": {
        "message": "File đầu vào không còn tồn tại trên máy chủ.",
        "action": "Tạo lại job với file video/audio mới.",
    },
    "LATENTSYNC_NOT_READY": {
        "message": "Engine lipsync chưa sẵn sàng.",
        "action": "Kiểm tra LATENTSYNC_DIR, checkpoint và cấu hình LatentSync trên worker.",
    },
    "FFMPEG_NOT_READY": {
        "message": "FFmpeg chưa sẵn sàng hoặc xử lý file thất bại.",
        "action": "Kiểm tra ffmpeg.exe trong portable root/PATH và thử lại với file đầu vào khác.",
    },
    "CUDA_NOT_READY": {
        "message": "GPU CUDA chưa sẵn sàng.",
        "action": "Kiểm tra driver NVIDIA, PyTorch CUDA và worker GPU.",
    },
    "GPU_OUT_OF_MEMORY": {
        "message": "GPU không đủ bộ nhớ cho job này.",
        "action": "Thử video ngắn hơn, giảm độ phân giải hoặc chuyển sang GPU lớn hơn.",
    },
    "VOICE_ENGINE_NOT_READY": {
        "message": "Engine tạo giọng chưa sẵn sàng.",
        "action": "Kiểm tra VIENEU_SCRIPT, VIENEU_PYTHON và model tạo giọng trên worker.",
    },
    "ENGINE_OUTPUT_MISSING": {
        "message": "Engine không tạo được file kết quả.",
        "action": "Mở log kỹ thuật để xem engine dừng ở bước nào.",
    },
    "ENGINE_COMMAND_FAILED": {
        "message": "Engine xử lý thất bại.",
        "action": "Kiểm tra log kỹ thuật, file đầu vào và cấu hình worker.",
    },
    "JOB_TIMEOUT": {
        "message": "Job xử lý lâu hơn dự kiến.",
        "action": "Kiểm tra worker còn online không, hoặc chờ job hoàn tất trong hàng đợi.",
    },
    "WORKER_TOKEN_INVALID": {
        "message": "Worker không được phép kết nối.",
        "action": "Kiểm tra WORKER_TOKEN giữa backend và worker.",
    },
    "VALIDATION_ERROR": {
        "message": "Dữ liệu gửi lên chưa hợp lệ.",
        "action": "Kiểm tra các trường bắt buộc và file đính kèm.",
    },
    "SERVER_ERROR": {
        "message": "Máy chủ gặp lỗi khi xử lý yêu cầu.",
        "action": "Thử lại sau. Nếu lỗi lặp lại, gửi log cho đội kỹ thuật.",
    },
}


app = FastAPI(title="Tun Lipsync SaaS")
templates = Jinja2Templates(directory=str(ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


def api_error(code: str, status_code: int = 400, detail: str = "") -> HTTPException:
    item = ERROR_CATALOG.get(code, ERROR_CATALOG["SERVER_ERROR"])
    payload = {"code": code, "message": item["message"], "action": item["action"]}
    if detail:
        payload["detail"] = detail
    return HTTPException(status_code=status_code, detail=payload)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "message" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": "SERVER_ERROR", **ERROR_CATALOG["SERVER_ERROR"], "detail": str(exc.detail)}},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "VALIDATION_ERROR", **ERROR_CATALOG["VALIDATION_ERROR"], "detail": str(exc)}},
    )


def now_ts() -> int:
    return int(time.time())


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def parse_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def render_settings_from_form(
    preset: str,
    lipsync_model: str,
    latentsync_steps: Any,
    latentsync_guidance: Any,
    crop_scale: Any,
    keep_original_frame: Any,
    mouth_mask_overlay: Any,
    latentsync_deepcache: Any,
    enhance_video: Any,
    enhance_backend: str,
    mouth_mask_x: Any = 0.32,
    mouth_mask_y: Any = 0.26,
    mouth_mask_w: Any = 0.36,
    mouth_mask_h: Any = 0.22,
) -> dict[str, Any]:
    model = (lipsync_model or "auto").strip().lower()
    if model not in {"auto", "v15", "v16", "efficient", "standard", "high512"}:
        model = "auto"
    quality = (preset or "auto").strip().lower()
    if quality not in {"auto", "fast", "pro", "ultra"}:
        quality = "auto"
    if FORCE_STABLE_LIPSYNC_PROFILE:
        model = "v15"
        latentsync_steps = 40
        latentsync_guidance = 1.8
        crop_scale = 0.75
        mouth_mask_overlay = "off"
        latentsync_deepcache = "off"
    return {
        "preset": quality,
        "lipsync_model": model,
        "keep_original_frame": parse_bool(keep_original_frame, True),
        "mouth_mask_overlay": parse_bool(mouth_mask_overlay, False),
        "mouth_mask_x": parse_float(mouth_mask_x, 0.32, 0.0, 1.0),
        "mouth_mask_y": parse_float(mouth_mask_y, 0.26, 0.0, 1.0),
        "mouth_mask_w": parse_float(mouth_mask_w, 0.36, 0.05, 1.0),
        "mouth_mask_h": parse_float(mouth_mask_h, 0.22, 0.05, 1.0),
        "crop_scale": parse_float(crop_scale, 0.75, 0.55, 1.0),
        "latentsync_steps": parse_int(latentsync_steps, 40, 5, 80),
        "latentsync_guidance": parse_float(latentsync_guidance, 1.8, 0.5, 5.0),
        "latentsync_deepcache": parse_bool(latentsync_deepcache, False),
        "enhance_video": parse_bool(enhance_video, False),
        "enhance_backend": enhance_backend,
    }


def dispatch_serverless_job(job_id: str) -> None:
    if not RUNPOD_DISPATCH_ENABLED:
        return
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        with db() as conn:
            append_log(conn, job_id, "RunPod dispatch skipped: RUNPOD_API_KEY/RUNPOD_ENDPOINT_ID is not configured")
        return

    url = f"{RUNPOD_API_BASE}/{RUNPOD_ENDPOINT_ID}/run"
    payload = {
        "input": {
            "job_id": job_id,
            "backend_url": RUNPOD_CALLBACK_BASE_URL,
            "worker_token": WORKER_TOKEN,
        }
    }
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        runpod_job_id = data.get("id") or data.get("job_id") or ""
        with db() as conn:
            append_log(conn, job_id, f"RunPod serverless job dispatched: {runpod_job_id or 'accepted'}")
            if runpod_job_id:
                conn.execute(
                    "UPDATE jobs SET message = ?, updated_at = ? WHERE id = ?",
                    (f"RunPod queued: {runpod_job_id}", now_ts(), job_id),
                )
    except Exception as exc:
        with db() as conn:
            append_log(conn, job_id, f"RunPod dispatch failed: {exc}")
            conn.execute(
                "UPDATE jobs SET status = 'failed', message = ?, finished_at = ?, updated_at = ? WHERE id = ?",
                (f"RunPod dispatch failed: {exc}", now_ts(), now_ts(), job_id),
            )


def safe_ext(filename: str, allowed: set[str]) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in allowed:
        raise api_error("UNSUPPORTED_FILE", 400, f"Extension: {ext or 'none'}")
    return ext


def require_worker_token(x_worker_token: str | None) -> None:
    if not WORKER_TOKEN or WORKER_TOKEN == "change-this-long-random-token":
        # Local default is allowed for first-run testing.
        expected = WORKER_TOKEN
    else:
        expected = WORKER_TOKEN
    if x_worker_token != expected:
        raise api_error("WORKER_TOKEN_INVALID", 401)


@contextmanager
def db() -> Any:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL DEFAULT 'lipsync',
                status TEXT NOT NULL,
                title TEXT NOT NULL,
                input_video TEXT NOT NULL DEFAULT '',
                input_audio TEXT NOT NULL DEFAULT '',
                voice_text TEXT NOT NULL DEFAULT '',
                voice_sample TEXT NOT NULL DEFAULT '',
                result_video TEXT,
                result_audio TEXT,
                settings_json TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                logs TEXT NOT NULL DEFAULT '',
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                started_at INTEGER,
                finished_at INTEGER
            )
            """
        )
        for statement in (
            "ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'lipsync'",
            "ALTER TABLE jobs ADD COLUMN voice_text TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE jobs ADD COLUMN voice_sample TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE jobs ADD COLUMN result_audio TEXT",
            "ALTER TABLE jobs ADD COLUMN started_at INTEGER",
            "ALTER TABLE jobs ADD COLUMN finished_at INTEGER",
        ):
            try:
                conn.execute(statement)
            except sqlite3.OperationalError:
                pass
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workers (
                id TEXT PRIMARY KEY,
                backend TEXT NOT NULL,
                status TEXT NOT NULL,
                current_job_id TEXT,
                message TEXT NOT NULL DEFAULT '',
                last_seen_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )


@app.on_event("startup")
def on_startup() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    init_db()


def row_to_job(row: sqlite3.Row, include_private: bool = False) -> dict[str, Any]:
    data = dict(row)
    data["settings"] = json.loads(data.pop("settings_json") or "{}")
    data["kind"] = data.get("kind") or JOB_KIND_LIPSYNC
    data["has_result"] = bool(data.get("result_video") or data.get("result_audio"))
    current_time = now_ts()
    created_at = data.get("created_at") or current_time
    started_at = data.get("started_at")
    finished_at = data.get("finished_at")
    data["queue_seconds"] = max(0, (started_at or current_time) - created_at)
    if started_at:
        data["runtime_seconds"] = max(0, (finished_at or current_time) - started_at)
    else:
        data["runtime_seconds"] = 0
    data["total_seconds"] = max(0, (finished_at or current_time) - created_at)
    data["input_video_name"] = Path(data["input_video"]).name if data.get("input_video") else ""
    data["input_audio_name"] = Path(data["input_audio"]).name if data.get("input_audio") else ""
    data["voice_sample_name"] = Path(data["voice_sample"]).name if data.get("voice_sample") else ""
    data["video_ext"] = Path(data["input_video"]).suffix if data.get("input_video") else ".mp4"
    data["audio_ext"] = Path(data["input_audio"]).suffix if data.get("input_audio") else ".wav"
    data["voice_sample_ext"] = Path(data["voice_sample"]).suffix if data.get("voice_sample") else ".wav"
    if not include_private:
        data.pop("input_video", None)
        data.pop("input_audio", None)
        data.pop("voice_sample", None)
        data.pop("result_video", None)
        data.pop("result_audio", None)
    return data


def append_log(conn: sqlite3.Connection, job_id: str, line: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "UPDATE jobs SET logs = logs || ?, updated_at = ? WHERE id = ?",
        (f"[{stamp}] {line}\n", now_ts(), job_id),
    )


def normalize_voice_name(value: str) -> str:
    value = (value or "").strip()
    if not value or "?" in value or "�" in value:
        return VOICE_NAME_MAP["ngoc_lan"]
    return unicodedata.normalize("NFC", VOICE_NAME_MAP.get(value, value))


def audio_media_type(path: Path) -> str:
    return {
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".wav": "audio/wav",
    }.get(path.suffix.lower(), "application/octet-stream")


def ffmpeg_path() -> str | None:
    local_ffmpeg = PROJECT_ROOT.parent / "ffmpeg.exe"
    if local_ffmpeg.exists():
        return str(local_ffmpeg)
    return shutil.which("ffmpeg")


def voice_preview_path(audio_path: Path) -> Path:
    preview_path = audio_path.with_name(f"{audio_path.stem}_preview.mp3")
    if preview_path.exists() and preview_path.stat().st_mtime >= audio_path.stat().st_mtime:
        return preview_path

    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        return audio_path

    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-fflags",
                "+genpts",
                "-i",
                str(audio_path),
                "-vn",
                "-map_metadata",
                "-1",
                "-af",
                "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(preview_path),
            ],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return audio_path
    return preview_path if preview_path.exists() else audio_path


async def save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with destination.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise api_error("UPLOAD_TOO_LARGE", 413)
            f.write(chunk)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request, "base_path": WEB_BASE_PATH})


@app.get("/api/health")
def health() -> JSONResponse:
    with db() as conn:
        queued = conn.execute("SELECT COUNT(*) AS c FROM jobs WHERE status = 'queued'").fetchone()["c"]
        processing = conn.execute("SELECT COUNT(*) AS c FROM jobs WHERE status = 'processing'").fetchone()["c"]
        done = conn.execute("SELECT COUNT(*) AS c FROM jobs WHERE status = 'done'").fetchone()["c"]
        failed = conn.execute("SELECT COUNT(*) AS c FROM jobs WHERE status = 'failed'").fetchone()["c"]
        voice = conn.execute("SELECT COUNT(*) AS c FROM jobs WHERE kind = ?", (JOB_KIND_VOICE,)).fetchone()["c"]
        lipsync = conn.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE kind IN (?, ?)",
            (JOB_KIND_LIPSYNC, JOB_KIND_VOICE_LIPSYNC),
        ).fetchone()["c"]
        workers = conn.execute("SELECT COUNT(*) AS c FROM workers WHERE last_seen_at >= ?", (now_ts() - 60,)).fetchone()["c"]
    return JSONResponse(
        {
            "ok": True,
            "time": now_ts(),
            "workers_online": workers,
            "jobs": {
                "queued": queued,
                "processing": processing,
                "done": done,
                "failed": failed,
                "voice": voice,
                "lipsync": lipsync,
            },
        }
    )


@app.get("/api/error-catalog")
def error_catalog() -> JSONResponse:
    return JSONResponse({"errors": ERROR_CATALOG})


@app.post("/api/jobs")
async def create_job(
    title: str = Form("Untitled job"),
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    preset: str = Form("ultra"),
    lipsync_model: str = Form("auto"),
    latentsync_steps: int = Form(40),
    latentsync_guidance: float = Form(1.8),
    crop_scale: float = Form(0.75),
    keep_original_frame: str = Form("on"),
    mouth_mask_overlay: str = Form("off"),
    mouth_mask_x: float = Form(0.32),
    mouth_mask_y: float = Form(0.26),
    mouth_mask_w: float = Form(0.36),
    mouth_mask_h: float = Form(0.22),
    latentsync_deepcache: str = Form("off"),
    enhance_video: str = Form("off"),
    enhance_backend: str = Form("realbasicvsr"),
) -> JSONResponse:
    video_ext = safe_ext(video.filename or "", ALLOWED_VIDEO_EXT)
    audio_ext = safe_ext(audio.filename or "", ALLOWED_AUDIO_EXT)
    job_id = uuid.uuid4().hex
    job_dir = STORAGE_DIR / job_id
    video_path = job_dir / f"input_video{video_ext}"
    audio_path = job_dir / f"input_audio{audio_ext}"
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    await save_upload(video, video_path, max_bytes)
    await save_upload(audio, audio_path, max_bytes)

    settings = render_settings_from_form(
        preset,
        lipsync_model,
        latentsync_steps,
        latentsync_guidance,
        crop_scale,
        keep_original_frame,
        mouth_mask_overlay,
        latentsync_deepcache,
        enhance_video,
        enhance_backend,
        mouth_mask_x,
        mouth_mask_y,
        mouth_mask_w,
        mouth_mask_h,
    )
    with db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id, kind, status, title, input_video, input_audio, settings_json,
                progress, message, created_at, updated_at
            ) VALUES (?, ?, 'queued', ?, ?, ?, ?, 0, 'Waiting for GPU worker', ?, ?)
            """,
            (
                job_id,
                JOB_KIND_LIPSYNC,
                title.strip() or "Untitled job",
                str(video_path),
                str(audio_path),
                json.dumps(settings),
                now_ts(),
                now_ts(),
            ),
        )
        append_log(conn, job_id, "Job created")
    dispatch_serverless_job(job_id)
    return JSONResponse({"job_id": job_id})


@app.post("/api/voice-jobs")
async def create_voice_job(
    title: str = Form("Voice job"),
    text: str = Form(...),
    sample: UploadFile | None = File(default=None),
    voice_mode: str = Form("clone"),
    voice_name: str = Form("Ngọc Lan"),
    voice_preset: str = Form("balanced"),
    language: str = Form("vi"),
) -> JSONResponse:
    if not text.strip():
        raise api_error("TEXT_REQUIRED", 400)
    voice_mode = voice_mode if voice_mode in {"preset", "clone"} else "clone"
    voice_name = normalize_voice_name(voice_name)
    if voice_mode == "clone" and (not sample or not sample.filename):
        raise api_error("VOICE_SAMPLE_REQUIRED", 400)

    job_id = uuid.uuid4().hex
    job_dir = STORAGE_DIR / job_id
    sample_path = ""
    if sample and sample.filename:
        sample_ext = safe_ext(sample.filename or "", ALLOWED_AUDIO_EXT)
        sample_file = job_dir / f"voice_sample{sample_ext}"
        await save_upload(sample, sample_file, MAX_UPLOAD_MB * 1024 * 1024)
        sample_path = str(sample_file)

    settings = {
        "voice_mode": voice_mode,
        "voice_name": voice_name,
        "voice_preset": voice_preset,
        "language": language,
        "provider": "voice_clone" if voice_mode == "clone" else "voice_preset",
        "output_format": "wav",
    }
    with db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id, kind, status, title, input_video, input_audio, voice_text, voice_sample, settings_json,
                progress, message, created_at, updated_at
            ) VALUES (?, ?, 'queued', ?, '', '', ?, ?, ?, 0, 'Waiting for voice worker', ?, ?)
            """,
            (
                job_id,
                JOB_KIND_VOICE,
                title.strip() or "Voice job",
                text.strip(),
                sample_path,
                json.dumps(settings),
                now_ts(),
                now_ts(),
            ),
        )
        append_log(conn, job_id, "Voice job created")
    dispatch_serverless_job(job_id)
    return JSONResponse({"job_id": job_id})


@app.post("/api/voice-jobs/{voice_job_id}/lipsync")
async def create_lipsync_from_voice_job(
    voice_job_id: str,
    video: UploadFile = File(...),
    title: str = Form("Video from voice"),
    preset: str = Form("ultra"),
    lipsync_model: str = Form("auto"),
    latentsync_steps: int = Form(40),
    latentsync_guidance: float = Form(1.8),
    crop_scale: float = Form(0.75),
    keep_original_frame: str = Form("on"),
    mouth_mask_overlay: str = Form("off"),
    mouth_mask_x: float = Form(0.32),
    mouth_mask_y: float = Form(0.26),
    mouth_mask_w: float = Form(0.36),
    mouth_mask_h: float = Form(0.22),
    latentsync_deepcache: str = Form("off"),
    enhance_video: str = Form("off"),
    enhance_backend: str = Form("realbasicvsr"),
) -> JSONResponse:
    with db() as conn:
        voice_row = conn.execute(
            "SELECT result_audio, title, status FROM jobs WHERE id = ? AND kind = ?",
            (voice_job_id, JOB_KIND_VOICE),
        ).fetchone()
    if not voice_row:
        raise api_error("VOICE_JOB_NOT_FOUND", 404)
    if voice_row["status"] != "done" or not voice_row["result_audio"]:
        raise api_error("VOICE_NOT_READY", 400)

    source_audio = Path(voice_row["result_audio"])
    if not source_audio.exists():
        raise api_error("VOICE_RESULT_MISSING", 404)

    video_ext = safe_ext(video.filename or "", ALLOWED_VIDEO_EXT)
    audio_ext = source_audio.suffix.lower()
    if audio_ext not in ALLOWED_AUDIO_EXT:
        raise api_error("UNSUPPORTED_FILE", 400, f"Voice output extension: {audio_ext}")

    job_id = uuid.uuid4().hex
    job_dir = STORAGE_DIR / job_id
    video_path = job_dir / f"input_video{video_ext}"
    audio_path = job_dir / f"input_audio{audio_ext}"
    await save_upload(video, video_path, MAX_UPLOAD_MB * 1024 * 1024)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_audio, audio_path)

    settings = render_settings_from_form(
        preset,
        lipsync_model,
        latentsync_steps,
        latentsync_guidance,
        crop_scale,
        keep_original_frame,
        mouth_mask_overlay,
        latentsync_deepcache,
        enhance_video,
        enhance_backend,
        mouth_mask_x,
        mouth_mask_y,
        mouth_mask_w,
        mouth_mask_h,
    )
    settings["source_voice_job_id"] = voice_job_id
    with db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id, kind, status, title, input_video, input_audio, settings_json,
                progress, message, created_at, updated_at
            ) VALUES (?, ?, 'queued', ?, ?, ?, ?, 0, 'Waiting for GPU worker', ?, ?)
            """,
            (
                job_id,
                JOB_KIND_LIPSYNC,
                title.strip() or f"Video from {voice_row['title']}",
                str(video_path),
                str(audio_path),
                json.dumps(settings),
                now_ts(),
                now_ts(),
            ),
        )
        append_log(conn, job_id, f"Job created from voice job {voice_job_id}")
    dispatch_serverless_job(job_id)
    return JSONResponse({"job_id": job_id})


@app.post("/api/voice-lipsync-jobs")
async def create_voice_lipsync_job(
    title: str = Form("Voice lipsync video"),
    video: UploadFile = File(...),
    text: str = Form(...),
    sample: UploadFile | None = File(default=None),
    voice_mode: str = Form("preset"),
    voice_name: str = Form("Ngá»c Lan"),
    voice_preset: str = Form("balanced"),
    language: str = Form("vi"),
    preset: str = Form("ultra"),
    lipsync_model: str = Form("auto"),
    latentsync_steps: int = Form(40),
    latentsync_guidance: float = Form(1.8),
    crop_scale: float = Form(0.75),
    keep_original_frame: str = Form("on"),
    mouth_mask_overlay: str = Form("off"),
    mouth_mask_x: float = Form(0.32),
    mouth_mask_y: float = Form(0.26),
    mouth_mask_w: float = Form(0.36),
    mouth_mask_h: float = Form(0.22),
    latentsync_deepcache: str = Form("off"),
    enhance_video: str = Form("off"),
    enhance_backend: str = Form("realbasicvsr"),
) -> JSONResponse:
    if not text.strip():
        raise api_error("TEXT_REQUIRED", 400)
    voice_mode = voice_mode if voice_mode in {"preset", "clone"} else "preset"
    voice_name = normalize_voice_name(voice_name)
    if voice_mode == "clone" and (not sample or not sample.filename):
        raise api_error("VOICE_SAMPLE_REQUIRED", 400)

    video_ext = safe_ext(video.filename or "", ALLOWED_VIDEO_EXT)
    job_id = uuid.uuid4().hex
    job_dir = STORAGE_DIR / job_id
    video_path = job_dir / f"input_video{video_ext}"
    await save_upload(video, video_path, MAX_UPLOAD_MB * 1024 * 1024)

    sample_path = ""
    if sample and sample.filename:
        sample_ext = safe_ext(sample.filename or "", ALLOWED_AUDIO_EXT)
        sample_file = job_dir / f"voice_sample{sample_ext}"
        await save_upload(sample, sample_file, MAX_UPLOAD_MB * 1024 * 1024)
        sample_path = str(sample_file)

    settings = render_settings_from_form(
        preset,
        lipsync_model,
        latentsync_steps,
        latentsync_guidance,
        crop_scale,
        keep_original_frame,
        mouth_mask_overlay,
        latentsync_deepcache,
        enhance_video,
        enhance_backend,
        mouth_mask_x,
        mouth_mask_y,
        mouth_mask_w,
        mouth_mask_h,
    )
    settings.update(
        {
            "voice_mode": voice_mode,
            "voice_name": voice_name,
            "voice_preset": voice_preset,
            "language": language,
            "provider": "voice_clone" if voice_mode == "clone" else "voice_preset",
            "output_format": "wav",
        }
    )

    with db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id, kind, status, title, input_video, input_audio, voice_text, voice_sample, settings_json,
                progress, message, created_at, updated_at
            ) VALUES (?, ?, 'queued', ?, ?, '', ?, ?, ?, 0, 'Waiting for GPU worker', ?, ?)
            """,
            (
                job_id,
                JOB_KIND_VOICE_LIPSYNC,
                title.strip() or "Voice lipsync video",
                str(video_path),
                text.strip(),
                sample_path,
                json.dumps(settings),
                now_ts(),
                now_ts(),
            ),
        )
        append_log(conn, job_id, "Voice+lipsync job created")
    dispatch_serverless_job(job_id)
    return JSONResponse({"job_id": job_id})


@app.get("/api/jobs")
def list_jobs() -> JSONResponse:
    with db() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 100").fetchall()
    return JSONResponse({"jobs": [row_to_job(row) for row in rows]})


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise api_error("JOB_NOT_FOUND", 404)
    return JSONResponse(row_to_job(row))


@app.get("/api/workers")
def list_workers() -> JSONResponse:
    with db() as conn:
        rows = conn.execute("SELECT * FROM workers ORDER BY last_seen_at DESC").fetchall()
    workers = []
    for row in rows:
        worker = dict(row)
        worker["online"] = worker["last_seen_at"] >= now_ts() - 60
        workers.append(worker)
    return JSONResponse({"workers": workers})


@app.get("/api/jobs/{job_id}/download")
def download_result(job_id: str) -> FileResponse:
    with db() as conn:
        row = conn.execute("SELECT result_video, result_audio, title, kind FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row or not (row["result_video"] or row["result_audio"]):
        raise api_error("RESULT_NOT_READY", 404)
    result = Path(row["result_audio"] if row["kind"] == JOB_KIND_VOICE else row["result_video"])
    if not result.exists():
        raise api_error("RESULT_MISSING", 404)
    media_type = audio_media_type(result) if row["kind"] == JOB_KIND_VOICE else "video/mp4"
    suffix = result.suffix or (".wav" if row["kind"] == JOB_KIND_VOICE else ".mp4")
    return FileResponse(str(result), media_type=media_type, filename=f"{row['title']}_{job_id[:8]}{suffix}")


@app.get("/api/jobs/{job_id}/preview")
def preview_result(job_id: str) -> FileResponse:
    with db() as conn:
        row = conn.execute("SELECT result_video, result_audio, kind FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row or not (row["result_video"] or row["result_audio"]):
        raise api_error("RESULT_NOT_READY", 404)
    result = Path(row["result_audio"] if row["kind"] == JOB_KIND_VOICE else row["result_video"])
    if not result.exists():
        raise api_error("RESULT_MISSING", 404)
    if row["kind"] == JOB_KIND_VOICE:
        preview = voice_preview_path(result)
        return FileResponse(str(preview), media_type=audio_media_type(preview))
    return FileResponse(str(result), media_type="video/mp4")


@app.post("/api/worker/heartbeat")
async def worker_heartbeat(
    payload: dict[str, Any],
    x_worker_token: str | None = Header(default=None),
) -> JSONResponse:
    require_worker_token(x_worker_token)
    worker_id = str(payload.get("worker_id") or "default-worker")[:120]
    backend = str(payload.get("backend") or "unknown")[:80]
    status = str(payload.get("status") or "online")[:40]
    current_job_id = payload.get("current_job_id")
    message = str(payload.get("message") or "")[:500]
    with db() as conn:
        conn.execute(
            """
            INSERT INTO workers (id, backend, status, current_job_id, message, last_seen_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                backend = excluded.backend,
                status = excluded.status,
                current_job_id = excluded.current_job_id,
                message = excluded.message,
                last_seen_at = excluded.last_seen_at
            """,
            (worker_id, backend, status, current_job_id, message, now_ts(), now_ts()),
        )
    return JSONResponse({"ok": True})


@app.get("/api/worker/jobs/{job_id}/{kind}")
def worker_download_input(job_id: str, kind: str, x_worker_token: str | None = Header(default=None)) -> FileResponse:
    require_worker_token(x_worker_token)
    if kind not in {"video", "audio", "voice-sample"}:
        raise api_error("INPUT_MISSING", 404)
    column = {"video": "input_video", "audio": "input_audio", "voice-sample": "voice_sample"}[kind]
    with db() as conn:
        row = conn.execute(f"SELECT {column} FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise api_error("JOB_NOT_FOUND", 404)
    path = Path(row[column])
    if not path.exists():
        raise api_error("INPUT_MISSING", 404)
    return FileResponse(str(path))


@app.get("/api/worker/next")
def worker_next(x_worker_token: str | None = Header(default=None)) -> JSONResponse:
    require_worker_token(x_worker_token)
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if not row:
            return JSONResponse({"job": None})
        conn.execute(
            """
            UPDATE jobs
            SET status = 'processing', progress = 5, message = 'GPU worker started',
                attempts = attempts + 1, started_at = COALESCE(started_at, ?), updated_at = ?
            WHERE id = ?
            """,
            (now_ts(), now_ts(), row["id"]),
        )
        append_log(conn, row["id"], "Worker claimed job")
    job = row_to_job(row, include_private=True)
    if job["kind"] == JOB_KIND_VOICE:
        if job.get("voice_sample"):
            job["voice_sample_url"] = f"{WORKER_BASE_URL}/api/worker/jobs/{job['id']}/voice-sample"
    elif job["kind"] == JOB_KIND_VOICE_LIPSYNC:
        job["video_url"] = f"{WORKER_BASE_URL}/api/worker/jobs/{job['id']}/video"
        if job.get("voice_sample"):
            job["voice_sample_url"] = f"{WORKER_BASE_URL}/api/worker/jobs/{job['id']}/voice-sample"
    else:
        job["video_url"] = f"{WORKER_BASE_URL}/api/worker/jobs/{job['id']}/video"
        job["audio_url"] = f"{WORKER_BASE_URL}/api/worker/jobs/{job['id']}/audio"
    return JSONResponse({"job": job})


@app.post("/api/worker/jobs/{job_id}/claim")
def worker_claim_job(job_id: str, x_worker_token: str | None = Header(default=None)) -> JSONResponse:
    require_worker_token(x_worker_token)
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise api_error("JOB_NOT_FOUND", 404)
        if row["status"] not in {"queued", "processing"}:
            return JSONResponse({"job": None, "status": row["status"]})
        conn.execute(
            """
            UPDATE jobs
            SET status = 'processing', progress = 5, message = 'RunPod worker started',
                attempts = attempts + 1, started_at = COALESCE(started_at, ?), updated_at = ?
            WHERE id = ?
            """,
            (now_ts(), now_ts(), row["id"]),
        )
        append_log(conn, row["id"], "RunPod worker claimed job")
    job = row_to_job(row, include_private=True)
    if job["kind"] == JOB_KIND_VOICE:
        if job.get("voice_sample"):
            job["voice_sample_url"] = f"{RUNPOD_CALLBACK_BASE_URL}/api/worker/jobs/{job['id']}/voice-sample"
    elif job["kind"] == JOB_KIND_VOICE_LIPSYNC:
        job["video_url"] = f"{RUNPOD_CALLBACK_BASE_URL}/api/worker/jobs/{job['id']}/video"
        if job.get("voice_sample"):
            job["voice_sample_url"] = f"{RUNPOD_CALLBACK_BASE_URL}/api/worker/jobs/{job['id']}/voice-sample"
    else:
        job["video_url"] = f"{RUNPOD_CALLBACK_BASE_URL}/api/worker/jobs/{job['id']}/video"
        job["audio_url"] = f"{RUNPOD_CALLBACK_BASE_URL}/api/worker/jobs/{job['id']}/audio"
    return JSONResponse({"job": job})


@app.post("/api/worker/jobs/{job_id}/log")
async def worker_log(
    job_id: str,
    payload: dict[str, Any],
    x_worker_token: str | None = Header(default=None),
) -> JSONResponse:
    require_worker_token(x_worker_token)
    progress = int(payload.get("progress", 0))
    message = str(payload.get("message", ""))[:500]
    line = str(payload.get("log", message))[:2000]
    with db() as conn:
        conn.execute(
            "UPDATE jobs SET progress = ?, message = ?, updated_at = ? WHERE id = ?",
            (max(0, min(progress, 99)), message, now_ts(), job_id),
        )
        if line:
            append_log(conn, job_id, line)
    return JSONResponse({"ok": True})


@app.post("/api/worker/jobs/{job_id}/complete")
async def worker_complete(
    job_id: str,
    result: UploadFile = File(...),
    x_worker_token: str | None = Header(default=None),
) -> JSONResponse:
    require_worker_token(x_worker_token)
    with db() as conn:
        row = conn.execute("SELECT kind FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        raise api_error("JOB_NOT_FOUND", 404)
    allowed = ALLOWED_AUDIO_EXT if row["kind"] == JOB_KIND_VOICE else ALLOWED_VIDEO_EXT
    default_name = "result.wav" if row["kind"] == JOB_KIND_VOICE else "result.mp4"
    result_ext = safe_ext(result.filename or default_name, allowed)
    result_path = STORAGE_DIR / job_id / f"result{result_ext}"
    await save_upload(result, result_path, MAX_UPLOAD_MB * 1024 * 1024)
    with db() as conn:
        if row["kind"] == JOB_KIND_VOICE:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'done', progress = 100, message = 'Voice ready',
                    result_audio = ?, finished_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(result_path), now_ts(), now_ts(), job_id),
            )
        else:
            conn.execute(
                """
                UPDATE jobs
                SET status = 'done', progress = 100, message = 'Done',
                    result_video = ?, finished_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(result_path), now_ts(), now_ts(), job_id),
            )
        append_log(conn, job_id, "Worker uploaded result")
    return JSONResponse({"ok": True})


@app.post("/api/worker/jobs/{job_id}/fail")
async def worker_fail(
    job_id: str,
    payload: dict[str, Any],
    x_worker_token: str | None = Header(default=None),
) -> JSONResponse:
    require_worker_token(x_worker_token)
    message = str(payload.get("message", "Render failed"))[:1000]
    with db() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'failed', message = ?, updated_at = ?, finished_at = ? WHERE id = ?",
            (message, now_ts(), now_ts(), job_id),
        )
        append_log(conn, job_id, message)
    return JSONResponse({"ok": True})


@app.post("/api/jobs/{job_id}/retry")
def retry_job(job_id: str) -> JSONResponse:
    with db() as conn:
        row = conn.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise api_error("JOB_NOT_FOUND", 404)
        conn.execute(
            """
            UPDATE jobs
            SET status = 'queued', progress = 0, message = 'Waiting for GPU worker',
                result_video = NULL, result_audio = NULL, updated_at = ?, finished_at = NULL
            WHERE id = ?
            """,
            (now_ts(), job_id),
        )
        append_log(conn, job_id, "Job requeued")
    dispatch_serverless_job(job_id)
    return JSONResponse({"ok": True})


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> JSONResponse:
    with db() as conn:
        row = conn.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise api_error("JOB_NOT_FOUND", 404)
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    job_dir = STORAGE_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
    return JSONResponse({"ok": True})
