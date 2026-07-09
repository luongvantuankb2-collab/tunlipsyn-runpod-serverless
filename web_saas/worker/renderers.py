import os
import audioop
import json
import math
import re
import shlex
import sys
import time
import wave
import shutil
import subprocess
from pathlib import Path
from typing import Callable


Progress = Callable[[str, int], None]

DEFAULT_MOUTH_MASK = {"x": 0.32, "y": 0.26, "w": 0.36, "h": 0.22}
LEGACY_LOW_MOUTH_MASK = {"x": 0.37, "y": 0.43, "w": 0.26, "h": 0.15}


def find_ffmpeg() -> str | None:
    candidates = [
        os.environ.get("FFMPEG_BIN"),
        "/usr/bin/ffmpeg",
        "/bin/ffmpeg",
        "/opt/conda/bin/ffmpeg",
        shutil.which("ffmpeg"),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if not Path(candidate).exists():
            continue
        try:
            proc = subprocess.run(
                [
                    candidate,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=16x16:rate=1",
                    "-t",
                    "0.1",
                    "-c:v",
                    "libx264",
                    "-f",
                    "null",
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
            )
        except Exception:
            continue
        if proc.returncode == 0:
            return candidate
    return shutil.which("ffmpeg")


def run_command(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> None:
    proc = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if proc.returncode != 0:
        combined = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
        tail = combined.strip()
        if tail:
            tail = tail[-8000:]
            raise RuntimeError(
                f"Command failed with exit code {proc.returncode}: {' '.join(command)}\n"
                f"--- stdout/stderr tail ---\n{tail}"
            )
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(command)}")


def _run_ffmpeg_with_fallbacks(
    ffmpeg: str,
    commands: list[list[str]],
    output_path: Path,
    fallback_source: Path,
    timeout: int = 120,
    allow_copy_fallback: bool = True,
) -> None:
    last_error = None
    for command in commands:
        try:
            run_command(command, timeout=timeout)
            if output_path.exists() and output_path.stat().st_size > 0:
                return
        except RuntimeError as exc:
            last_error = exc
    if allow_copy_fallback:
        shutil.copy2(fallback_source, output_path)
    if allow_copy_fallback and output_path.exists() and output_path.stat().st_size > 0:
        return
    if last_error:
        raise last_error
    raise RuntimeError(f"FFmpeg fallback failed to create: {output_path}")


LIPSYNC_MODEL_PRESETS = {
    "v15": {
        "label": "LatentSync v1.5 256 / 8GB",
        "config": "configs/unet/stage2.yaml",
        "checkpoint": "checkpoints/latentsync_unet_v15.pt",
        "steps": "40",
        "guidance": "1.8",
        "min_vram_gb": 8,
    },
    "v16": {
        "label": "LatentSync v1.6 256 / 18GB",
        "config": "configs/unet/stage2.yaml",
        "checkpoint": "checkpoints/latentsync_unet_v16.pt",
        "steps": "20",
        "guidance": "1.5",
        "min_vram_gb": 18,
    },
    "efficient": {
        "label": "LatentSync v1.6 Efficient 256",
        "config": "configs/unet/stage2_efficient.yaml",
        "checkpoint": "checkpoints/latentsync_unet_v16.pt",
        "steps": "20",
        "guidance": "1.5",
        "min_vram_gb": 8,
    },
    "standard": {
        "label": "LatentSync v1.6 Standard 256",
        "config": "configs/unet/stage2.yaml",
        "checkpoint": "checkpoints/latentsync_unet_v16.pt",
        "steps": "20",
        "guidance": "1.5",
        "min_vram_gb": 18,
    },
    "high512": {
        "label": "LatentSync v1.6 512",
        "config": "configs/unet/stage2_512.yaml",
        "checkpoint": "checkpoints/latentsync_unet_v16.pt",
        "steps": "20",
        "guidance": "1.5",
        "min_vram_gb": 48,
    },
}


def select_lipsync_model(total_vram_gb: float, requested: str | None = None) -> str:
    requested_key = (requested or "").strip().lower()
    if requested_key in LIPSYNC_MODEL_PRESETS:
        return requested_key
    if total_vram_gb >= 48:
        return "high512"
    if total_vram_gb >= 18:
        return "v16"
    return "v15"


def python_json_probe(python_exe: str, code: str, env: dict[str, str] | None = None) -> dict:
    proc = subprocess.run(
        [python_exe, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "Python probe failed").strip())
    return json.loads(proc.stdout)


def gpu_runtime_info(python_exe: str, env: dict[str, str]) -> dict:
    return python_json_probe(
        python_exe,
        (
            "import json, torch; "
            "info={'cuda': torch.cuda.is_available(), 'torch': torch.__version__, "
            "'cuda_version': torch.version.cuda, 'gpu': None, 'capability': None, "
            "'total_vram_gb': 0, 'arch_list': []}; "
            "info['arch_list']=torch.cuda.get_arch_list() if hasattr(torch.cuda, 'get_arch_list') else []; "
            "info.update({'gpu': torch.cuda.get_device_name(0), "
            "'capability': torch.cuda.get_device_capability(0), "
            "'total_vram_gb': round(torch.cuda.get_device_properties(0).total_memory/(1024**3), 2)}) "
            "if torch.cuda.is_available() else None; "
            "print(json.dumps(info))"
        ),
        env=env,
    )


def media_duration(path: Path) -> float | None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return None
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stdout)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def effective_lipsync_duration(video_path: Path, audio_path: Path) -> float:
    audio_seconds = media_duration(audio_path) or 0
    video_seconds = media_duration(video_path) or 0
    if audio_seconds > 0:
        return audio_seconds
    if video_seconds > 0:
        return video_seconds
    return 6.0


def video_probe(path: Path) -> dict:
    ffmpeg = find_ffmpeg()
    info = {
        "path": str(path),
        "bytes": path.stat().st_size if path.exists() else 0,
        "duration_sec": None,
        "width": None,
        "height": None,
        "fps": None,
    }
    if not ffmpeg or not path.exists():
        return info
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = proc.stdout
    duration = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if duration:
        hours, minutes, seconds = duration.groups()
        info["duration_sec"] = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    video = re.search(r"Video:.*?,\s*(\d+)x(\d+).*?(?:(\d+(?:\.\d+)?)\s*fps)?", text)
    if video:
        info["width"] = int(video.group(1))
        info["height"] = int(video.group(2))
        if video.group(3):
            info["fps"] = float(video.group(3))
    if not info["fps"]:
        tbr = re.search(r"(\d+(?:\.\d+)?)\s*tbr", text)
        if tbr:
            info["fps"] = float(tbr.group(1))
    return info


def stats_line(label: str, stats: dict) -> str:
    mb = (stats.get("bytes") or 0) / (1024 * 1024)
    duration = stats.get("duration_sec")
    duration_text = f"{duration:.2f}s" if isinstance(duration, (int, float)) else "unknown"
    resolution = "unknown"
    if stats.get("width") and stats.get("height"):
        resolution = f"{stats['width']}x{stats['height']}"
    fps = stats.get("fps")
    fps_text = f"{fps:.2f}" if isinstance(fps, (int, float)) else "unknown"
    return f"{label}: {resolution}, {duration_text}, {fps_text}fps, {mb:.2f}MB"


def square_crop_params(width: int, height: int, crop_scale: float = 0.75, y_bias: float = 0.24) -> tuple[int, int, int]:
    full_side = min(int(width or 0), int(height or 0))
    scale = max(0.55, min(float(crop_scale or 0.75), 1.0))
    side = int(round(full_side * scale))
    if side <= 0:
        return 0, 0, 0
    side = max(256, min(side, full_side))
    x = max(0, int(round((int(width) - side) / 2)))
    y = max(0, int(round((int(height) - side) * y_bias)))
    return side, x, min(y, max(0, int(height) - side))


def lipsync_crop_scale(settings: dict) -> float:
    raw = os.environ.get("LATENTSYNC_CROP_SCALE") or settings.get("crop_scale") or 0.75
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.75
    return max(0.55, min(value, 1.0))


def ffmpeg_video_quality(settings: dict) -> tuple[str, str]:
    preset = str(settings.get("preset") or settings.get("quality_preset") or "auto").lower()
    if preset in {"ultra", "best"}:
        return "12", "slow"
    if preset in {"pro", "quality"}:
        return "15", "medium"
    return "18", "veryfast"


def ffmpeg_video_encode_variants(crf: str, preset: str) -> list[list[str]]:
    return [
        ["-c:v", "libx264", "-crf", crf, "-preset", preset, "-pix_fmt", "yuv420p"],
        ["-c:v", "libx264", "-crf", crf, "-pix_fmt", "yuv420p"],
        ["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p"],
    ]


def setting_bool(settings: dict, key: str, default: bool = False) -> bool:
    if key not in settings or settings.get(key) is None:
        return default
    return str(settings.get(key)).strip().lower() in {"1", "true", "yes", "on"}


def setting_int(settings: dict, key: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(settings.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def setting_float(settings: dict, key: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(settings.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def mouth_mask_settings(settings: dict) -> tuple[float, float, float, float]:
    mask_x = setting_float(settings, "mouth_mask_x", DEFAULT_MOUTH_MASK["x"], 0.0, 1.0)
    mask_y = setting_float(settings, "mouth_mask_y", DEFAULT_MOUTH_MASK["y"], 0.0, 1.0)
    mask_w = setting_float(settings, "mouth_mask_w", DEFAULT_MOUTH_MASK["w"], 0.05, 1.0)
    mask_h = setting_float(settings, "mouth_mask_h", DEFAULT_MOUTH_MASK["h"], 0.05, 1.0)
    is_legacy_low_mask = (
        abs(mask_x - LEGACY_LOW_MOUTH_MASK["x"]) < 0.015
        and abs(mask_y - LEGACY_LOW_MOUTH_MASK["y"]) < 0.015
        and abs(mask_w - LEGACY_LOW_MOUTH_MASK["w"]) < 0.015
        and abs(mask_h - LEGACY_LOW_MOUTH_MASK["h"]) < 0.015
    )
    if is_legacy_low_mask:
        return (
            DEFAULT_MOUTH_MASK["x"],
            DEFAULT_MOUTH_MASK["y"],
            DEFAULT_MOUTH_MASK["w"],
            DEFAULT_MOUTH_MASK["h"],
        )
    return mask_x, mask_y, mask_w, mask_h


def dbfs_from_rms(rms: int) -> float:
    if rms <= 0:
        return -120.0
    return 20.0 * math.log10(rms / 32768.0)


def detect_voice_trim_start(audio_path: Path) -> float:
    max_trim = float(os.environ.get("VOICE_TRIM_MAX_START_SECONDS", "1.5"))
    if max_trim <= 0:
        return 0.0

    chunk_seconds = 0.02
    silence_threshold = float(os.environ.get("VOICE_TRIM_SILENCE_DB", "-35"))
    speech_threshold = float(os.environ.get("VOICE_TRIM_SPEECH_DB", "-28"))
    min_silence_chunks = 4
    min_speech_chunks = 4

    try:
        with wave.open(str(audio_path), "rb") as wav:
            rate = wav.getframerate()
            width = wav.getsampwidth()
            channels = wav.getnchannels()
            chunk_frames = max(1, int(rate * chunk_seconds))
            chunks_to_scan = int(max_trim / chunk_seconds)
            levels: list[float] = []
            for _ in range(chunks_to_scan):
                data = wav.readframes(chunk_frames)
                if not data:
                    break
                if channels > 1:
                    data = audioop.tomono(data, width, 0.5, 0.5)
                levels.append(dbfs_from_rms(audioop.rms(data, width)))
    except (EOFError, OSError, wave.Error):
        return 0.0

    if not levels:
        return 0.0

    def has_sustained_speech(start_index: int) -> bool:
        run = 0
        for level in levels[start_index:]:
            if level >= speech_threshold:
                run += 1
                if run >= min_speech_chunks:
                    return True
            else:
                run = 0
        return False

    silence_run = 0
    saw_early_audio = False
    silence_start = 0
    for index, level in enumerate(levels):
        if level >= speech_threshold:
            saw_early_audio = True

        if level <= silence_threshold:
            if silence_run == 0:
                silence_start = index
            silence_run += 1
        else:
            if saw_early_audio and silence_run >= min_silence_chunks and has_sustained_speech(index):
                return max(0.0, index * chunk_seconds - 0.02)
            silence_run = 0

    if levels[0] <= silence_threshold:
        for index in range(len(levels)):
            if has_sustained_speech(index):
                return max(0.0, index * chunk_seconds - 0.02)

    return 0.0


def clean_voice_audio(audio_path: Path, progress: Progress) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg or not audio_path.exists():
        return
    staged_path = audio_path.with_name(f"{audio_path.stem}_staged.wav")
    cleaned_path = audio_path.with_name(f"{audio_path.stem}_cleaned.wav")
    progress("Cleaning generated audio", 90)
    run_command(
        [
            ffmpeg,
            "-y",
            "-fflags",
            "+genpts",
            "-i",
            str(audio_path),
            "-af",
            "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0,loudnorm=I=-18:TP=-2:LRA=11",
            "-map_metadata",
            "-1",
            "-vn",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(staged_path),
        ]
    )

    trim_start = detect_voice_trim_start(staged_path)
    if trim_start >= 0.04:
        progress(f"Trimming voice pre-roll: {trim_start:.2f}s", 91)
    run_command(
        [
            ffmpeg,
            "-y",
            "-fflags",
            "+genpts",
            "-i",
            str(staged_path),
            "-af",
            f"atrim=start={trim_start:.3f},asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0",
            "-map_metadata",
            "-1",
            "-vn",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(cleaned_path),
        ]
    )
    if cleaned_path.exists() and cleaned_path.stat().st_size > 1024:
        staged_seconds = media_duration(staged_path) or 0
        cleaned_seconds = media_duration(cleaned_path) or 0
        if staged_seconds >= 1.2 and (cleaned_seconds < 1.0 or cleaned_seconds < staged_seconds * 0.35):
            progress("Voice trim skipped: cleaned audio became too short", 91)
            staged_path.replace(audio_path)
            cleaned_path.unlink(missing_ok=True)
        else:
            cleaned_path.replace(audio_path)
    staged_path.unlink(missing_ok=True)


def prepare_lipsync_media(
    video_path: Path,
    audio_path: Path,
    work_dir: Path,
    progress: Progress,
    settings: dict | None = None,
) -> tuple[Path, Path]:
    settings = settings or {}
    ffmpeg = find_ffmpeg()
    video_seconds = media_duration(video_path)
    audio_seconds = media_duration(audio_path)
    if not ffmpeg or not video_seconds or not audio_seconds:
        progress("Media duration check skipped", 22)
        return video_path, audio_path

    progress(f"Media duration: video {video_seconds:.2f}s, audio {audio_seconds:.2f}s", 22)
    prepared_video = work_dir / "prepared_video.mp4"
    prepared_audio = audio_path
    target_seconds = audio_seconds

    info = video_probe(video_path)
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    crop_scale = lipsync_crop_scale(settings)
    side, crop_x, crop_y = square_crop_params(width, height, crop_scale=crop_scale)
    target_width = int(os.environ.get("LATENTSYNC_PREP_WIDTH", "512"))
    crf, encode_preset = ffmpeg_video_quality(settings)

    if side > 0:
        progress(f"Cropping face region for LatentSync: {side}px @ {crop_scale:.2f}", 23)
        crop_filter = (
            f"crop={side}:{side}:{crop_x}:{crop_y},"
            f"scale='min({target_width},iw)':-2:flags=lanczos,"
            "fps=25"
        )
    else:
        progress("Normalizing video for LatentSync", 23)
        crop_filter = f"scale='min({target_width},iw)':-2:flags=lanczos,fps=25"

    prep_commands: list[list[str]] = []
    for encode_args in ffmpeg_video_encode_variants(crf, encode_preset):
        prep_commands.append(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video_path),
                "-t",
                f"{target_seconds:.3f}",
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                crop_filter,
                *encode_args,
                "-movflags",
                "+faststart",
                str(prepared_video),
            ]
        )
    for encode_args in ffmpeg_video_encode_variants(crf, encode_preset):
        prep_commands.append(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video_path),
                "-t",
                f"{target_seconds:.3f}",
                "-map",
                "0:v:0",
                "-an",
                "-r",
                "25",
                *encode_args,
                "-movflags",
                "+faststart",
                str(prepared_video),
            ]
        )
    prep_commands.append(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-t",
            f"{target_seconds:.3f}",
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            str(prepared_video),
        ]
    )
    try:
        _run_ffmpeg_with_fallbacks(
            ffmpeg,
            prep_commands,
            prepared_video,
            video_path,
            timeout=max(90, int(target_seconds) + 90),
            allow_copy_fallback=False,
        )
    except RuntimeError:
        raise RuntimeError("Video normalize failed; could not trim input video to audio length.")

    if audio_seconds > video_seconds + 0.25:
        prepared_video = work_dir / "prepared_video_looped.mp4"
        progress("Audio is longer than video; looping video to match audio", 23)
        loop_commands: list[list[str]] = []
        for loop_source in (work_dir / "prepared_video.mp4", video_path):
            for encode_args in ffmpeg_video_encode_variants(crf, encode_preset):
                loop_commands.append(
                    [
                        ffmpeg,
                        "-y",
                        "-stream_loop",
                        "-1",
                        "-i",
                        str(loop_source),
                        "-t",
                        f"{audio_seconds:.3f}",
                        "-an",
                        *encode_args,
                        str(prepared_video),
                    ]
                )
        _run_ffmpeg_with_fallbacks(
            ffmpeg,
            loop_commands,
            prepared_video,
            work_dir / "prepared_video.mp4",
            timeout=max(90, int(audio_seconds) + 90),
            allow_copy_fallback=False,
        )
    elif video_seconds > audio_seconds + 0.25:
        progress("Audio is shorter than video; trimming video to audio length", 23)

    return prepared_video, prepared_audio


def render_mock(video_path: Path, audio_path: Path, result_path: Path, progress: Progress) -> None:
    progress("Mock render: muxing uploaded video and audio", 35)
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        shutil.copy2(video_path, result_path)
        return
    run_command(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(result_path),
        ]
    )
    progress("Mock render complete", 90)


def composite_lipsync_crop_to_original(
    source_video: Path,
    rendered_crop: Path,
    source_audio: Path,
    output_path: Path,
    work_dir: Path,
    settings: dict,
    progress: Progress,
    target_duration: float | None = None,
) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        shutil.copy2(rendered_crop, output_path)
        return

    source_info = video_probe(source_video)
    width = int(source_info.get("width") or 0)
    height = int(source_info.get("height") or 0)
    if width <= 0 or height <= 0:
        shutil.copy2(rendered_crop, output_path)
        return

    crop_scale = lipsync_crop_scale(settings)
    side, crop_x, crop_y = square_crop_params(width, height, crop_scale=crop_scale)
    if side <= 0:
        shutil.copy2(rendered_crop, output_path)
        return

    duration = target_duration or media_duration(source_audio) or media_duration(rendered_crop) or source_info.get("duration_sec") or 0
    if not isinstance(duration, (int, float)) or duration <= 0:
        duration = 6.0

    crf, encode_preset = ffmpeg_video_quality(settings)
    base_video = work_dir / "base_original_for_composite.mp4"
    loop_args: list[str] = []
    source_duration = source_info.get("duration_sec") or 0
    if not isinstance(source_duration, (int, float)) or source_duration < duration - 0.1:
        loop_args = ["-stream_loop", "-1"]

    progress("Preparing original frame for clean composite", 91)
    run_command(
        [
            ffmpeg,
            "-y",
            *loop_args,
            "-i",
            str(source_video),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"scale={width}:{height},fps=25",
            "-c:v",
            "libx264",
            "-crf",
            crf,
            "-preset",
            encode_preset,
            "-pix_fmt",
            "yuv420p",
            str(base_video),
        ],
        timeout=max(120, int(duration) + 120),
    )

    mouth_mask_overlay = setting_bool(settings, "mouth_mask_overlay", False)
    if mouth_mask_overlay:
        mask_x, mask_y, mask_w, mask_h = mouth_mask_settings(settings)
        box_x = max(0, min(side - 1, int(side * mask_x)))
        box_y = max(0, min(side - 1, int(side * mask_y)))
        box_w = max(4, min(side - box_x, int(side * mask_w)))
        box_h = max(4, min(side - box_y, int(side * mask_h)))
        blur_sigma = max(18, int(min(box_w, box_h) * 0.32))
        filter_complex = (
            f"[1:v]scale={side}:{side}:flags=lanczos,unsharp=5:5:0.35:3:3:0.15[fg];"
            f"color=black:s={side}x{side}:d={duration:.3f},format=gray,"
            f"drawbox=x={box_x}:y={box_y}:w={box_w}:h={box_h}:color=white:t=fill,"
            f"gblur=sigma={blur_sigma}[mask];"
            "[fg][mask]alphamerge[fgm0];"
            "[fgm0]chromakey=0x29a957:0.22:0.08[fgm];"
            f"[0:v][fgm]overlay={crop_x}:{crop_y}:format=auto[v]"
        )
    else:
        filter_complex = (
            f"[1:v]scale={side}:{side}:flags=lanczos,unsharp=5:5:0.55:3:3:0.25[crop];"
            f"[0:v][crop]overlay={crop_x}:{crop_y}:format=auto[v]"
        )

    progress("Compositing clean lipsync crop back to original frame", 92)
    run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(base_video),
            "-i",
            str(rendered_crop),
            "-i",
            str(source_audio),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "2:a:0?",
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-crf",
            crf,
            "-preset",
            encode_preset,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_path),
        ],
        timeout=max(180, int(duration) + 180),
    )
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Composite did not create output: {output_path}")


def trim_video_to_audio_duration(video_path: Path, audio_path: Path, progress: Progress) -> None:
    ffmpeg = find_ffmpeg()
    duration = media_duration(audio_path) or 0
    video_duration = media_duration(video_path) or 0
    if not ffmpeg or duration <= 0 or video_duration <= 0:
        return
    if video_duration <= duration + 0.08:
        return

    progress(f"Trimming final video to audio duration: {duration:.2f}s", 93)
    trimmed = video_path.with_name(f"{video_path.stem}_trimmed{video_path.suffix}")
    trim_commands = [
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(trimmed),
        ],
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(trimmed),
        ],
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "mpeg4",
            "-q:v",
            "3",
            "-c:a",
            "aac",
            "-shortest",
            str(trimmed),
        ],
    ]
    _run_ffmpeg_with_fallbacks(
        ffmpeg,
        trim_commands,
        trimmed,
        video_path,
        timeout=max(120, int(duration) + 120),
        allow_copy_fallback=False,
    )
    if trimmed.exists() and trimmed.stat().st_size > 0:
        trimmed.replace(video_path)


def render_latentsync(video_path: Path, audio_path: Path, result_path: Path, settings: dict, progress: Progress) -> None:
    stable_profile = os.environ.get("LATENTSYNC_FORCE_STABLE_PROFILE", "1").lower() not in {"0", "false", "no"}
    if stable_profile:
        settings = dict(settings or {})
        settings["lipsync_model"] = "v15"
        settings["latentsync_steps"] = 40
        settings["latentsync_guidance"] = 1.8
        settings["crop_scale"] = 0.75
        settings["mouth_mask_overlay"] = False
        settings["latentsync_deepcache"] = False

    latentsync_dir = Path(os.environ.get("LATENTSYNC_DIR", "/workspace/LatentSync")).resolve()
    python_exe = os.environ.get("LATENTSYNC_PYTHON") or sys.executable
    temp_dir = result_path.parent / "latentsync_temp"

    script = latentsync_dir / "scripts" / "inference.py"
    if not script.exists():
        raise RuntimeError(f"LatentSync script not found: {script}")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["LATENTSYNC_MOUTH_ONLY"] = "0"
    env["LATENTSYNC_VAE_PATH"] = os.environ.get(
        "LATENTSYNC_VAE_PATH",
        str(latentsync_dir / "checkpoints" / "sd-vae-ft-mse"),
    )
    env["PYTHONPATH"] = str(latentsync_dir) + os.pathsep + env.get("PYTHONPATH", "")

    runtime = gpu_runtime_info(python_exe, env)
    if not runtime.get("cuda"):
        raise RuntimeError("CUDA_NOT_READY: GPU CUDA is not available for LatentSync.")
    capability = runtime.get("capability") or [0, 0]
    arch = f"sm_{capability[0]}{capability[1]}"
    arch_list = runtime.get("arch_list") or []
    # RTX 4090 can run with CUDA wheels that omit sm_89 from get_arch_list().
    # The live CUDA tensor probe in gpu_runtime_info is the source of truth here.
    total_vram = float(runtime.get("total_vram_gb") or 0)
    requested_model = str(settings.get("lipsync_model") or os.environ.get("LATENTSYNC_MODEL_PRESET") or "").strip().lower()
    if requested_model == "auto":
        requested_model = ""
    model_key = select_lipsync_model(total_vram, requested_model)
    model_preset = LIPSYNC_MODEL_PRESETS[model_key]
    progress(f"Preflight: {model_preset['label']}", 21)
    config = os.environ.get(
        f"LATENTSYNC_CONFIG_{model_key.upper()}",
        str(latentsync_dir / model_preset["config"]),
    )
    checkpoint = os.environ.get(
        f"LATENTSYNC_CHECKPOINT_{model_key.upper()}",
        str(latentsync_dir / model_preset["checkpoint"]),
    )
    if not Path(checkpoint).exists():
        checkpoint = os.environ.get("LATENTSYNC_CHECKPOINT", str(latentsync_dir / "checkpoints" / "latentsync_unet.pt"))
    default_steps = os.environ.get(f"LATENTSYNC_STEPS_{model_key.upper()}", str(model_preset["steps"]))
    default_guidance = os.environ.get(f"LATENTSYNC_GUIDANCE_{model_key.upper()}", str(model_preset["guidance"]))
    default_steps = os.environ.get("LATENTSYNC_STEPS", default_steps)
    default_guidance = os.environ.get("LATENTSYNC_GUIDANCE", default_guidance)
    steps = setting_int(settings, "latentsync_steps", int(float(default_steps)), 5, 80)
    guidance = setting_float(settings, "latentsync_guidance", float(default_guidance), 0.5, 5.0)
    if model_key == "v15" and steps < 40:
        steps = 40
    min_vram = float(model_preset["min_vram_gb"])
    if total_vram and total_vram < min_vram:
        raise RuntimeError(
            f"GPU_OUT_OF_MEMORY: {model_preset['label']} needs about {min_vram:.0f}GB VRAM; "
            f"this GPU has {total_vram:.1f}GB. Choose the efficient lipsync model."
        )

    config_path = Path(config)
    checkpoint_path = Path(checkpoint)
    vae_path = Path(env["LATENTSYNC_VAE_PATH"])
    if not config_path.exists():
        raise RuntimeError(f"LATENTSYNC_NOT_READY: config not found: {config_path}")
    if not checkpoint_path.exists():
        raise RuntimeError(f"LATENTSYNC_NOT_READY: checkpoint not found: {checkpoint_path}")
    if not (vae_path / "config.json").exists():
        raise RuntimeError(f"LATENTSYNC_NOT_READY: local VAE not found: {vae_path}")

    progress(
        f"GPU OK: {runtime.get('gpu')} {total_vram:.1f}GB, torch {runtime.get('torch')}, model {model_preset['label']}, ckpt {checkpoint_path.name}",
        22,
    )
    source_video_path = video_path
    target_duration = effective_lipsync_duration(video_path, audio_path)
    video_path, audio_path = prepare_lipsync_media(video_path, audio_path, result_path.parent, progress, settings=settings)
    progress(f"LatentSync params: steps={steps}, guidance={guidance}, crop={lipsync_crop_scale(settings):.2f}", 24)
    keep_original_frame = setting_bool(settings, "keep_original_frame", True)
    latentsync_output_path = (
        result_path.parent / "latentsync_crop_result.mp4" if keep_original_frame else result_path
    )

    command = [
        python_exe,
        str(script),
        "--unet_config_path",
        str(config_path),
        "--inference_ckpt_path",
        str(checkpoint_path),
        "--video_path",
        str(video_path),
        "--audio_path",
        str(audio_path),
        "--video_out_path",
        str(latentsync_output_path),
        "--inference_steps",
        str(steps),
        "--guidance_scale",
        str(guidance),
        "--temp_dir",
        str(temp_dir),
    ]
    use_deepcache = setting_bool(
        settings,
        "latentsync_deepcache",
        os.environ.get("LATENTSYNC_DEEPCACHE", "1").lower() not in {"0", "false", "no"},
    )
    if use_deepcache:
        command.append("--enable_deepcache")

    progress("LatentSync started", 25)
    try:
        run_command(command, cwd=latentsync_dir, env=env)
    except RuntimeError as exc:
        text = str(exc)
        if "--enable_deepcache" in command and (
            "no kernel image" in text.lower()
            or "cuda error" in text.lower()
            or "cuda kernel errors" in text.lower()
            or "failed to execute" in text.lower()
        ):
            progress("LatentSync retrying without DeepCache", 26)
            retry_command = [part for part in command if part != "--enable_deepcache"]
            run_command(retry_command, cwd=latentsync_dir, env=env)
        else:
            raise
    progress("LatentSync complete", 90)
    if keep_original_frame:
        composite_lipsync_crop_to_original(
            source_video_path,
            latentsync_output_path,
            audio_path,
            result_path,
            result_path.parent,
            settings,
            progress,
            target_duration=target_duration,
        )
    trim_video_to_audio_duration(result_path, audio_path, progress)


def realbasicvsr_defaults() -> tuple[Path, str, str, str]:
    project_root = Path(__file__).resolve().parents[2]
    rbvsr_dir = Path(os.environ.get("REALBASICVSR_DIR", project_root / "engines" / "RealBasicVSR")).resolve()
    python_exe = os.environ.get("REALBASICVSR_PYTHON", sys.executable)
    config = os.environ.get("REALBASICVSR_CONFIG", str(rbvsr_dir / "configs" / "realbasicvsr_x4.py"))
    checkpoint = os.environ.get("REALBASICVSR_CHECKPOINT", str(rbvsr_dir / "checkpoints" / "RealBasicVSR_x4.pth"))
    return rbvsr_dir, python_exe, config, checkpoint


def realbasicvsr_json_probe(args: list[str]) -> dict:
    rbvsr_dir, python_exe, _, _ = realbasicvsr_defaults()
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run(
            [python_exe, *args],
            cwd=str(rbvsr_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except Exception as exc:
        return {"error": str(exc)}
    if proc.returncode != 0:
        return {"error": proc.stderr.strip() or proc.stdout.strip()}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON probe output", "stdout": proc.stdout[-1000:]}


def realbasicvsr_runtime_info() -> dict:
    return realbasicvsr_json_probe([
        "-c",
        (
            "import json, torch; "
            "print(json.dumps({"
            "'torch': torch.__version__, "
            "'cuda': torch.cuda.is_available(), "
            "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
            "'capability': torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None"
            "}))"
        ),
    ])


def video_quality_probe(video_path: Path) -> dict:
    rbvsr_dir, _, _, _ = realbasicvsr_defaults()
    script = rbvsr_dir / "video_quality_metrics.py"
    if not script.exists():
        return {"error": f"Quality script not found: {script}"}
    return realbasicvsr_json_probe([str(script), str(video_path)])


def mux_video_with_original_audio(video_path: Path, audio_source: Path, output_path: Path) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        shutil.copy2(video_path, output_path)
        return
    run_command(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-i",
            str(audio_source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-crf",
            os.environ.get("ENHANCE_FFMPEG_CRF", "18"),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_path),
        ]
    )


def enhance_video_realbasicvsr(video_path: Path, progress: Progress) -> Path:
    rbvsr_dir, python_exe, config, checkpoint = realbasicvsr_defaults()
    script = rbvsr_dir / "inference_realbasicvsr.py"
    config_path = Path(config)
    checkpoint_path = Path(checkpoint)

    if not script.exists():
        raise RuntimeError(f"RealBasicVSR script not found: {script}")
    if not config_path.exists():
        raise RuntimeError(f"RealBasicVSR config not found: {config_path}")
    if not checkpoint_path.exists():
        raise RuntimeError(
            f"RealBasicVSR checkpoint not found: {checkpoint_path}. "
            "Download RealBasicVSR_x4.pth into engines\\RealBasicVSR\\checkpoints."
        )

    input_stats = video_probe(video_path)
    fps = input_stats.get("fps") or 25
    raw_output = video_path.with_name(f"{video_path.stem}_realbasicvsr_raw.mp4")
    muxed_output = video_path.with_name(f"{video_path.stem}_realbasicvsr.mp4")
    max_seq_len = os.environ.get("REALBASICVSR_MAX_SEQ_LEN", "12")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(rbvsr_dir) + os.pathsep + env.get("PYTHONPATH", "")

    command = [
        python_exe,
        str(script),
        str(config_path),
        str(checkpoint_path),
        str(video_path),
        str(raw_output),
        "--max_seq_len",
        str(max_seq_len),
        "--fps",
        f"{fps:.6f}",
    ]
    progress("RealBasicVSR enhance started", 91)
    start = time.perf_counter()
    run_command(command, cwd=rbvsr_dir, env=env)
    model_seconds = time.perf_counter() - start
    if not raw_output.exists() or raw_output.stat().st_size < 1024:
        raise RuntimeError(f"RealBasicVSR did not create output: {raw_output}")

    progress("RealBasicVSR muxing original audio", 96)
    mux_video_with_original_audio(raw_output, video_path, muxed_output)
    if not muxed_output.exists() or muxed_output.stat().st_size < 1024:
        raise RuntimeError(f"Enhanced mux output missing: {muxed_output}")

    output_stats = video_probe(muxed_output)
    source_duration = input_stats.get("duration_sec") or 0
    realtime_factor = model_seconds / source_duration if source_duration else None
    benchmark = {
        "backend": "realbasicvsr",
        "runtime": realbasicvsr_runtime_info(),
        "model_seconds": round(model_seconds, 3),
        "seconds_per_video_second": round(realtime_factor, 3) if realtime_factor else None,
        "input": input_stats,
        "output": output_stats,
        "quality": {
            "input": video_quality_probe(video_path),
            "output": video_quality_probe(muxed_output),
        },
    }
    (video_path.parent / "enhance_benchmark.json").write_text(
        json.dumps(benchmark, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    progress(stats_line("Enhance input", input_stats), 97)
    progress(stats_line("Enhance output", output_stats), 98)
    if realtime_factor:
        progress(f"Enhance time: {model_seconds:.2f}s ({realtime_factor:.2f}s per video second)", 98)
    return muxed_output


def maybe_enhance_video(result_path: Path, settings: dict, progress: Progress) -> None:
    if not settings.get("enhance_video"):
        return
    backend = str(settings.get("enhance_backend") or "realbasicvsr").lower()
    if backend != "realbasicvsr":
        raise RuntimeError(f"Unknown enhance_backend={backend}")

    enhanced_path = enhance_video_realbasicvsr(result_path, progress)
    backup_path = result_path.with_name(f"{result_path.stem}_before_enhance{result_path.suffix}")
    if backup_path.exists():
        backup_path.unlink()
    result_path.replace(backup_path)
    enhanced_path.replace(result_path)
    progress("AI enhance complete", 99)


def render_voice_mock(sample_path: Path, text: str, result_path: Path, progress: Progress) -> None:
    progress("Mock voice clone: generating placeholder WAV", 35)
    sample_rate = 22050
    duration = max(1.0, min(12.0, len(text) / 14.0))
    frames = int(sample_rate * duration)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(result_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(frames):
            # Quiet speech-like placeholder so the full web flow can be tested without a TTS model.
            amp = 0.18 * math.sin(2 * math.pi * 3 * i / sample_rate) + 0.08
            value = int(16000 * amp * math.sin(2 * math.pi * 185 * i / sample_rate))
            wav.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))
    progress("Mock voice ready", 90)


def render_voice_command(sample_path: Path, text: str, result_path: Path, settings: dict, progress: Progress) -> None:
    command_template = os.environ.get("VOICE_CLONE_COMMAND", "").strip()
    if not command_template:
        raise RuntimeError("VOICE_CLONE_COMMAND is not set")
    text_path = result_path.with_suffix(".txt")
    text_path.write_text(text, encoding="utf-8")
    command = command_template.format(
        sample=str(sample_path),
        text=str(text_path),
        text_value=text,
        output=str(result_path),
        language=settings.get("language", "vi"),
        preset=settings.get("voice_preset", "balanced"),
    )
    progress("Voice clone command started", 30)
    proc = subprocess.run(command, shell=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Voice clone command failed with exit code {proc.returncode}")
    if not result_path.exists():
        raise RuntimeError(f"Voice clone command did not create output: {result_path}")
    progress("Voice clone complete", 90)


def find_vieneu_script() -> Path:
    configured = os.environ.get("VIENEU_SCRIPT", "").strip()
    if configured:
        script = Path(configured).expanduser().resolve()
        if script.exists():
            return script
        raise RuntimeError(f"VIENEU_SCRIPT not found: {script}")

    roots: list[Path] = []
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        roots.append(Path(local_appdata) / "Programs" / "TunLipsyncStudio")
    roots.extend([Path.cwd(), Path(__file__).resolve().parents[2]])

    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)
        for candidate in root.rglob("tao_giong_nhanh_vieneu.py"):
            if (candidate.parent / "VieNeu-TTS-v3-Turbo").exists():
                return candidate
    raise RuntimeError("VieNeu script not found. Set VIENEU_SCRIPT to tao_giong_nhanh_vieneu.py")


def vieneu_python_command() -> list[str]:
    configured = os.environ.get("VIENEU_PYTHON", "").strip()
    if configured:
        return shlex.split(configured)
    if os.name == "nt" and shutil.which("py"):
        return ["py", "-3.10"]
    return [sys.executable]


def render_voice_vieneu(sample_path: Path | None, text: str, result_path: Path, settings: dict, progress: Progress) -> None:
    script = find_vieneu_script()
    text_path = result_path.with_suffix(".txt")
    text_path.write_text(text, encoding="utf-8")
    backend = os.environ.get("VOICE_VIENEU_BACKEND", "gpu")
    workers = os.environ.get("VOICE_VIENEU_WORKERS", "1")
    max_chars = os.environ.get("VOICE_VIENEU_MAX_CHARS", "1200")
    silence_ms = os.environ.get("VOICE_VIENEU_SILENCE_MS", "80")
    temperature = os.environ.get("VOICE_VIENEU_TEMPERATURE", str(settings.get("temperature", "0.8")))
    max_new_frames = os.environ.get("VOICE_VIENEU_MAX_NEW_FRAMES", str(settings.get("max_new_frames", "420")))

    command = [
        *vieneu_python_command(),
        str(script),
        "--text-file",
        str(text_path),
        "--out",
        str(result_path),
        "--backend",
        backend,
        "--workers",
        workers,
        "--max-chars",
        max_chars,
        "--silence-ms",
        silence_ms,
        "--temperature",
        temperature,
        "--max-new-frames",
        max_new_frames,
    ]
    use_preset = settings.get("voice_mode") == "preset" or sample_path is None
    if use_preset:
        command.extend(["--voice", str(settings.get("voice_name") or "Ngọc Lan")])
    else:
        command.extend(["--ref-audio", str(sample_path)])

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    progress("VieNeu preset voice started" if use_preset else "VieNeu voice clone started", 30)
    run_command(command, cwd=script.parent, env=env)
    if not result_path.exists():
        raise RuntimeError(f"VieNeu did not create output: {result_path}")
    progress("VieNeu voice ready", 90)


def render_voice_job(
    sample_path: Path | None,
    text: str,
    result_path: Path,
    settings: dict,
    progress: Progress,
) -> None:
    backend = os.environ.get("VOICE_BACKEND", "mock").lower()
    if backend == "mock":
        render_voice_mock(sample_path, text, result_path, progress)
    elif backend == "vieneu":
        render_voice_vieneu(sample_path, text, result_path, settings, progress)
    elif backend in {"command", "f5tts", "xtts", "custom"}:
        render_voice_command(sample_path, text, result_path, settings, progress)
    else:
        raise RuntimeError(f"Unknown VOICE_BACKEND={backend}")
    clean_voice_audio(result_path, progress)


def render_lipsync_job(
    video_path: Path,
    audio_path: Path,
    result_path: Path,
    settings: dict,
    progress: Progress,
) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    backend = os.environ.get("RENDER_BACKEND", "mock").lower()
    if backend == "latentsync":
        render_latentsync(video_path, audio_path, result_path, settings, progress)
    elif backend == "mock":
        render_mock(video_path, audio_path, result_path, progress)
    else:
        raise RuntimeError(f"Unknown RENDER_BACKEND={backend}")
    maybe_enhance_video(result_path, settings, progress)

