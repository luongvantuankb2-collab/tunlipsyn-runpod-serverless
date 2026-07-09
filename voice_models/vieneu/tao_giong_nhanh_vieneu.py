# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import concurrent.futures
import math
import re
import time
from pathlib import Path

import numpy as np
from vieneu import Vieneu


MODEL_DIR = Path(__file__).resolve().parent / "VieNeu-TTS-v3-Turbo"
ONNX_DIR = MODEL_DIR / "onnx"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "vieneu_fast_lipsync.wav"


def patch_torchaudio_load_fallback() -> None:
    try:
        import soundfile as sf
        import torch
        import torchaudio
    except Exception:
        return

    original_load = torchaudio.load

    def load_with_soundfile_fallback(path, *args, **kwargs):
        try:
            return original_load(path, *args, **kwargs)
        except ImportError as exc:
            if "TorchCodec" not in str(exc):
                raise
            data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
            audio = torch.from_numpy(np.ascontiguousarray(data.T))
            return audio, sample_rate

    torchaudio.load = load_with_soundfile_fallback


def split_text(text: str, max_chars: int) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?;:。！？…])\s+", text)
    chunks: list[str] = []
    current = ""

    def flush_current() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            flush_current()
            parts = re.split(r"(?<=[,，])\s+", sentence)
            piece = ""
            for part in parts:
                if len(part) > max_chars:
                    if piece:
                        chunks.append(piece.strip())
                        piece = ""
                    for start in range(0, len(part), max_chars):
                        chunks.append(part[start:start + max_chars].strip())
                elif len(piece) + len(part) + 1 <= max_chars:
                    piece = f"{piece} {part}".strip()
                else:
                    chunks.append(piece.strip())
                    piece = part
            if piece:
                chunks.append(piece.strip())
            continue

        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            flush_current()
            current = sentence

    flush_current()
    return chunks


def dbfs_from_rms(rms: float) -> float:
    if rms <= 0:
        return -120.0
    return 20.0 * math.log10(rms)


def audio_to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    if audio.ndim > 1:
        if audio.shape[0] <= 8:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=-1)
    return np.clip(audio.reshape(-1), -1.0, 1.0)


def trim_start_index(audio: np.ndarray, sample_rate: int = 48_000) -> int:
    max_scan_seconds = 2.0
    chunk_seconds = 0.02
    silence_threshold = -35.0
    speech_threshold = -28.0
    min_silence_chunks = 4
    min_speech_chunks = 4
    chunk_size = max(1, int(sample_rate * chunk_seconds))
    scan_samples = min(len(audio), int(sample_rate * max_scan_seconds))
    if scan_samples <= chunk_size:
        return 0

    levels: list[float] = []
    for start in range(0, scan_samples - chunk_size + 1, chunk_size):
        chunk = audio[start:start + chunk_size]
        levels.append(dbfs_from_rms(float(np.sqrt(np.mean(np.square(chunk))))))

    def has_sustained_speech(index: int) -> bool:
        run = 0
        for level in levels[index:]:
            if level >= speech_threshold:
                run += 1
                if run >= min_speech_chunks:
                    return True
            else:
                run = 0
        return False

    # VieNeu sometimes emits a short loud artifact, then silence, then real speech.
    # Use the last early silence gap before sustained speech as the real start.
    saw_audio = False
    silence_run = 0
    silence_start = 0
    candidate_index = 0
    for index, level in enumerate(levels):
        if level >= speech_threshold:
            saw_audio = True

        if level <= silence_threshold:
            if silence_run == 0:
                silence_start = index
            silence_run += 1
            continue

        if saw_audio and silence_run >= min_silence_chunks and has_sustained_speech(index):
            candidate_index = max(candidate_index, index - 1)
        silence_run = 0

    if candidate_index:
        return int(max(0.0, candidate_index * chunk_seconds - 0.02) * sample_rate)

    if levels[0] <= silence_threshold:
        for index in range(len(levels)):
            if has_sustained_speech(index):
                return int(max(0.0, index * chunk_seconds - 0.02) * sample_rate)

    return 0


def trim_tail_index(audio: np.ndarray, sample_rate: int = 48_000) -> int:
    chunk_seconds = 0.02
    silence_threshold = -42.0
    keep_silence_seconds = 0.08
    chunk_size = max(1, int(sample_rate * chunk_seconds))
    if len(audio) <= chunk_size:
        return len(audio)

    last_voice_end = len(audio)
    for end in range(len(audio), chunk_size, -chunk_size):
        chunk = audio[end - chunk_size:end]
        level = dbfs_from_rms(float(np.sqrt(np.mean(np.square(chunk)))))
        if level > silence_threshold:
            last_voice_end = min(len(audio), end + int(sample_rate * keep_silence_seconds))
            break
    return max(chunk_size, last_voice_end)


def compress_long_silences(audio: np.ndarray, sample_rate: int = 48_000) -> tuple[np.ndarray, float]:
    chunk_seconds = 0.02
    silence_threshold = -30.0
    min_silence_seconds = 0.22
    keep_silence_seconds = 0.10
    chunk_size = max(1, int(sample_rate * chunk_seconds))
    min_silence_chunks = max(1, int(min_silence_seconds / chunk_seconds))
    keep_silence_chunks = max(1, int(keep_silence_seconds / chunk_seconds))
    if len(audio) <= chunk_size * min_silence_chunks:
        return audio, 0.0

    pieces: list[np.ndarray] = []
    removed = 0
    silence_start: int | None = None
    cursor = 0

    chunk_count = len(audio) // chunk_size
    for chunk_index in range(chunk_count):
        start = chunk_index * chunk_size
        end = start + chunk_size
        chunk = audio[start:end]
        level = dbfs_from_rms(float(np.sqrt(np.mean(np.square(chunk)))))

        if level <= silence_threshold:
            if silence_start is None:
                silence_start = start
            continue

        if silence_start is not None:
            silence_end = start
            silence_chunks = (silence_end - silence_start) // chunk_size
            if silence_chunks >= min_silence_chunks:
                keep_end = silence_start + keep_silence_chunks * chunk_size
                pieces.append(audio[cursor:keep_end])
                removed += max(0, silence_end - keep_end)
                cursor = silence_end
            silence_start = None

    if silence_start is not None:
        silence_end = chunk_count * chunk_size
        silence_chunks = (silence_end - silence_start) // chunk_size
        if silence_chunks >= min_silence_chunks:
            keep_end = silence_start + keep_silence_chunks * chunk_size
            pieces.append(audio[cursor:keep_end])
            removed += max(0, silence_end - keep_end)
            cursor = silence_end

    pieces.append(audio[cursor:])
    if not pieces:
        return audio, 0.0
    return np.concatenate(pieces).astype(np.float32, copy=False), removed / sample_rate


def sanitize_chunk_audio(audio: np.ndarray, text: str = "", sample_rate: int = 48_000) -> tuple[np.ndarray, float, float, float, bool]:
    original = audio_to_mono_float32(audio)
    cleaned = original.copy()
    total_trim_start = 0

    # Run twice because some generations contain artifact + silence + artifact + silence + speech.
    for _ in range(2):
        start = trim_start_index(cleaned, sample_rate)
        if start < int(sample_rate * 0.04):
            break
        cleaned = cleaned[start:]
        total_trim_start += start

    end = trim_tail_index(cleaned, sample_rate)
    trimmed_tail = max(0, len(cleaned) - end)
    cleaned = cleaned[:end]
    cleaned, removed_silence = compress_long_silences(cleaned, sample_rate)

    # Tiny fade avoids clicks after hard trimming.
    fade = min(int(sample_rate * 0.01), len(cleaned) // 4)
    if fade > 0:
        cleaned[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        cleaned[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    original_seconds = len(original) / sample_rate
    cleaned_seconds = len(cleaned) / sample_rate
    expected_min_seconds = max(1.0, len(text.strip()) / 30.0)
    if original_seconds >= 1.2 and (
        cleaned_seconds < 1.0
        or cleaned_seconds < original_seconds * 0.35
        or cleaned_seconds < expected_min_seconds
    ):
        fallback = original.copy()
        fallback_end = trim_tail_index(fallback, sample_rate)
        fallback_trimmed_tail = max(0, len(fallback) - fallback_end)
        fallback = fallback[:fallback_end]
        fallback, fallback_removed_silence = compress_long_silences(fallback, sample_rate)
        fallback_seconds = len(fallback) / sample_rate
        if fallback_seconds >= 1.0:
            fade = min(int(sample_rate * 0.01), len(fallback) // 4)
            if fade > 0:
                fallback[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
                fallback[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            return fallback, 0.0, fallback_trimmed_tail / sample_rate, fallback_removed_silence, False
        return original, 0.0, 0.0, 0.0, False

    return cleaned, total_trim_start / sample_rate, trimmed_tail / sample_rate, removed_silence, True


def concat_audio(parts: list[np.ndarray], sample_rate: int, silence_ms: int) -> np.ndarray:
    if not parts:
        return np.zeros(0, dtype=np.float32)
    silence = np.zeros(int(sample_rate * silence_ms / 1000), dtype=np.float32)
    merged: list[np.ndarray] = []
    for index, part in enumerate(parts):
        if index:
            merged.append(silence)
        merged.append(np.asarray(part, dtype=np.float32))
    return np.concatenate(merged)


def make_tts(args: argparse.Namespace) -> Vieneu:
    device = "auto"
    backend = "auto"
    if args.backend == "gpu":
        device = "cuda"
        backend = "pytorch"
    elif args.backend == "cpu":
        device = "cpu"
        backend = "onnx"

    return Vieneu(
        mode="v3turbo",
        backbone_repo=str(MODEL_DIR),
        onnx_dir=str(ONNX_DIR),
        device=device,
        backend=backend,
    )


def main() -> None:
    patch_torchaudio_load_fallback()
    parser = argparse.ArgumentParser(description="Tao giong VieNeu nhanh, uu tien doc mot lan de tranh loi ghep audio.")
    parser.add_argument("--text", help="Noi dung can doc.")
    parser.add_argument("--text-file", help="File .txt chua noi dung can doc.")
    parser.add_argument("--voice", default="Ngọc Lan", help="Giong co san, vi du: Ngọc Lan, Xuân Vĩnh, Bình An.")
    parser.add_argument("--ref-audio", help="File WAV/MP3 3-5 giay de clone giong rieng.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="File WAV dau ra.")
    parser.add_argument("--backend", choices=["auto", "gpu", "cpu"], default="gpu")
    parser.add_argument("--workers", type=int, default=1, help="So luong doan tao song song. Mac dinh 1 de tranh ghep loi.")
    parser.add_argument("--max-chars", type=int, default=1200, help="Do dai toi da moi doan. Mac dinh lon de khong cat nho TTS.")
    parser.add_argument("--silence-ms", type=int, default=80, help="Khoang lang chen giua cac doan neu bat buoc phai chia.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-frames", type=int, default=420)
    args = parser.parse_args()

    if not MODEL_DIR.exists() or not ONNX_DIR.exists():
        raise SystemExit(f"Khong thay model tai: {MODEL_DIR}")

    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        raise SystemExit("Can truyen --text hoac --text-file.")

    chunks = split_text(text, args.max_chars)
    if not chunks:
        raise SystemExit("Text rong.")

    workers = max(1, min(args.workers, 4))
    if workers != args.workers:
        print(f"Workers duoc gioi han ve {workers}; tren may nay 2 la toi uu.")

    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"So doan: {len(chunks)}")
    print(f"Workers: {workers}")
    print("Dang load VieNeu...")
    start = time.perf_counter()
    tts = make_tts(args)
    print(f"Backend dang dung: {tts.backend}")
    print(f"Load: {time.perf_counter() - start:.2f}s")

    ref_codes = None
    if args.ref_audio:
        print("Dang ma hoa giong clone...")
        ref_codes = tts.encode_reference(args.ref_audio)

    def infer_chunk(item: tuple[int, str]) -> tuple[int, np.ndarray, float]:
        index, chunk = item
        infer_args = {
            "temperature": args.temperature,
            "max_new_frames": args.max_new_frames,
            "apply_watermark": False,
        }
        if ref_codes is not None:
            infer_args["ref_codes"] = ref_codes
        else:
            infer_args["voice"] = args.voice

        chunk_start = time.perf_counter()
        audio = tts.infer(chunk, **infer_args)
        audio, trim_head, trim_tail, removed_silence, sanitized = sanitize_chunk_audio(audio, chunk, sample_rate=48_000)
        elapsed = time.perf_counter() - chunk_start
        trim_msg = f", cat dau {trim_head:.2f}s" if trim_head >= 0.04 else ""
        if trim_tail >= 0.04:
            trim_msg += f", cat cuoi {trim_tail:.2f}s"
        if removed_silence >= 0.04:
            trim_msg += f", nen lang {removed_silence:.2f}s"
        if not sanitized:
            trim_msg += ", bo qua cat vi audio qua ngan"
        print(f"Doan {index + 1:02d}/{len(chunks)}: {elapsed:.2f}s, {len(chunk)} ky tu{trim_msg}")
        return index, audio, elapsed

    gen_start = time.perf_counter()
    ordered_parts: list[np.ndarray | None] = [None] * len(chunks)
    elapsed_parts: list[float] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(infer_chunk, item) for item in enumerate(chunks)]
        for future in concurrent.futures.as_completed(futures):
            index, audio, elapsed = future.result()
            ordered_parts[index] = audio
            elapsed_parts.append(elapsed)

    parts = [part for part in ordered_parts if part is not None]
    merged = concat_audio(parts, sample_rate=48_000, silence_ms=args.silence_ms)
    tts.save(merged, str(output_path))
    total = time.perf_counter() - gen_start
    audio_seconds = len(merged) / 48_000

    print("Xong.")
    print(f"File: {output_path}")
    print(f"Thoi gian tao: {total:.2f}s")
    print(f"Do dai audio: {audio_seconds:.2f}s")
    print(f"RTF: {total / audio_seconds:.2f}")


if __name__ == "__main__":
    main()
