import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel
import psutil

from app.schemas import RequestMetadata, Segment, TranscriptionResponse


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

_model: WhisperModel | None = None
_model_lock = threading.Lock()


class ValidationError(ValueError):
    pass


class AudioExtractionError(RuntimeError):
    pass


class TranscriptionError(RuntimeError):
    pass


def validate_video_path(video_path: str) -> Path:
    if not video_path.strip():
        raise ValidationError("Video path is required.")

    path = Path(video_path).expanduser()

    if not path.exists():
        raise ValidationError("Video file does not exist.")

    if not path.is_file():
        raise ValidationError("Video path must point to a file.")

    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValidationError(f"Unsupported video extension. Supported extensions: {supported}.")

    return path


def transcribe_video(video_path: str) -> TranscriptionResponse:
    request_started_at = time.perf_counter()
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)

    path = validate_video_path(video_path)
    file_size_mb = path.stat().st_size / (1024 * 1024)
    memory_start_mb = _rss_mb(process)
    extraction_seconds = 0.0
    transcription_seconds = 0.0

    with tempfile.TemporaryDirectory(prefix="video-transcript-") as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"

        extraction_started_at = time.perf_counter()
        extract_audio(path, wav_path)
        extraction_seconds = time.perf_counter() - extraction_started_at

        transcription_started_at = time.perf_counter()
        response = transcribe_audio(wav_path)
        transcription_seconds = time.perf_counter() - transcription_started_at

    total_seconds = time.perf_counter() - request_started_at
    memory_end_mb = _rss_mb(process)
    response.metadata = RequestMetadata(
        model_size=os.getenv("WHISPER_MODEL_SIZE", "small"),
        device=os.getenv("WHISPER_DEVICE", "cpu"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        processing_time_seconds=round(total_seconds, 3),
        audio_extraction_time_seconds=round(extraction_seconds, 3),
        transcription_time_seconds=round(transcription_seconds, 3),
        video_file_size_mb=round(file_size_mb, 3),
        cpu_percent=round(process.cpu_percent(interval=None), 1),
        memory_rss_mb_start=round(memory_start_mb, 3),
        memory_rss_mb_end=round(memory_end_mb, 3),
        memory_rss_mb_delta=round(memory_end_mb - memory_start_mb, 3),
        **detect_gpu(),
    )

    return response


def extract_audio(video_path: Path, wav_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise AudioExtractionError("ffmpeg is not installed or is not available on PATH.") from exc

    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or "Unknown ffmpeg error."
        raise AudioExtractionError(f"ffmpeg failed to extract audio: {details}")


def transcribe_audio(wav_path: Path) -> TranscriptionResponse:
    model = get_model()

    try:
        segments_iter, info = model.transcribe(str(wav_path), language="id")
        segments = [
            Segment(start=float(segment.start), end=float(segment.end), text=segment.text.strip())
            for segment in segments_iter
        ]
    except Exception as exc:
        raise TranscriptionError(f"faster-whisper failed to transcribe audio: {exc}") from exc

    full_text = " ".join(segment.text for segment in segments).strip()

    return TranscriptionResponse(
        language="id",
        duration=float(getattr(info, "duration", 0.0)),
        text=full_text,
        segments=segments,
        metadata=RequestMetadata(
            model_size=os.getenv("WHISPER_MODEL_SIZE", "small"),
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            processing_time_seconds=0.0,
            audio_extraction_time_seconds=0.0,
            transcription_time_seconds=0.0,
            video_file_size_mb=0.0,
            cpu_percent=0.0,
            memory_rss_mb_start=0.0,
            memory_rss_mb_end=0.0,
            memory_rss_mb_delta=0.0,
            **detect_gpu(),
        ),
    )


def get_model() -> WhisperModel:
    global _model

    if _model is None:
        with _model_lock:
            if _model is None:
                _model = WhisperModel(
                    os.getenv("WHISPER_MODEL_SIZE", "small"),
                    device=os.getenv("WHISPER_DEVICE", "cpu"),
                    compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
                )

    return _model


def get_runtime_config() -> dict[str, Any]:
    return {
        "model_size": os.getenv("WHISPER_MODEL_SIZE", "small"),
        "device": os.getenv("WHISPER_DEVICE", "cpu"),
        "compute_type": os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        "supported_video_extensions": sorted(SUPPORTED_VIDEO_EXTENSIONS),
        **detect_gpu(),
    }


def _rss_mb(process: psutil.Process) -> float:
    return process.memory_info().rss / (1024 * 1024)


def detect_gpu() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"gpu_available": False, "gpu_name": None}

    if result.returncode != 0:
        return {"gpu_available": False, "gpu_name": None}

    gpu_name = result.stdout.strip().splitlines()[0] if result.stdout.strip() else None
    return {"gpu_available": gpu_name is not None, "gpu_name": gpu_name}
