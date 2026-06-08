# FastAPI Local Video Transcription MVP

Minimal FastAPI backend for transcribing server-local video files with local `ffmpeg` and `faster-whisper`.

## Requirements

- Python 3.10+
- `ffmpeg` installed on the server and available on `PATH`
- Enough temporary disk space for extracted WAV files
- Python dependencies from `requirements.txt`

This MVP is synchronous. If videos can be long, move transcription into a background job queue later.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install `ffmpeg` if needed:

```bash
brew install ffmpeg
```

## Run

```bash
uvicorn app.main:app --reload
```

## Transcribe

```bash
curl -X POST http://127.0.0.1:8000/transcribe \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/Users/jojojow/Projects/auto-affiliate/video-transcript/sample/ssstik.io_@buser.sctv_1780221404608.mp4"}'
```

Response:

```json
{
  "language": "id",
  "duration": 123.4,
  "text": "...full transcript...",
  "segments": [
    {
      "start": 0.0,
      "end": 4.2,
      "text": "..."
    }
  ],
  "metadata": {
    "model_size": "small",
    "device": "cpu",
    "compute_type": "int8",
    "processing_time_seconds": 84.2,
    "audio_extraction_time_seconds": 1.3,
    "transcription_time_seconds": 82.7,
    "video_file_size_mb": 3.0,
    "cpu_percent": 78.5,
    "memory_rss_mb_start": 210.4,
    "memory_rss_mb_end": 912.8,
    "memory_rss_mb_delta": 702.4,
    "gpu_available": false,
    "gpu_name": null
  }
}
```

Supported video extensions: `.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.webm`.

## Configuration

Defaults are CPU-friendly:

```bash
export WHISPER_MODEL_SIZE=small
export WHISPER_DEVICE=cpu
export WHISPER_COMPUTE_TYPE=int8
```

Use `WHISPER_MODEL_SIZE=medium` if you can tolerate slower processing for potentially better quality.

## Manual Checks

```bash
python -m compileall app
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Acceptance checks:

- Valid local `.mp4` path returns `language`, `duration`, `text`, `segments`, and request `metadata`.
- Missing path returns HTTP `400`.
- Unsupported extension returns HTTP `400`.
- Missing or broken `ffmpeg` returns HTTP `500` with a clear error.
# video-transcript
