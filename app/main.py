from fastapi import FastAPI, HTTPException

from app.schemas import TranscriptionRequest, TranscriptionResponse
from app.transcriber import (
    AudioExtractionError,
    TranscriptionError,
    ValidationError,
    get_runtime_config,
    transcribe_video,
)

app = FastAPI(title="Local Video Transcription API")


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "runtime": get_runtime_config()}


@app.post("/transcribe", response_model=TranscriptionResponse)
def transcribe(request: TranscriptionRequest) -> TranscriptionResponse:
    try:
        return transcribe_video(request.video_path)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except AudioExtractionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except TranscriptionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
