from pydantic import BaseModel


class TranscriptionRequest(BaseModel):
    video_path: str


class Segment(BaseModel):
    start: float
    end: float
    text: str


class RequestMetadata(BaseModel):
    model_size: str
    device: str
    compute_type: str
    processing_time_seconds: float
    audio_extraction_time_seconds: float
    transcription_time_seconds: float
    video_file_size_mb: float
    cpu_percent: float
    memory_rss_mb_start: float
    memory_rss_mb_end: float
    memory_rss_mb_delta: float
    gpu_available: bool
    gpu_name: str | None


class TranscriptionResponse(BaseModel):
    language: str
    duration: float
    text: str
    segments: list[Segment]
    metadata: RequestMetadata
