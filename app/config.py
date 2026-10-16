from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("MODEL_PATH", "/models/construction-safety.pt"))
    model_id: str = os.getenv("MODEL_ID", "construction-safety-model")
    model_version: str = os.getenv("MODEL_VERSION", "unspecified")
    api_key: str = os.getenv("API_KEY", "")
    gpu_uuid: str = os.getenv("GPU0_UUID", "")
    device: str = os.getenv("INFERENCE_DEVICE", "cuda:0")
    confidence: float = _bounded_float("CONFIDENCE_THRESHOLD", 0.25, 0.05, 1.0)
    iou: float = _bounded_float("IOU_THRESHOLD", 0.45, 0.1, 0.95)
    image_size: int = _bounded_int("IMAGE_SIZE", 640, 320, 1280)
    inference_timeout_seconds: float = _bounded_float("INFERENCE_TIMEOUT_SECONDS", 120.0, 1.0, 3600.0)
    max_upload_bytes: int = _bounded_int("MAX_UPLOAD_BYTES", 10_485_760, 1, 52_428_800)
    max_inflight: int = _bounded_int("MAX_INFLIGHT_INFERENCES", 1, 1, 8)
    allowed_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        if origin.strip()
    )

    def __post_init__(self) -> None:
        if self.device != "cuda:0":
            raise ValueError("INFERENCE_DEVICE must be cuda:0; select the production GPU in Docker")
        if "*" in self.allowed_origins:
            raise ValueError("CORS_ALLOWED_ORIGINS must not contain a wildcard")
        if not 0.05 <= self.confidence <= 1.0:
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0.05 and 1.0")
        if not 0.1 <= self.iou <= 0.95:
            raise ValueError("IOU_THRESHOLD must be between 0.1 and 0.95")
        if not 320 <= self.image_size <= 1280:
            raise ValueError("IMAGE_SIZE must be between 320 and 1280")
        if not 1.0 <= self.inference_timeout_seconds <= 3600.0:
            raise ValueError("INFERENCE_TIMEOUT_SECONDS must be between 1 and 3600")
        if not 1 <= self.max_upload_bytes <= 52_428_800:
            raise ValueError("MAX_UPLOAD_BYTES must be between 1 and 52428800")
        if not 1 <= self.max_inflight <= 8:
            raise ValueError("MAX_INFLIGHT_INFERENCES must be between 1 and 8")
