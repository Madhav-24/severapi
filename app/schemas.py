from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class BoxCoordinates(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Prediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    box_coordinates: BoxCoordinates


class ImageSize(BaseModel):
    width: int
    height: int


class PredictionResponse(BaseModel):
    request_id: str
    model: str
    image: ImageSize
    processing_time_ms: float
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    model: str
    model_version: str
    classes: dict[int, str]
    device: str
    image_size: int
    confidence_threshold: float
    iou_threshold: float
    model_loaded: bool


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    model_loaded: bool
    reason: str | None = None


class GpuResponse(BaseModel):
    cuda_available: bool
    device_name: str | None = None
    device_uuid: str | None = None
    configured_host_gpu_uuid: str | None = None
    identity_verified: bool | None = None
    container_device_index: int | None = None
    device_count: int = 0
    allocated_memory_bytes: int | None = None
    reserved_memory_bytes: int | None = None
    error: str | None = None
