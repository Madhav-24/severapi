from __future__ import annotations

import asyncio
import logging
import time
import warnings
from io import BytesIO

from fastapi import APIRouter, File, Header, HTTPException, Query, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from app.model_manager import ModelUnavailableError
from app.schemas import ModelInfo, PredictionResponse
from app.security import valid_api_key

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["inference"])
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


@router.get("/model/info")
async def model_info(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> ModelInfo:
    settings = request.app.state.settings
    if not valid_api_key(x_api_key, settings.api_key):
        raise HTTPException(status_code=401, detail="Authentication required")
    manager = request.app.state.model_manager
    if not manager.ready:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return ModelInfo(**manager.info())


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    confidence: float | None = Query(default=None, ge=0.05, le=1.0),
    iou: float | None = Query(default=None, ge=0.1, le=0.95),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> PredictionResponse:
    settings = request.app.state.settings
    manager = request.app.state.model_manager
    request_id = request.state.request_id
    if not valid_api_key(x_api_key, settings.api_key):
        await file.close()
        raise HTTPException(status_code=401, detail="Authentication required")
    if not manager.ready:
        await file.close()
        raise HTTPException(status_code=503, detail="Model is not ready")

    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > settings.max_upload_bytes + 65_536:
        await file.close()
        raise HTTPException(status_code=413, detail="Upload exceeds the configured size limit")
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        await file.close()
        raise HTTPException(status_code=415, detail="Unsupported image media type")

    started = time.perf_counter()
    try:
        payload = await file.read(settings.max_upload_bytes + 1)
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(payload) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Upload exceeds the configured size limit")
        image = _decode_image(payload, file.content_type)

        selected_confidence = confidence if confidence is not None else settings.confidence
        selected_iou = iou if iou is not None else settings.iou
        width, height = image.size

        def execute_inference() -> list[dict[str, object]]:
            try:
                predictions = manager.predict(image, selected_confidence, selected_iou)
                logger.info(
                    "Inference completed",
                    extra={
                        "request_id": request_id,
                        "model_id": settings.model_id,
                        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                        "inference_success": True,
                    },
                )
                return predictions
            finally:
                image.close()
                request.app.state.inference_slots.release()

        request.state.inference_slot_transferred = True
        try:
            inference = asyncio.get_running_loop().run_in_executor(None, execute_inference)
        except Exception:
            request.state.inference_slot_transferred = False
            raise
        try:
            predictions = await asyncio.wait_for(
                asyncio.shield(inference),
                timeout=settings.inference_timeout_seconds,
            )
        except TimeoutError as exc:
            inference.add_done_callback(
                lambda completed: _log_late_inference_failure(completed, request_id, settings.model_id)
            )
            logger.warning(
                "Inference request timed out; GPU work continues until it finishes",
                extra={
                    "request_id": request_id,
                    "model_id": settings.model_id,
                    "error_category": "inference_timeout",
                    "inference_success": False,
                },
            )
            raise HTTPException(status_code=504, detail="Inference timed out") from exc
        except asyncio.CancelledError:
            inference.add_done_callback(
                lambda completed: _log_late_inference_failure(completed, request_id, settings.model_id)
            )
            raise
        return PredictionResponse(
            request_id=request_id,
            model=settings.model_id,
            image={"width": width, "height": height},
            processing_time_ms=round((time.perf_counter() - started) * 1000, 2),
            predictions=predictions,
        )
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(status_code=400, detail="Uploaded content is not a valid supported image") from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Model is not ready") from exc
    except Exception as exc:
        logger.exception(
            "Inference failed",
            extra={
                "request_id": request_id,
                "model_id": settings.model_id,
                "error_category": "inference_failure",
                "inference_success": False,
            },
        )
        raise HTTPException(status_code=500, detail="Inference failed") from exc
    finally:
        image_ref = locals().get("image")
        if image_ref is not None and not request.state.inference_slot_transferred:
            image_ref.close()
        await file.close()


def _log_late_inference_failure(
    future: asyncio.Future[list[dict[str, object]]],
    request_id: str,
    model_id: str,
) -> None:
    if future.cancelled():
        return
    error = future.exception()
    if error is not None:
        logger.error(
            "Inference failed after the request timed out",
            extra={
                "request_id": request_id,
                "model_id": model_id,
                "error_category": "inference_failure",
                "inference_success": False,
            },
            exc_info=(type(error), error, error.__traceback__),
        )


def _decode_image(payload: bytes, content_type: str | None) -> Image.Image:
    expected_formats = {
        "image/jpeg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(payload)) as source:
            if source.format not in ALLOWED_FORMATS:
                raise UnidentifiedImageError("Unsupported image format")
            if content_type and source.format != expected_formats.get(content_type.lower()):
                raise UnidentifiedImageError("Image content does not match declared media type")
            source.verify()
        with Image.open(BytesIO(payload)) as source:
            source.load()
            return source.convert("RGB")
