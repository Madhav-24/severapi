from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from threading import BoundedSemaphore
from typing import Any, AsyncIterator

import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings
from app.logging_config import configure_logging
from app.model_manager import ModelManager
from app.routers.health import router as health_router
from app.routers.predictions import router as predictions_router
from app.security import valid_api_key

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, model_manager: Any | None = None) -> FastAPI:
    configure_logging()
    config = settings or Settings()
    manager = model_manager if model_manager is not None else ModelManager(config)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        try:
            if len(config.api_key) < 32:
                raise RuntimeError("API_KEY must be configured with at least 32 characters")
            await anyio.to_thread.run_sync(manager.load)
        except Exception as exc:
            manager.record_failure(exc)
            logger.exception(
                "Model initialisation failed",
                extra={"model_id": config.model_id, "error_category": type(exc).__name__},
            )
        try:
            yield
        finally:
            manager.close()

    application = FastAPI(
        title="Construction Safety Inference API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    application.state.settings = config
    application.state.model_manager = manager
    application.state.inference_slots = BoundedSemaphore(config.max_inflight)
    if config.allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.allowed_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["content-type", "x-api-key"],
        )

    @application.middleware("http")
    async def request_context(request: Request, call_next: Any) -> JSONResponse:
        request.state.request_id = str(uuid.uuid4())
        started = time.perf_counter()
        admitted = False
        request.state.inference_slot_transferred = False
        response: JSONResponse | None = None
        if request.url.path == "/v1/predict":
            if not valid_api_key(request.headers.get("x-api-key"), config.api_key):
                response = error_response(request, 401, "authentication_failed", "Authentication required")
            elif not application.state.model_manager.ready:
                response = error_response(request, 503, "service_unavailable", "Service is not ready")
            else:
                content_length = request.headers.get("content-length")
                if not content_length or not content_length.isdigit():
                    response = error_response(request, 411, "content_length_required", "Content-Length is required")
                elif int(content_length) > config.max_upload_bytes + 65_536:
                    response = error_response(request, 413, "upload_too_large", "Upload exceeds the configured size limit")
                elif not application.state.inference_slots.acquire(blocking=False):
                    response = error_response(
                        request,
                        503,
                        "inference_capacity_full",
                        "Inference capacity is currently full",
                    )
                else:
                    admitted = True
        try:
            if response is None:
                response = await call_next(request)
        except Exception:
            logger.exception(
                "Unhandled request error",
                extra={"request_id": request.state.request_id, "error_category": "internal_error"},
            )
            response = JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "internal_error",
                        "message": "An unexpected error occurred",
                        "request_id": request.state.request_id,
                    }
                },
            )
        finally:
            if admitted and not request.state.inference_slot_transferred:
                application.state.inference_slots.release()
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info(
            "Request completed",
            extra={
                "request_id": request.state.request_id,
                "endpoint": request.url.path,
                "duration_ms": duration_ms,
                "status_code": response.status_code,
            },
        )
        return response

    def error_response(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": code,
                    "message": message,
                    "request_id": getattr(request.state, "request_id", "unknown"),
                }
            },
        )

    @application.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        mapping = {
            400: ("invalid_request", "The request could not be processed"),
            401: ("authentication_failed", "Authentication required"),
            403: ("forbidden", "Access denied"),
            404: ("not_found", "Resource not found"),
            413: ("upload_too_large", "Upload exceeds the configured size limit"),
            415: ("unsupported_media_type", "Unsupported image media type"),
            422: ("invalid_request", "Request validation failed"),
            503: ("service_unavailable", "Service is not ready or is at capacity"),
            500: ("inference_failed", "Inference failed"),
            504: ("inference_timeout", "Inference exceeded the configured time limit"),
        }
        code, message = mapping.get(exc.status_code, ("request_failed", "Request failed"))
        return error_response(request, exc.status_code, code, message)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(request, 422, "invalid_request", "Request validation failed")

    @application.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled application error",
            extra={"request_id": getattr(request.state, "request_id", "unknown"), "error_category": "internal_error"},
        )
        return error_response(request, 500, "internal_error", "An unexpected error occurred")

    application.include_router(health_router)
    application.include_router(predictions_router)
    return application


app = create_app()
