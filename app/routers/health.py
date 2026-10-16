from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas import GpuResponse, HealthResponse, ReadinessResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    return HealthResponse(status="alive")


@router.get("/ready", response_model=ReadinessResponse)
async def ready(request: Request) -> JSONResponse | ReadinessResponse:
    manager = request.app.state.model_manager
    if not manager.ready:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "model_loaded": False,
                "reason": manager.failure or "model_not_loaded",
            },
        )
    return ReadinessResponse(status="ready", model_loaded=True)


@router.get("/gpu", response_model=GpuResponse)
async def gpu(request: Request) -> GpuResponse:
    return GpuResponse(**request.app.state.model_manager.gpu_status())
