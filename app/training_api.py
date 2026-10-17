from __future__ import annotations

import logging
import os
import shutil
import stat
import tempfile
import threading
import uuid
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator

import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile

from app.security import valid_api_key

logger = logging.getLogger(__name__)
CHUNK_SIZE = 1024 * 1024
MAX_ARCHIVE_FILES = 100_000
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class TrainingSettings:
    api_key: str = field(default_factory=lambda: os.getenv("TRAINING_API_KEY", os.getenv("API_KEY", "")))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("TRAINING_DATA_DIR", "training_runs")).resolve())
    base_model: str = field(default_factory=lambda: os.getenv("TRAINING_BASE_MODEL", "yolov8n.pt"))
    gpu_device: int = field(default_factory=lambda: int(os.getenv("TRAINING_GPU_DEVICE", "0")))
    max_upload_bytes: int = field(default_factory=lambda: int(os.getenv("TRAINING_MAX_UPLOAD_BYTES", str(2 * 1024**3))))
    max_extracted_bytes: int = field(default_factory=lambda: int(os.getenv("TRAINING_MAX_EXTRACTED_BYTES", str(20 * 1024**3))))


@dataclass
class TrainingJob:
    job_id: str
    status: str
    created_at: str
    epochs: int
    image_size: int
    batch: int
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    has_model: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "epochs": self.epochs,
            "image_size": self.image_size,
            "batch": self.batch,
            "has_model": self.has_model,
            "error": self.error,
        }


def _gpu_name(device_index: int) -> str:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in this environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise RuntimeError("Configured training GPU is not visible to this process")
    return str(torch.cuda.get_device_name(device_index))


def _safe_dataset_yaml(archive_path: Path, dataset_dir: Path, max_extracted_bytes: int) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=False)
    extracted_bytes = 0
    seen_paths: set[str] = set()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > MAX_ARCHIVE_FILES:
                raise ValueError("Dataset archive contains an invalid number of files")

            for entry in entries:
                name = entry.filename.replace("\\", "/")
                relative = PurePosixPath(name)
                if (
                    not name
                    or "\x00" in name
                    or relative.is_absolute()
                    or not relative.parts
                    or ":" in relative.parts[0]
                    or any(part in {"", ".", ".."} for part in relative.parts)
                ):
                    raise ValueError("Dataset archive contains an unsafe path")
                normalized = relative.as_posix()
                if normalized in seen_paths:
                    raise ValueError("Dataset archive contains duplicate paths")
                seen_paths.add(normalized)

                mode = entry.external_attr >> 16
                kind = stat.S_IFMT(mode)
                if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise ValueError("Dataset archive contains a special file")
                if entry.file_size < 0 or entry.file_size > max_extracted_bytes:
                    raise ValueError("Dataset archive exceeds the expanded size limit")
                extracted_bytes += entry.file_size
                if extracted_bytes > max_extracted_bytes:
                    raise ValueError("Dataset archive exceeds the expanded size limit")

                destination = dataset_dir.joinpath(*relative.parts)
                if not destination.resolve().is_relative_to(dataset_dir.resolve()):
                    raise ValueError("Dataset archive contains an unsafe path")
                if entry.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue

                destination.parent.mkdir(parents=True, exist_ok=True)
                copied = 0
                with archive.open(entry) as source, destination.open("xb") as target:
                    while chunk := source.read(CHUNK_SIZE):
                        copied += len(chunk)
                        if copied > entry.file_size or copied > max_extracted_bytes:
                            raise ValueError("Dataset archive exceeds the expanded size limit")
                        target.write(chunk)
                if copied != entry.file_size:
                    raise ValueError("Dataset archive contains an invalid file")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError("Dataset must be a valid, readable ZIP archive") from exc

    source_yaml = dataset_dir / "data.yaml"
    if not source_yaml.is_file():
        raise ValueError("ZIP must contain data.yaml at its root")
    if source_yaml.stat().st_size > 1024 * 1024:
        raise ValueError("data.yaml exceeds the 1 MiB limit")
    try:
        document = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("data.yaml is not valid YAML") from exc
    if not isinstance(document, dict):
        raise ValueError("data.yaml must contain a YAML mapping")

    names = document.get("names")
    if not isinstance(names, (list, dict)) or not names:
        raise ValueError("data.yaml must define class names in names")

    root = dataset_dir.resolve()
    for split in ("train", "val"):
        value = document.get(split)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"data.yaml must define a {split} image directory")
        relative = Path(value)
        resolved = (root / relative).resolve()
        if relative.is_absolute() or not resolved.is_relative_to(root) or not resolved.is_dir():
            raise ValueError(f"data.yaml {split} must point to a directory inside the ZIP")
        if not any(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in resolved.rglob("*")):
            raise ValueError(f"data.yaml {split} directory contains no supported images")

    document["path"] = str(root)
    document.pop("download", None)
    normalized_yaml = dataset_dir.parent / "data.yaml"
    normalized_yaml.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return normalized_yaml


def _train_with_ultralytics(
    dataset_yaml: Path,
    job_dir: Path,
    options: dict[str, int],
    device_index: int,
    base_model: str,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed in this environment") from exc

    model = YOLO(base_model)
    model.train(
        data=str(dataset_yaml),
        epochs=options["epochs"],
        imgsz=options["image_size"],
        batch=options["batch"],
        device=device_index,
        workers=2,
        project=str(job_dir / "ultralytics"),
        name="train",
        exist_ok=True,
    )
    save_dir = Path(model.trainer.save_dir)
    best_model = save_dir / "weights" / "best.pt"
    if not best_model.is_file():
        raise RuntimeError("Training completed without producing best.pt")
    artifact = job_dir / "best.pt"
    shutil.copy2(best_model, artifact)
    return artifact


class TrainingJobManager:
    def __init__(
        self,
        settings: TrainingSettings,
        trainer: Callable[[Path, Path, dict[str, int], int, str], Path] = _train_with_ultralytics,
        gpu_check: Callable[[int], str] = _gpu_name,
    ) -> None:
        self.settings = settings
        self.trainer = trainer
        self.gpu_check = gpu_check
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, TrainingJob] = {}
        self._active = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo-training")

    def gpu_name(self) -> str:
        return self.gpu_check(self.settings.gpu_device)

    def is_busy(self) -> bool:
        with self._lock:
            return self._active

    def submit(self, archive_path: Path, options: dict[str, int]) -> TrainingJob | None:
        with self._lock:
            if self._active:
                return None
            job_id = str(uuid.uuid4())
            job = TrainingJob(
                job_id=job_id,
                status="queued",
                created_at=datetime.now(timezone.utc).isoformat(),
                epochs=options["epochs"],
                image_size=options["image_size"],
                batch=options["batch"],
            )
            self._jobs[job_id] = job
            self._active = True
            self._executor.submit(self._run, job_id, archive_path, options)
            return job

    def _run(self, job_id: str, archive_path: Path, options: dict[str, int]) -> None:
        job_dir = self.settings.data_dir / job_id
        dataset_dir = job_dir / "dataset"
        try:
            job_dir.mkdir(parents=True, exist_ok=False)
            with self._lock:
                job = self._jobs[job_id]
                job.status = "running"
                job.started_at = datetime.now(timezone.utc).isoformat()

            dataset_yaml = _safe_dataset_yaml(archive_path, dataset_dir, self.settings.max_extracted_bytes)
            artifact = self.trainer(
                dataset_yaml,
                job_dir,
                options,
                self.settings.gpu_device,
                self.settings.base_model,
            )
            if not artifact.is_file() or not artifact.resolve().is_relative_to(job_dir.resolve()):
                raise RuntimeError("Training did not produce a valid model artifact")
            with self._lock:
                job = self._jobs[job_id]
                job.status = "succeeded"
                job.has_model = True
        except Exception as exc:
            logger.exception("Training job failed", extra={"job_id": job_id, "error_category": type(exc).__name__})
            with self._lock:
                job = self._jobs[job_id]
                job.status = "failed"
                job.error = type(exc).__name__
        finally:
            archive_path.unlink(missing_ok=True)
            shutil.rmtree(dataset_dir, ignore_errors=True)
            with self._lock:
                job = self._jobs[job_id]
                job.finished_at = datetime.now(timezone.utc).isoformat()
                self._active = False

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.as_dict() if job is not None else None

    def artifact(self, job_id: str) -> Path | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or not job.has_model:
                return None
            path = self.settings.data_dir / job_id / "best.pt"
            return path if path.is_file() else None

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def create_training_app(
    settings: TrainingSettings | None = None,
    trainer: Callable[[Path, Path, dict[str, int], int, str], Path] = _train_with_ultralytics,
    gpu_check: Callable[[int], str] = _gpu_name,
) -> FastAPI:
    config = settings or TrainingSettings()
    manager = TrainingJobManager(config, trainer, gpu_check)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.training_jobs = manager
        yield
        manager.close()

    application = FastAPI(
        title="YOLO Dataset Training API",
        version="1.0.0",
        description="Upload a YOLO detection dataset ZIP, train on the configured GPU, and retrieve the resulting model.",
        lifespan=lifespan,
    )
    application.state.training_jobs = manager

    def authenticate(provided_key: str | None) -> None:
        if len(config.api_key) < 32 or not valid_api_key(provided_key, config.api_key):
            raise HTTPException(status_code=401, detail="Valid X-API-Key header required")

    @application.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @application.get("/health/ready")
    async def ready() -> dict[str, Any]:
        if len(config.api_key) < 32:
            raise HTTPException(status_code=503, detail="TRAINING_API_KEY must be at least 32 characters")
        try:
            gpu = manager.gpu_name()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Configured CUDA training GPU is unavailable") from exc
        return {"status": "ready", "gpu": gpu, "training_busy": manager.is_busy()}

    @application.get("/health/gpu")
    async def gpu_status() -> dict[str, Any]:
        try:
            return {"cuda_available": True, "device": config.gpu_device, "name": manager.gpu_name()}
        except Exception:
            return {"cuda_available": False, "device": config.gpu_device}

    @application.post("/v1/training/jobs", status_code=202)
    async def create_training_job(
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> dict[str, Any]:
        authenticate(x_api_key)
        if manager.is_busy():
            raise HTTPException(status_code=503, detail="A training job is already active")
        try:
            manager.gpu_name()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Configured CUDA training GPU is unavailable") from exc

        content_length = request.headers.get("content-length")
        if not content_length or not content_length.isdigit():
            raise HTTPException(status_code=411, detail="Content-Length is required")
        if int(content_length) > config.max_upload_bytes + 65_536:
            raise HTTPException(status_code=413, detail="Dataset ZIP exceeds the configured upload limit")

        form = await request.form(max_files=1, max_fields=3)
        upload = form.get("file")
        archive_path: Path | None = None
        try:
            if not isinstance(upload, UploadFile):
                raise HTTPException(status_code=422, detail="Provide the dataset ZIP in multipart field 'file'")
            if not (upload.filename or "").lower().endswith(".zip"):
                raise HTTPException(status_code=415, detail="Dataset must be uploaded as a .zip file")

            options = {
                "epochs": _form_integer(form.get("epochs"), "epochs", 50, 1, 300),
                "image_size": _form_integer(form.get("image_size"), "image_size", 640, 320, 1280),
                "batch": _form_integer(form.get("batch"), "batch", 16, 1, 64),
            }
            with tempfile.NamedTemporaryFile(dir=config.data_dir, suffix=".zip", delete=False) as target:
                archive_path = Path(target.name)
                total = 0
                while chunk := await upload.read(CHUNK_SIZE):
                    total += len(chunk)
                    if total > config.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Dataset ZIP exceeds the configured upload limit")
                    target.write(chunk)
            if total == 0 or not zipfile.is_zipfile(archive_path):
                raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive")

            job = manager.submit(archive_path, options)
            if job is None:
                raise HTTPException(status_code=503, detail="A training job is already active")
            archive_path = None
            return {**job.as_dict(), "status_url": f"/v1/training/jobs/{job.job_id}"}
        finally:
            await form.close()
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)

    @application.get("/v1/training/jobs/{job_id}")
    async def training_job_status(
        job_id: str,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> dict[str, Any]:
        authenticate(x_api_key)
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Training job not found")
        return job

    @application.get("/v1/training/jobs/{job_id}/model")
    async def download_trained_model(
        job_id: str,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> FileResponse:
        authenticate(x_api_key)
        artifact = manager.artifact(job_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Trained model is not available")
        return FileResponse(artifact, filename=f"{job_id}-best.pt", media_type="application/octet-stream")

    return application


def _form_integer(value: Any, name: str, default: int, minimum: int, maximum: int) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"{name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise HTTPException(status_code=422, detail=f"{name} must be between {minimum} and {maximum}")
    return parsed


app = create_training_app()