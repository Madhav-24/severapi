from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._names: dict[int, str] = {}
        self._failure: str | None = None
        self._inference_lock = Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None and self._failure is None

    @property
    def failure(self) -> str | None:
        return self._failure

    def load(self) -> None:
        model_path = Path(self.settings.model_path)
        if not model_path.is_file():
            raise ModelUnavailableError("Configured model file is missing")
        if not self.settings.gpu_uuid:
            raise ModelUnavailableError("GPU0_UUID is not configured")

        import torch

        if not torch.cuda.is_available():
            raise ModelUnavailableError("CUDA is not available in the inference container")
        if torch.cuda.device_count() != 1:
            raise ModelUnavailableError("Expected exactly one visible GPU; check GPU isolation")

        device_uuid: str | None = None
        properties = torch.cuda.get_device_properties(0)
        raw_uuid = getattr(properties, "uuid", None)
        if raw_uuid is not None:
            device_uuid = raw_uuid.decode() if isinstance(raw_uuid, bytes) else str(raw_uuid)
        if device_uuid and self._normalise_uuid(device_uuid) != self._normalise_uuid(self.settings.gpu_uuid):
            raise ModelUnavailableError("Visible GPU UUID does not match configured production GPU")

        from ultralytics import YOLO

        model = YOLO(str(model_path))
        model.to(self.settings.device)
        self._model = model
        names = model.names
        self._names = (
            {int(key): str(value) for key, value in names.items()}
            if isinstance(names, dict)
            else {index: str(value) for index, value in enumerate(names)}
        )
        self._failure = None
        logger.info(
            "Model loaded",
            extra={"model_id": self.settings.model_id},
        )

    def record_failure(self, error: Exception) -> None:
        self._model = None
        self._failure = type(error).__name__

    def close(self) -> None:
        with self._inference_lock:
            if self._model is None:
                return
            self._model = None
            self._names.clear()
            import torch

            torch.cuda.empty_cache()
            logger.info("Model unloaded", extra={"model_id": self.settings.model_id})

    def predict(self, image: Any, confidence: float, iou: float) -> list[dict[str, Any]]:
        with self._inference_lock:
            if not self.ready:
                raise ModelUnavailableError("Model is not ready")
            results = self._model.predict(
                source=image,
                conf=confidence,
                iou=iou,
                imgsz=self.settings.image_size,
                device=0,
                verbose=False,
            )
            if not results:
                return []

            result = results[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return []
            coordinates = boxes.xyxy.detach().cpu().tolist()
            class_ids = boxes.cls.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            predictions: list[dict[str, Any]] = []
            for raw_box, raw_class_id, raw_confidence in zip(coordinates, class_ids, confidences):
                class_id = int(raw_class_id)
                if class_id not in self._names:
                    raise RuntimeError("Model returned an unknown class ID")
                predictions.append(
                    {
                        "class_id": class_id,
                        "class_name": self._names[class_id],
                        "confidence": float(raw_confidence),
                        "box_coordinates": {
                            "x1": float(raw_box[0]),
                            "y1": float(raw_box[1]),
                            "x2": float(raw_box[2]),
                            "y2": float(raw_box[3]),
                        },
                    }
                )
            return predictions

    def info(self) -> dict[str, Any]:
        return {
            "model": self.settings.model_id,
            "model_version": self.settings.model_version,
            "classes": self._names.copy(),
            "device": self.settings.device,
            "image_size": self.settings.image_size,
            "confidence_threshold": self.settings.confidence,
            "iou_threshold": self.settings.iou,
            "model_loaded": self.ready,
        }

    def gpu_status(self) -> dict[str, Any]:
        try:
            import torch

            available = bool(torch.cuda.is_available())
            count = int(torch.cuda.device_count()) if available else 0
            if not available or count == 0:
                return {
                    "cuda_available": False,
                    "device_count": count,
                    "configured_host_gpu_uuid": self.settings.gpu_uuid or None,
                    "error": "CUDA device unavailable",
                }
            properties = torch.cuda.get_device_properties(0)
            raw_uuid = getattr(properties, "uuid", None)
            uuid = None
            if raw_uuid is not None:
                uuid = raw_uuid.decode() if isinstance(raw_uuid, bytes) else str(raw_uuid)
            identity_verified = (
                self._normalise_uuid(uuid) == self._normalise_uuid(self.settings.gpu_uuid)
                if uuid and self.settings.gpu_uuid
                else None
            )
            return {
                "cuda_available": True,
                "device_name": str(properties.name),
                "device_uuid": uuid,
                "configured_host_gpu_uuid": self.settings.gpu_uuid or None,
                "identity_verified": identity_verified,
                "container_device_index": 0,
                "device_count": count,
                "allocated_memory_bytes": int(torch.cuda.memory_allocated(0)),
                "reserved_memory_bytes": int(torch.cuda.memory_reserved(0)),
            }
        except Exception as exc:
            logger.exception("Unable to inspect CUDA device")
            return {
                "cuda_available": False,
                "device_count": 0,
                "configured_host_gpu_uuid": self.settings.gpu_uuid or None,
                "error": type(exc).__name__,
            }

    @staticmethod
    def _normalise_uuid(value: str) -> str:
        return value.lower().replace("gpu-", "").replace("-", "")
