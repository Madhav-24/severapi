from __future__ import annotations

from dataclasses import replace
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app

TEST_API_KEY = "test-key-0123456789abcdef0123456789"


class FakeManager:
    def __init__(self, ready: bool = True, fail_prediction: bool = False) -> None:
        self.ready = ready
        self.failure = None if ready else "ModelUnavailableError"
        self.fail_prediction = fail_prediction

    def load(self) -> None:
        return

    def close(self) -> None:
        return

    def record_failure(self, error: Exception) -> None:
        self.ready = False
        self.failure = type(error).__name__

    def info(self) -> dict[str, object]:
        return {
            "model": "test-model",
            "model_version": "test",
            "classes": {0: "person"},
            "device": "cuda:0",
            "image_size": 640,
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "model_loaded": self.ready,
        }

    def gpu_status(self) -> dict[str, object]:
        return {"cuda_available": False, "device_count": 0}

    def predict(self, image: Image.Image, confidence: float, iou: float) -> list[dict[str, object]]:
        if self.fail_prediction:
            raise RuntimeError("internal inference detail")
        return [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.9,
                "box_coordinates": {"x1": 1.0, "y1": 2.0, "x2": 10.0, "y2": 20.0},
            }
        ]


def png_bytes() -> bytes:
    image = Image.new("RGB", (32, 24), color="white")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def client() -> TestClient:
    settings = replace(Settings(), api_key=TEST_API_KEY, max_upload_bytes=1024 * 1024)
    with TestClient(create_app(settings, FakeManager())) as test_client:
        yield test_client
