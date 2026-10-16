from __future__ import annotations

from dataclasses import replace
import time

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import FakeManager, TEST_API_KEY, png_bytes


AUTH = {"X-API-Key": TEST_API_KEY}


def test_prediction_response_shape(client: TestClient) -> None:
    response = client.post(
        "/v1/predict",
        headers=AUTH,
        files={"file": ("sample.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["image"] == {"width": 32, "height": 24}
    assert data["request_id"] == response.headers["X-Request-ID"]
    assert data["predictions"][0]["class_name"] == "person"
    assert data["predictions"][0]["box_coordinates"] == {
        "x1": 1.0,
        "y1": 2.0,
        "x2": 10.0,
        "y2": 20.0,
    }


def test_prediction_requires_authentication(client: TestClient) -> None:
    response = client.post(
        "/v1/predict",
        files={"file": ("sample.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 401


def test_empty_upload_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/predict",
        headers=AUTH,
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert response.status_code == 400


def test_malformed_image_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/predict",
        headers=AUTH,
        files={"file": ("bad.png", b"not an image", "image/png")},
    )
    assert response.status_code == 400


def test_unsupported_content_type_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/predict",
        headers=AUTH,
        files={"file": ("sample.gif", b"GIF89a", "image/gif")},
    )
    assert response.status_code == 415


def test_image_mime_type_must_match_decodable_format(client: TestClient) -> None:
    response = client.post(
        "/v1/predict",
        headers=AUTH,
        files={"file": ("sample.jpg", png_bytes(), "image/jpeg")},
    )
    assert response.status_code == 400


def test_oversize_upload_is_rejected() -> None:
    settings = replace(Settings(), api_key=TEST_API_KEY, max_upload_bytes=64)
    with TestClient(create_app(settings, FakeManager())) as client:
        response = client.post(
            "/v1/predict",
            headers=AUTH,
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )
    assert response.status_code == 413


def test_inference_capacity_rejects_request_before_upload_processing(client: TestClient) -> None:
    assert client.app.state.inference_slots.acquire(blocking=False)
    try:
        response = client.post(
            "/v1/predict",
            headers=AUTH,
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )
    finally:
        client.app.state.inference_slots.release()
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "inference_capacity_full"


def test_missing_model_rejects_prediction_before_upload_processing() -> None:
    settings = replace(Settings(), api_key=TEST_API_KEY)
    with TestClient(create_app(settings, FakeManager(ready=False))) as client:
        response = client.post(
            "/v1/predict",
            headers=AUTH,
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"


def test_inference_timeout_keeps_slot_until_gpu_work_finishes() -> None:
    settings = replace(Settings(), api_key=TEST_API_KEY, inference_timeout_seconds=1.0)

    class SlowManager(FakeManager):
        def predict(self, image, confidence, iou):
            time.sleep(1.1)
            return super().predict(image, confidence, iou)

    with TestClient(create_app(settings, SlowManager())) as client:
        response = client.post(
            "/v1/predict",
            headers=AUTH,
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )
        assert response.status_code == 504
        assert not client.app.state.inference_slots.acquire(blocking=False)
        deadline = time.monotonic() + 1.5
        acquired = False
        while time.monotonic() < deadline and not acquired:
            acquired = client.app.state.inference_slots.acquire(blocking=False)
            if not acquired:
                time.sleep(0.01)
        assert acquired
        client.app.state.inference_slots.release()


def test_inference_failure_does_not_leak_exception_text() -> None:
    settings = replace(Settings(), api_key=TEST_API_KEY)
    with TestClient(create_app(settings, FakeManager(fail_prediction=True))) as client:
        response = client.post(
            "/v1/predict",
            headers=AUTH,
            files={"file": ("sample.png", png_bytes(), "image/png")},
        )
    assert response.status_code == 500
    assert "internal inference detail" not in response.text
    assert response.json()["error"]["code"] == "inference_failed"


def test_query_threshold_limits_are_enforced(client: TestClient) -> None:
    response = client.post(
        "/v1/predict?confidence=0.01",
        headers=AUTH,
        files={"file": ("sample.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 422
