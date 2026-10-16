from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.main import create_app
from app.model_manager import ModelManager, ModelUnavailableError
from tests.conftest import FakeManager, TEST_API_KEY


def test_liveness_does_not_claim_model_ready(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_swagger_docs_show_image_upload_endpoint(client: TestClient) -> None:
    docs = client.get("/docs")
    schema = client.get("/openapi.json")
    assert docs.status_code == 200
    assert "Swagger UI" in docs.text
    assert schema.status_code == 200
    openapi = schema.json()
    operation = openapi["paths"]["/v1/predict"]["post"]
    upload_schema_ref = operation["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    schema_name = upload_schema_ref.rsplit("/", 1)[-1]
    upload_schema = openapi["components"]["schemas"][schema_name]
    assert upload_schema["properties"]["file"]["format"] == "binary"


def test_readiness_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_missing_model_fails_readiness() -> None:
    manager = FakeManager(ready=False)
    settings = replace(Settings(), api_key=TEST_API_KEY)
    with TestClient(create_app(settings, manager)) as client:
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_model_manager_reports_missing_weights_without_loading_runtime(tmp_path: Path) -> None:
    manager = ModelManager(
        replace(
            Settings(),
            model_path=tmp_path / "missing.pt",
            gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
        )
    )
    with pytest.raises(ModelUnavailableError, match="model file is missing"):
        manager.load()


def test_gpu_endpoint_reports_measured_unavailable_state(client: TestClient) -> None:
    response = client.get("/health/gpu")
    assert response.status_code == 200
    assert response.json()["cuda_available"] is False
