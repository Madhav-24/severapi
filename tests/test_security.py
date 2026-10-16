from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import TEST_API_KEY


def test_model_info_requires_api_key(client: TestClient) -> None:
    response = client.get("/v1/model/info")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_failed"
    assert "request_id" in response.json()["error"]


def test_model_info_accepts_api_key(client: TestClient) -> None:
    response = client.get("/v1/model/info", headers={"X-API-Key": TEST_API_KEY})
    assert response.status_code == 200
    assert response.json()["model"] == "test-model"
