from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
import yaml

from app.training_api import TrainingSettings, create_training_app
from tests.conftest import TEST_API_KEY


def dataset_zip(*, traversal: bool = False) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "data.yaml",
            "train: images/train\nval: images/val\nnames:\n  0: helmet\ndownload: echo unsafe\n",
        )
        archive.writestr("images/train/sample.jpg", b"train-image")
        archive.writestr("images/val/sample.jpg", b"val-image")
        if traversal:
            archive.writestr("../outside.txt", "must not escape")
    return output.getvalue()


def wait_for_terminal_status(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(100):
        response = client.get(
            f"/v1/training/jobs/{job_id}",
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 200
        job = response.json()
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError("Training job did not finish")


def test_training_requires_api_key(tmp_path: Path) -> None:
    settings = TrainingSettings(api_key=TEST_API_KEY, data_dir=tmp_path)
    with TestClient(create_training_app(settings, gpu_check=lambda _: "Test GPU")) as client:
        response = client.get("/v1/training/jobs/unknown")
    assert response.status_code == 401


def test_training_job_runs_and_model_can_be_downloaded(tmp_path: Path) -> None:
    def fake_trainer(
        dataset_yaml: Path,
        job_dir: Path,
        options: dict[str, int],
        device_index: int,
        base_model: str,
    ) -> Path:
        assert dataset_yaml.is_file()
        assert "download" not in yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        assert options == {"epochs": 3, "image_size": 640, "batch": 4}
        assert device_index == 0
        assert base_model == "yolov8n.pt"
        artifact = job_dir / "best.pt"
        artifact.write_bytes(b"trained-model")
        return artifact

    settings = TrainingSettings(api_key=TEST_API_KEY, data_dir=tmp_path)
    app = create_training_app(settings, trainer=fake_trainer, gpu_check=lambda _: "Test GPU")
    headers = {"X-API-Key": TEST_API_KEY}
    with TestClient(app) as client:
        response = client.post(
            "/v1/training/jobs",
            headers=headers,
            data={"epochs": "3", "batch": "4"},
            files={"file": ("dataset.zip", dataset_zip(), "application/zip")},
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        job = wait_for_terminal_status(client, job_id)
        assert job["status"] == "succeeded"
        assert job["has_model"] is True

        model_response = client.get(f"/v1/training/jobs/{job_id}/model", headers=headers)
        assert model_response.status_code == 200
        assert model_response.content == b"trained-model"


def test_training_rejects_archive_path_traversal(tmp_path: Path) -> None:
    settings = TrainingSettings(api_key=TEST_API_KEY, data_dir=tmp_path)
    app = create_training_app(settings, gpu_check=lambda _: "Test GPU")
    headers = {"X-API-Key": TEST_API_KEY}
    with TestClient(app) as client:
        response = client.post(
            "/v1/training/jobs",
            headers=headers,
            files={"file": ("dataset.zip", dataset_zip(traversal=True), "application/zip")},
        )
        assert response.status_code == 202
        job = wait_for_terminal_status(client, response.json()["job_id"])
        assert job["status"] == "failed"
        assert job["error"] == "ValueError"
    assert not (tmp_path.parent / "outside.txt").exists()