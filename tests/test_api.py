from __future__ import annotations

from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_mlops_model_status() -> None:
    response = client.get(
        "/api/mlops/model-status"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["success"] is True
    assert payload["alias"] == "champion"
    assert int(payload["version"]) >= 1