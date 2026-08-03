from __future__ import annotations

from dataclasses import dataclass

import app as app_module
from fastapi.testclient import TestClient


client = TestClient(app_module.app)


@dataclass
class FakeModelVersion:
    version: str = "3"
    run_id: str = "fake-run-id"
    source: str = "fake-model-source"


class FakeMlflowClient:
    def get_model_version_by_alias(
        self,
        name: str,
        alias: str,
    ) -> FakeModelVersion:
        assert name == "STAR_Fraud_IsolationForest_Model"
        assert alias == "champion"

        return FakeModelVersion()


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_mlops_model_status(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        app_module,
        "MlflowClient",
        FakeMlflowClient,
    )

    response = client.get(
        "/api/mlops/model-status"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["success"] is True
    assert payload["alias"] == "champion"
    assert payload["version"] == 3 
    assert payload["run_id"] == "fake-run-id"