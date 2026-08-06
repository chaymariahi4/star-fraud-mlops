from __future__ import annotations

import mlflow
from mlflow import MlflowClient

from StarGenAI.mlops.mlflow_config import DATABASE_URI
from StarGenAI.services.mlflow_model_loader import (
    MODEL_ALIAS,
    MODEL_NAME,
)


def check_champion() -> None:
    mlflow.set_tracking_uri(DATABASE_URI)

    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    tags = dict(champion.tags or {})

    validation_status = tags.get(
        "validation_status"
    )

    deployment_status = tags.get(
        "deployment_status"
    )

    errors: list[str] = []

    if validation_status != "approved":
        errors.append(
            "Le champion n'est pas marqué approved."
        )

    if deployment_status != "production":
        errors.append(
            "Le champion n'est pas marqué production."
        )

    if errors:
        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    print("=" * 60)
    print("Champion MLflow cohérent")
    print(f"Modèle     : {MODEL_NAME}")
    print(f"Version    : {champion.version}")
    print(f"Run ID     : {champion.run_id}")
    print(f"Validation : {validation_status}")
    print(f"Déploiement: {deployment_status}")
    print("=" * 60)


if __name__ == "__main__":
    check_champion()