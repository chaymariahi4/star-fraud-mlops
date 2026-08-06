from __future__ import annotations

import argparse

import mlflow
from mlflow import MlflowClient

from StarGenAI.mlops.mlflow_config import DATABASE_URI
from StarGenAI.services.mlflow_model_loader import MODEL_NAME


def rollback(target_version: str) -> None:
    mlflow.set_tracking_uri(DATABASE_URI)

    client = MlflowClient()

    target = client.get_model_version(
        name=MODEL_NAME,
        version=target_version,
    )

    current = None

    try:
        current = client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias="champion",
        )
    except Exception:
        pass

    if current is not None:
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=current.version,
            key="deployment_status",
            value="archived",
        )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=target.version,
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=target.version,
        key="validation_status",
        value="approved",
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=target.version,
        key="deployment_status",
        value="production",
    )

    print(
        f"Rollback terminé : version "
        f"{target.version} devenue champion."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replace le champion MLflow sur "
            "une version existante."
        )
    )

    parser.add_argument(
        "--version",
        required=True,
        help="Version MLflow à restaurer.",
    )

    arguments = parser.parse_args()
    rollback(arguments.version)


if __name__ == "__main__":
    main()