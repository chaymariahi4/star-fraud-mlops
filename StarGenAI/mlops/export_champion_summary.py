from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from StarGenAI.mlops.mlflow_config import DATABASE_URI
from StarGenAI.services.mlflow_model_loader import (
    MODEL_ALIAS,
    MODEL_NAME,
)


OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "champion_summary.json"
)


def export_champion_summary() -> None:
    mlflow.set_tracking_uri(DATABASE_URI)

    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    tags = dict(champion.tags or {})

    payload = {
        "model_name": MODEL_NAME,
        "version": str(champion.version),
        "alias": MODEL_ALIAS,
        "run_id": champion.run_id,
        "validation_status": tags.get(
            "validation_status",
            "unknown",
        ),
        "deployment_status": tags.get(
            "deployment_status",
            "unknown",
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Résumé du champion exporté : {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    export_champion_summary()