from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn

from StarGenAI.mlops.mlflow_config import (
    FRAUD_EXPERIMENT_NAME,
    STAR_DIR,
    configure_mlflow,
)


MODELS_DIR = STAR_DIR / "models"
DATA_DIR = STAR_DIR / "data" / "processed"

MODEL_PATH = (
    MODELS_DIR / "isolation_forest.joblib"
)

SCALER_PATH = (
    MODELS_DIR / "scaler.joblib"
)

METADATA_PATH = (
    MODELS_DIR / "model_metadata.json"
)

SCORED_DATA_PATH = (
    DATA_DIR / "sinistres_scored.csv"
)


def load_metadata() -> dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def log_isolation_forest_experiment() -> None:
    configure_mlflow(
        FRAUD_EXPERIMENT_NAME
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler introuvable : {SCALER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    metadata = load_metadata()

    run_name = (
        "isolation_forest_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    with mlflow.start_run(
        run_name=run_name
    ) as run:
        # ---------------------------------
        # Paramètres
        # ---------------------------------

        mlflow.log_param(
            "model_type",
            "IsolationForest",
        )

        mlflow.log_param(
            "n_estimators",
            getattr(
                model,
                "n_estimators",
                "unknown",
            ),
        )

        mlflow.log_param(
            "contamination",
            getattr(
                model,
                "contamination",
                "unknown",
            ),
        )

        mlflow.log_param(
            "max_samples",
            getattr(
                model,
                "max_samples",
                "unknown",
            ),
        )

        mlflow.log_param(
            "random_state",
            getattr(model, "random_state", "unknown"),
        )

        features = metadata.get("features", [])

        mlflow.log_param(
            "number_of_features",
            len(features),
        )

        if features:
            mlflow.log_param(
                "features",
                ",".join(features),
            )

        # ---------------------------------
        # Métriques déjà calculées
        # ---------------------------------

        metric_mapping = {
            "silhouette": "silhouette_score",
            "pct_suspects": "suspect_rate_percent",
            "nb_suspects": "suspect_count",
            "n_samples": "sample_count",
            "noyau_dur_if_lof": "if_lof_core_size",
            "score_min": "anomaly_score_min",
            "score_max": "anomaly_score_max",
        }

        for metadata_key, mlflow_key in metric_mapping.items():
            value = metadata.get(metadata_key)

            if isinstance(value, (int, float)):
                mlflow.log_metric(mlflow_key, float(value))

        # ---------------------------------
        # Modèle MLflow
        # ---------------------------------

        model_info = mlflow.sklearn.log_model(
    sk_model=model,
    name="isolation_forest",
    registered_model_name="STAR_Fraud_IsolationForest_Model",)
        print(
    "Model URI :",
    model_info.model_uri,
        )
        # ---------------------------------
        # Artefacts complémentaires
        # ---------------------------------

        mlflow.log_artifact(
            str(SCALER_PATH),
            artifact_path="preprocessing",
        )

        if METADATA_PATH.exists():
            mlflow.log_artifact(
                str(METADATA_PATH),
                artifact_path="metadata",
            )

        if SCORED_DATA_PATH.exists():
            mlflow.log_artifact(
                str(SCORED_DATA_PATH),
                artifact_path="outputs",
            )

        mlflow.set_tags({
            "project": (
                "STAR Assurances"
            ),
            "use_case": (
                "fraud_detection"
            ),
            "model_family": (
                "unsupervised"
            ),
            "deployment_target": (
                "FastAPI"
            ),
        })

        print("=" * 70)
        print("Expérience MLflow enregistrée")
        print(f"Run ID : {run.info.run_id}")
        print(
            f"Expérience : "
            f"{FRAUD_EXPERIMENT_NAME}"
        )
        print("=" * 70)


if __name__ == "__main__":
    log_isolation_forest_experiment()