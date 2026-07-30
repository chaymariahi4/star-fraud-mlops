from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn

from StarGenAI.mlops.mlflow_config import DATABASE_URI


MODEL_NAME = "STAR_Fraud_IsolationForest_Model"
MODEL_ALIAS = "champion"

STAR_DIR = Path(__file__).resolve().parents[1]

SCALER_PATH = (
    STAR_DIR
    / "models"
    / "scaler.joblib"
)


@lru_cache(maxsize=1)
def load_fraud_model() -> Any:
    """
    Charge le modèle Isolation Forest désigné par
    l'alias @champion dans le registre MLflow.

    Le cache évite de recharger le modèle à chaque requête.
    """

    mlflow.set_tracking_uri(DATABASE_URI)

    model_uri = (
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    try:
        model = mlflow.sklearn.load_model(
            model_uri=model_uri
        )
    except Exception as exc:
        raise RuntimeError(
            "Impossible de charger le modèle MLflow "
            f"depuis {model_uri}."
        ) from exc

    print("=" * 70)
    print("Modèle de fraude chargé depuis MLflow")
    print(f"URI : {model_uri}")
    print("=" * 70)

    return model


@lru_cache(maxsize=1)
def load_fraud_scaler() -> Any:
    """
    Charge le scaler utilisé durant l'entraînement.
    """

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler introuvable : {SCALER_PATH}"
        )

    scaler = joblib.load(
        SCALER_PATH
    )

    print(
        f"Scaler chargé : {SCALER_PATH}"
    )

    return scaler


def clear_model_cache() -> None:
    """
    Vide le cache pour forcer le rechargement du modèle.

    Utile après le déplacement de l'alias @champion.
    """

    load_fraud_model.cache_clear()
    load_fraud_scaler.cache_clear()