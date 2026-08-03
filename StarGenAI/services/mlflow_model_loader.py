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


def get_fraud_model_uri() -> str:
    """
    Retourne l'URI MLflow du modèle de fraude actif.
    """

    return f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


@lru_cache(maxsize=1)
def load_fraud_model() -> Any:
    """
    Charge le modèle Isolation Forest désigné par
    l'alias @champion dans le registre MLflow.

    Le modèle n'est chargé qu'au premier appel.
    Les appels suivants utilisent le cache.
    """

    mlflow.set_tracking_uri(DATABASE_URI)

    model_uri = get_fraud_model_uri()

    try:
        model = mlflow.sklearn.load_model(
            model_uri=model_uri,
        )

    except Exception as exc:
        raise RuntimeError(
            "Impossible de charger le modèle MLflow "
            f"depuis {model_uri}. "
            "Vérifiez la base MLflow, les artefacts "
            "et l'alias @champion."
        ) from exc

    print("=" * 70)
    print("Modèle de fraude chargé depuis MLflow")
    print(f"URI : {model_uri}")
    print("=" * 70)

    return model


@lru_cache(maxsize=1)
def load_fraud_scaler() -> Any:
    """
    Charge le scaler utilisé pendant l'entraînement.

    Le scaler n'est chargé qu'au premier appel.
    """

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler introuvable : {SCALER_PATH}"
        )

    try:
        scaler = joblib.load(
            SCALER_PATH
        )

    except Exception as exc:
        raise RuntimeError(
            "Impossible de charger le scaler : "
            f"{SCALER_PATH}"
        ) from exc

    print(
        f"Scaler chargé : {SCALER_PATH}"
    )

    return scaler


def clear_model_cache() -> None:
    """
    Vide les caches du modèle et du scaler.

    Cette fonction est utile après le déplacement
    de l'alias @champion vers une nouvelle version.
    """

    load_fraud_model.cache_clear()
    load_fraud_scaler.cache_clear()

    print(
        "Caches du modèle de fraude et du scaler vidés."
    )