from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from StarGenAI.services.ml_explainer import (
    explain_isolation_forest,
)
from StarGenAI.services.mlflow_model_loader import (
    load_fraud_model,
    load_fraud_scaler,
)


STAR_DIR = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    STAR_DIR
    / "models"
    / "model_metadata.json"
)

REFERENCE_PATH = (
    STAR_DIR
    / "models"
    / "feature_reference.json"
)


MODEL_FEATURES = [
    "delai_declaration",
    "nb_sinistres_assure",
    "nb_sinistres_vehicule",
    "nb_sinistres_meme_jour",
    "nb_sinistres_expert",
    "ratio_montant_moyenne_annee",
    "montant_zero",
    "conducteur_different",
    "usage_risque",
    "type_sinistre_encode",
    "est_contrat_collectif",
    "age",
    "ratio_prime_sinistre",
    "delai_souscription_sinistre",
    "PRIME",
    "annee_sinistre",
]


@lru_cache(maxsize=1)
def get_model() -> Any:
    """
    Retourne le modèle MLflow actif.

    Aucun chargement n'est effectué lors
    de l'import de ce module.
    """

    return load_fraud_model()


@lru_cache(maxsize=1)
def get_scaler() -> Any:
    """
    Retourne le scaler du modèle.

    Aucun chargement n'est effectué lors
    de l'import de ce module.
    """

    return load_fraud_scaler()


def clear_ml_pipeline_cache() -> None:
    """
    Vide les caches locaux du pipeline tabulaire.
    """

    get_model.cache_clear()
    get_scaler.cache_clear()


def prepare_ml_features(
    sinistre_data: dict[str, Any],
) -> pd.DataFrame:
    """
    Prépare les 16 variables attendues par
    Isolation Forest dans le bon ordre.
    """

    row: dict[str, float] = {}

    for feature in MODEL_FEATURES:
        value = sinistre_data.get(
            feature,
            0,
        )

        if value is None:
            value = 0

        try:
            numeric_value = float(value)

        except (TypeError, ValueError):
            numeric_value = 0.0

        if not np.isfinite(numeric_value):
            numeric_value = 0.0

        row[feature] = numeric_value

    return pd.DataFrame(
        [row],
        columns=MODEL_FEATURES,
    )


def compute_ml_score(
    sinistre_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Calcule le score tabulaire de risque avec
    Isolation Forest.

    Le modèle et le scaler sont chargés uniquement
    lors du premier appel à cette fonction.
    """

    model = get_model()
    scaler = get_scaler()

    features_dataframe = prepare_ml_features(
        sinistre_data
    )

    try:
        scaled_features = scaler.transform(
            features_dataframe
        )

    except Exception as exc:
        raise RuntimeError(
            "Erreur lors de la standardisation "
            "des variables du sinistre."
        ) from exc

    try:
        decision_value = float(
            model.decision_function(
                scaled_features
            )[0]
        )

        prediction = int(
            model.predict(
                scaled_features
            )[0]
        )

    except Exception as exc:
        raise RuntimeError(
            "Erreur lors de la prédiction "
            "Isolation Forest."
        ) from exc

    # Plus la décision est négative,
    # plus le dossier est statistiquement atypique.
    score_ml = 1.0 / (
        1.0
        + np.exp(
            8.0 * decision_value
        )
    )

    try:
        explanation = explain_isolation_forest(
            X=features_dataframe,
            model=model,
            scaler=scaler,
            reference_path=REFERENCE_PATH,
            max_factors=5,
            min_factors=3,
        )

    except Exception as exc:
        # L'explication ne doit pas bloquer
        # la prédiction principale.
        explanation = {
            "success": False,
            "message": (
                "L'explication détaillée du score "
                "ML n'a pas pu être générée."
            ),
            "error": str(exc),
            "factors": [],
        }

    return {
        "score_ml": round(
            float(score_ml),
            4,
        ),
        "raw_decision_function": round(
            decision_value,
            6,
        ),
        "is_anomaly": prediction == -1,
        "prediction_if": prediction,
        "features_used": MODEL_FEATURES,
        "explanation": explanation,
        "model_source": "mlflow_registry",
        "model_alias": "champion",
    }