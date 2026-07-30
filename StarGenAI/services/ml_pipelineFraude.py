# api/ml_pipeline.py

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


isolation_model = load_fraud_model()
scaler = load_fraud_scaler()


def prepare_ml_features(
    sinistre_data: dict[str, Any]
) -> pd.DataFrame:
    row = {}

    for feature in MODEL_FEATURES:
        value = sinistre_data.get(feature, 0)

        if value is None:
            value = 0

        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0

        row[feature] = value

    return pd.DataFrame(
        [row],
        columns=MODEL_FEATURES,
    )


def compute_ml_score(
    sinistre_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Transforme le score Isolation Forest en score de risque 0-1.
    """

    X = prepare_ml_features(sinistre_data)
    X_scaled = scaler.transform(X)

    decision_value = float(
        isolation_model.decision_function(X_scaled)[0]
    )

    prediction = int(
        isolation_model.predict(X_scaled)[0]
    )

    # Plus decision_value est négatif, plus l'observation est atypique.
    # Transformation monotone vers 0-1.
    score_ml = 1.0 / (
        1.0 + np.exp(8.0 * decision_value)
    )

    explanation = explain_isolation_forest(
        X=X,
        model=isolation_model,
        scaler=scaler,
        reference_path=REFERENCE_PATH,
        max_factors=5,
        min_factors=3,
    )

    return {
        "score_ml": round(float(score_ml), 4),
        "raw_decision_function": round(
            decision_value,
            6,
        ),
        "is_anomaly": prediction == -1,
        "prediction_if": prediction,
        "features_used": MODEL_FEATURES,
        "explanation": explanation,
    }
