from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "StarGenAI"
    / "Data"
    / "processed"
    / "features_finales.csv"
)

EXPECTED_FEATURES = [
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


def test_fraud_dataset_exists() -> None:
    assert DATA_PATH.exists(), (
        f"Dataset introuvable : {DATA_PATH}"
    )


def test_fraud_dataset_contains_expected_features() -> None:
    dataframe = pd.read_csv(
        DATA_PATH,
        nrows=5,
        low_memory=False,
    )

    missing_features = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in dataframe.columns
    ]

    assert not missing_features, (
        "Features absentes du dataset : "
        + ", ".join(missing_features)
    )