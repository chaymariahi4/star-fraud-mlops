from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Final

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow import MlflowClient
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from StarGenAI.mlops.mlflow_config import DATABASE_URI


MODEL_NAME: Final[str] = (
    "STAR_Fraud_IsolationForest_Model"
)

EXPERIMENT_NAME: Final[str] = (
    "STAR_Fraud_IsolationForest_Experiments"
)

BASE_DIR: Final[Path] = (
    Path(__file__).resolve().parents[1]
)

DATA_PATH: Final[Path] = (
    BASE_DIR
    / "Data"
    / "processed"
    / "features_finales.csv"
)

MODELS_DIR: Final[Path] = (
    BASE_DIR
    / "models"
)

MODEL_OUTPUT_PATH: Final[Path] = (
    MODELS_DIR
    / "isolation_forest.joblib"
)

SCALER_OUTPUT_PATH: Final[Path] = (
    MODELS_DIR
    / "scaler.joblib"
)

METADATA_OUTPUT_PATH: Final[Path] = (
    MODELS_DIR
    / "model_metadata.json"
)


FEATURES: Final[list[str]] = [
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


def get_git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_dvc_dataset_hash() -> str:
    dvc_file = DATA_PATH.with_suffix(
        DATA_PATH.suffix + ".dvc"
    )

    if not dvc_file.exists():
        return "not_versioned"

    for line in dvc_file.read_text(
        encoding="utf-8"
    ).splitlines():
        stripped_line = line.strip()

        if stripped_line.startswith("md5:"):
            return stripped_line.split(
                ":",
                maxsplit=1,
            )[1].strip()

    return "unknown"


def load_training_data() -> pd.DataFrame:
    """
    Charge et prépare les données utilisées
    pour entraîner Isolation Forest.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset d'entraînement introuvable : "
            f"{DATA_PATH}"
        )

    dataframe = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Les colonnes suivantes sont absentes "
            "du dataset : "
            + ", ".join(missing_features)
        )

    training_data = dataframe[
        FEATURES
    ].copy()

    training_data = training_data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    training_data = training_data.apply(
        pd.to_numeric,
        errors="coerce",
    )

    training_data = training_data.fillna(0)

    if training_data.empty:
        raise ValueError(
            "Le dataset d'entraînement est vide."
        )

    return training_data


def calculate_silhouette_score(
    scaled_data: np.ndarray,
    predictions: np.ndarray,
) -> float | None:
    """
    Calcule le score de silhouette lorsque
    le modèle produit deux groupes différents.
    """

    unique_labels = np.unique(
        predictions
    )

    if len(unique_labels) < 2:
        return None

    normal_count = int(
        np.sum(predictions == 1)
    )

    anomaly_count = int(
        np.sum(predictions == -1)
    )

    if normal_count < 2 or anomaly_count < 2:
        return None

    try:
        return float(
            silhouette_score(
                scaled_data,
                predictions,
            )
        )
    except ValueError:
        return None


def save_local_artifacts(
    model: IsolationForest,
    scaler: StandardScaler,
    metadata: dict,
) -> None:
    """
    Sauvegarde également les fichiers localement.

    Ces copies restent utiles pour les tests,
    la comparaison et un éventuel mécanisme
    de secours.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_OUTPUT_PATH,
    )

    joblib.dump(
        scaler,
        SCALER_OUTPUT_PATH,
    )

    with METADATA_OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            ensure_ascii=False,
            indent=2,
        )


def find_registered_version(
    client: MlflowClient,
    run_id: str,
) -> str:
    """
    Recherche la version créée à partir
    du run MLflow courant.
    """

    model_versions = (
        client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )
    )

    matching_versions = [
        model_version
        for model_version in model_versions
        if model_version.run_id == run_id
    ]

    if not matching_versions:
        raise RuntimeError(
            "La version du modèle enregistrée "
            "dans MLflow n'a pas été retrouvée."
        )

    latest_version = max(
        matching_versions,
        key=lambda item: int(item.version),
    )

    return str(
        latest_version.version
    )


def assign_candidate_alias(
    version: str,
) -> None:
    """
    Attribue l'alias candidate à la nouvelle
    version enregistrée.
    """

    client = MlflowClient()

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="candidate",
        version=version,
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="validation_status",
        value="pending",
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="deployment_status",
        value="not_deployed",
    )


def train_and_register_model() -> str:
    """
    Entraîne Isolation Forest, enregistre
    les paramètres, les métriques et les
    artefacts dans MLflow, puis crée une
    nouvelle version candidate.
    """

    mlflow.set_tracking_uri(
        DATABASE_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    training_data = (
        load_training_data()
    )

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(
        training_data
    )

    parameters = {
        "n_estimators": 200,
        "contamination": 0.05,
        "random_state": 42,
        "n_jobs": -1,
    }

    model = IsolationForest(
        **parameters
    )

    predictions = model.fit_predict(
        scaled_data
    )

    raw_scores = model.decision_function(
        scaled_data
    )

    suspect_count = int(
        np.sum(predictions == -1)
    )

    normal_count = int(
        np.sum(predictions == 1)
    )

    suspect_rate = float(
        suspect_count
        / len(predictions)
        * 100
    )

    silhouette = (
        calculate_silhouette_score(
            scaled_data=scaled_data,
            predictions=predictions,
        )
    )

    metadata = {
        "features": FEATURES,
        "n_estimators": (
            parameters["n_estimators"]
        ),
        "contamination": (
            parameters["contamination"]
        ),
        "random_state": (
            parameters["random_state"]
        ),
        "n_samples": int(
            len(training_data)
        ),
        "n_features": int(
            len(FEATURES)
        ),
        "score_min": float(
            np.min(raw_scores)
        ),
        "score_max": float(
            np.max(raw_scores)
        ),
        "nb_suspects": suspect_count,
        "nb_normaux": normal_count,
        "pct_suspects": suspect_rate,
        "silhouette": silhouette,
    }

    save_local_artifacts(
        model=model,
        scaler=scaler,
        metadata=metadata,
    )

    with mlflow.start_run(
        run_name=(
            "fraud_isolation_forest_training"
        )
    ) as run:

        mlflow.log_params(
            parameters
        )

        mlflow.log_param(
            "n_samples",
            len(training_data),
        )

        mlflow.log_param(
            "n_features",
            len(FEATURES),
        )

        mlflow.log_param(
            "dataset_path",
            str(DATA_PATH),
        )

        mlflow.log_param(
            "dataset_file",
            DATA_PATH.name,
        )

        mlflow.log_param(
            "git_revision",
            get_git_revision(),
        )

        mlflow.log_param(
            "dataset_dvc_hash",
            get_dvc_dataset_hash(),
        )

        mlflow.set_tag(
            "dataset_managed_by",
            "DVC",
        )

        mlflow.log_metric(
            "suspect_count",
            suspect_count,
        )

        mlflow.log_metric(
            "normal_count",
            normal_count,
        )

        mlflow.log_metric(
            "suspect_rate_percent",
            suspect_rate,
        )

        mlflow.log_metric(
            "anomaly_score_min",
            float(np.min(raw_scores)),
        )

        mlflow.log_metric(
            "anomaly_score_max",
            float(np.max(raw_scores)),
        )

        mlflow.log_metric(
            "anomaly_score_mean",
            float(np.mean(raw_scores)),
        )

        if silhouette is not None:
            mlflow.log_metric(
                "silhouette_score",
                silhouette,
            )

        mlflow.log_artifact(
            str(METADATA_OUTPUT_PATH),
            artifact_path="metadata",
        )

        mlflow.log_artifact(
            str(SCALER_OUTPUT_PATH),
            artifact_path="preprocessing",
        )

        mlflow.log_artifact(
            str(MODEL_OUTPUT_PATH),
            artifact_path="local_backup",
        )

        model_info = (
            mlflow.sklearn.log_model(
                sk_model=model,
                name="fraud_model",
                registered_model_name=MODEL_NAME,
                input_example=scaled_data[:5]
            )
        )

        run_id = run.info.run_id

    client = MlflowClient()

    version = find_registered_version(
        client=client,
        run_id=run_id,
    )

    assign_candidate_alias(
        version=version
    )

    print("=" * 70)
    print(
        "Entraînement et enregistrement terminés"
    )
    print(f"Run MLflow       : {run_id}")
    print(f"Modèle           : {MODEL_NAME}")
    print(f"Version créée    : {version}")
    print("Alias            : @candidate")
    print(
        f"Nombre de lignes : {len(training_data)}"
    )
    print(
        f"Cas normaux      : {normal_count}"
    )
    print(
        f"Cas suspects     : {suspect_count}"
    )
    print(
        f"Taux suspect     : {suspect_rate:.2f} %"
    )
    print(
        f"Score min        : {np.min(raw_scores):.6f}"
    )
    print(
        f"Score max        : {np.max(raw_scores):.6f}"
    )

    if silhouette is not None:
        print(
            f"Silhouette       : {silhouette:.4f}"
        )
    else:
        print(
            "Silhouette       : non calculable"
        )

    print(
        f"URI du modèle    : {model_info.model_uri}"
    )
    print("=" * 70)

    return version


if __name__ == "__main__":
    train_and_register_model()