from __future__ import annotations

from pathlib import Path

import mlflow


STAR_DIR = Path(__file__).resolve().parents[1]

MLOPS_DIR = STAR_DIR / "mlops_storage"
MLOPS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARTIFACTS_DIR = MLOPS_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DATABASE_PATH = MLOPS_DIR / "mlflow.db"

# SQLite utilise des slashs, même sous Windows.
DATABASE_URI = (
    f"sqlite:///{DATABASE_PATH.resolve().as_posix()}"
)

ARTIFACTS_URI = ARTIFACTS_DIR.resolve().as_uri()


FRAUD_EXPERIMENT_NAME = (
    "STAR_Fraud_IsolationForest"
)

YOLO_EXPERIMENT_NAME = (
    "STAR_Fraud_YOLOv8"
)

CROSS_SELL_EXPERIMENT_NAME = (
    "STAR_CrossSell"
)


def configure_mlflow(
    experiment_name: str,
) -> str:
    """
    Configure MLflow avec SQLite pour les métadonnées
    et un dossier local pour les artefacts.
    """

    mlflow.set_tracking_uri(
        DATABASE_URI
    )

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=ARTIFACTS_URI,
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(
        experiment_name
    )

    print(
        "MLflow tracking URI :",
        mlflow.get_tracking_uri(),
    )

    print(
        "MLflow artifact location :",
        ARTIFACTS_URI,
    )

    return str(experiment_id)