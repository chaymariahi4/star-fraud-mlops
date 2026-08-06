from __future__ import annotations

import mlflow
from mlflow import MlflowClient

from StarGenAI.mlops.mlflow_config import DATABASE_URI


MODEL_NAME = "STAR_Fraud_IsolationForest_Model"


def main() -> None:
    mlflow.set_tracking_uri(DATABASE_URI)

    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias="champion",
    )

    print(f"Champion actif : version {champion.version}")
    print(f"Run MLflow : {champion.run_id}")


if __name__ == "__main__":
    main()