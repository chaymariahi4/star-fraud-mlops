from mlflow import MlflowClient
import mlflow

from StarGenAI.mlops.mlflow_config import DATABASE_URI

MODEL_NAME = "STAR_Fraud_IsolationForest_Model"

mlflow.set_tracking_uri(DATABASE_URI)

client = MlflowClient()

for alias in ("candidate", "champion"):
    try:
        version = client.get_model_version_by_alias(
            MODEL_NAME,
            alias,
        )
        print(f"{alias}: version {version.version}")
    except Exception:
        print(f"{alias}: absent")