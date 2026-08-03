from __future__ import annotations

import json
from pathlib import Path

import joblib

import pytest


pytestmark = pytest.mark.integration

BASE_DIR = Path(__file__).resolve().parents[1]

MODELS_DIR = BASE_DIR / "StarGenAI" / "Models"

MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def test_local_model_artifacts_exist() -> None:
    assert MODEL_PATH.exists()
    assert SCALER_PATH.exists()
    assert METADATA_PATH.exists()


def test_model_and_scaler_can_be_loaded() -> None:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    assert hasattr(model, "predict")
    assert hasattr(model, "decision_function")
    assert hasattr(scaler, "transform")


def test_metadata_contains_16_features() -> None:
    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    features = metadata.get("features", [])

    assert len(features) == 16