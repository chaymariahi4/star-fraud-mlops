from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

REQUIRED_FILES = [
    (
        BASE_DIR
        / "StarGenAI"
        / "Data"
        / "processed"
        / "features_finales.csv.dvc"
    ),
    (
        BASE_DIR
        / "StarGenAI"
        / "mlops"
        / "train_fraud_model.py"
    ),
    (
        BASE_DIR
        / "StarGenAI"
        / "mlops"
        / "monitor_fraud_drift.py"
    ),
    (
        BASE_DIR
        / "StarGenAI"
        / "mlops"
        / "promote_fraud_model.py"
    ),
]


def main() -> None:
    missing_files = [
        str(path)
        for path in REQUIRED_FILES
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Ressources MLOps absentes :\n"
            + "\n".join(missing_files)
        )

    print("Toutes les ressources MLOps sont présentes.")

    for path in REQUIRED_FILES:
        print(f"[OK] {path}")


if __name__ == "__main__":
    main()