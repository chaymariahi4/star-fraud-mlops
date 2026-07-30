from __future__ import annotations

import json
from pathlib import Path


STAR_DIR = Path(__file__).resolve().parents[1]
METADATA_PATH = STAR_DIR / "models" / "model_metadata.json"


def validate_model() -> bool:
    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    errors: list[str] = []

    silhouette = float(
        metadata.get("silhouette", 0.0)
    )

    suspect_rate = float(
        metadata.get("pct_suspects", 0.0)
    )

    n_features = int(
        metadata.get("n_features", 0)
    )

    if silhouette < 0.40:
        errors.append(
            f"Silhouette insuffisante : {silhouette}"
        )

    if not 3.0 <= suspect_rate <= 8.0:
        errors.append(
            f"Taux de suspects inattendu : {suspect_rate}%"
        )

    if n_features != 16:
        errors.append(
            f"Nombre de features incorrect : {n_features}"
        )

    if errors:
        print("Validation refusée")

        for error in errors:
            print("-", error)

        return False

    print("Validation réussie")
    print(f"Silhouette : {silhouette}")
    print(f"Taux suspects : {suspect_rate}%")
    print(f"Features : {n_features}")

    return True


if __name__ == "__main__":
    success = validate_model()
    raise SystemExit(0 if success else 1)