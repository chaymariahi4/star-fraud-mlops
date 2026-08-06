from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run

from StarGenAI.mlops.mlflow_config import DATABASE_URI


MODEL_NAME = "STAR_Fraud_IsolationForest_Model"

CANDIDATE_ALIAS = "candidate"
CHAMPION_ALIAS = "champion"

MIN_SILHOUETTE = 0.40
MIN_SUSPECT_RATE = 4.0
MAX_SUSPECT_RATE = 6.0
MAX_ALLOWED_SILHOUETTE_DROP = 0.01


@dataclass(frozen=True)
class ValidationResult:
    approved: bool
    reasons: list[str]
    candidate_version: str
    champion_version: str | None
    candidate_metrics: dict[str, float]
    champion_metrics: dict[str, float]


def configure_mlflow() -> MlflowClient:
    mlflow.set_tracking_uri(DATABASE_URI)
    return MlflowClient()


def get_version_by_alias(
    client: MlflowClient,
    alias: str,
) -> Any | None:
    try:
        return client.get_model_version_by_alias(
            name=MODEL_NAME,
            alias=alias,
        )
    except Exception:
        return None


def get_run(
    client: MlflowClient,
    run_id: str,
) -> Run:
    return client.get_run(run_id)


def extract_metrics(run: Run) -> dict[str, float]:
    metrics = run.data.metrics

    return {
        "silhouette_score": float(
            metrics.get("silhouette_score", 0.0)
        ),
        "suspect_rate_percent": float(
            metrics.get("suspect_rate_percent", 0.0)
        ),
        "suspect_count": float(
            metrics.get("suspect_count", 0.0)
        ),
        "normal_count": float(
            metrics.get("normal_count", 0.0)
        ),
    }


def validate_candidate(
    candidate_metrics: dict[str, float],
    champion_metrics: dict[str, float],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    candidate_silhouette = candidate_metrics[
        "silhouette_score"
    ]

    candidate_suspect_rate = candidate_metrics[
        "suspect_rate_percent"
    ]

    champion_silhouette = champion_metrics.get(
        "silhouette_score",
        0.0,
    )

    if candidate_silhouette < MIN_SILHOUETTE:
        reasons.append(
            "Silhouette candidate insuffisante : "
            f"{candidate_silhouette:.4f} < {MIN_SILHOUETTE:.4f}"
        )

    if not (
        MIN_SUSPECT_RATE
        <= candidate_suspect_rate
        <= MAX_SUSPECT_RATE
    ):
        reasons.append(
            "Taux de cas suspects hors plage : "
            f"{candidate_suspect_rate:.2f} %"
        )

    if (
        champion_silhouette > 0
        and candidate_silhouette
        < champion_silhouette - MAX_ALLOWED_SILHOUETTE_DROP
    ):
        reasons.append(
            "Régression de silhouette par rapport au champion : "
            f"{candidate_silhouette:.4f} contre "
            f"{champion_silhouette:.4f}"
        )

    return len(reasons) == 0, reasons


def promote_candidate(
    client: MlflowClient,
    candidate_version: str,
    previous_champion_version: str | None,
) -> None:
    # 1. Promouvoir la version candidate en champion
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=candidate_version,
    )

    # 2. Retirer l'alias candidate après promotion
    try:
        client.delete_registered_model_alias(
            name=MODEL_NAME,
            alias=CANDIDATE_ALIAS,
        )
    except Exception:
        # L'alias peut déjà être absent.
        pass

    # 3. Mettre à jour les tags de la nouvelle version
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=candidate_version,
        key="validation_status",
        value="approved",
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=candidate_version,
        key="deployment_status",
        value="production",
    )

    # 4. Archiver l'ancien champion
    if (
        previous_champion_version is not None
        and previous_champion_version != candidate_version
    ):
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=previous_champion_version,
            key="deployment_status",
            value="archived",
        )

def reject_candidate(
    client: MlflowClient,
    candidate_version: str,
    reasons: list[str],
) -> None:
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=candidate_version,
        key="validation_status",
        value="rejected",
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=candidate_version,
        key="rejection_reason",
        value=" | ".join(reasons),
    )


def validate_and_promote() -> ValidationResult:
    client = configure_mlflow()

    candidate = get_version_by_alias(
        client,
        CANDIDATE_ALIAS,
    )

    if candidate is None:
        raise RuntimeError(
            "Aucune version candidate n'est définie."
        )

    champion = get_version_by_alias(
        client,
        CHAMPION_ALIAS,
    )

    candidate_run = get_run(
        client,
        candidate.run_id,
    )

    candidate_metrics = extract_metrics(
        candidate_run
    )

    champion_metrics: dict[str, float] = {}

    if champion is not None:
        champion_run = get_run(
            client,
            champion.run_id,
        )

        champion_metrics = extract_metrics(
            champion_run
        )

    approved, reasons = validate_candidate(
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
    )

    previous_champion_version = (
        champion.version
        if champion is not None
        else None
    )

    if approved:
        promote_candidate(
            client=client,
            candidate_version=candidate.version,
            previous_champion_version=previous_champion_version,
        )

        print("=" * 70)
        print("Candidat approuvé et promu champion")
        print(f"Version : {candidate.version}")
        print(
            "Silhouette : "
            f"{candidate_metrics['silhouette_score']:.4f}"
        )
        print(
            "Taux suspect : "
            f"{candidate_metrics['suspect_rate_percent']:.2f} %"
        )
        print("=" * 70)

    else:
        reject_candidate(
            client=client,
            candidate_version=candidate.version,
            reasons=reasons,
        )

        print("=" * 70)
        print("Candidat rejeté")
        print(f"Version : {candidate.version}")

        for reason in reasons:
            print(f"- {reason}")

        print("=" * 70)

    return ValidationResult(
        approved=approved,
        reasons=reasons,
        candidate_version=str(candidate.version),
        champion_version=(
            str(previous_champion_version)
            if previous_champion_version is not None
            else None
        ),
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
    )


if __name__ == "__main__":
    validate_and_promote()