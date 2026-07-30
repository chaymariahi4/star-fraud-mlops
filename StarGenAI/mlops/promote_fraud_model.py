from __future__ import annotations

import argparse
from typing import Final

import mlflow
from mlflow import MlflowClient

from StarGenAI.mlops.mlflow_config import DATABASE_URI


MODEL_NAME: Final[str] = "STAR_Fraud_IsolationForest_Model"

ALLOWED_ALIASES: Final[set[str]] = {
    "candidate",
    "champion",
}


def promote_model(
    version: str,
    alias: str,
) -> None:
    """
    Attribue un alias MLflow à une version du modèle.

    Aliases autorisés :
    - candidate : modèle en attente de validation
    - champion : modèle validé et utilisé en production
    """

    alias = alias.strip().lower()
    version = version.strip()

    if not version:
        raise ValueError(
            "La version du modèle ne peut pas être vide."
        )

    if alias not in ALLOWED_ALIASES:
        allowed = ", ".join(
            sorted(ALLOWED_ALIASES)
        )

        raise ValueError(
            f"Alias invalide : {alias}. "
            f"Aliases autorisés : {allowed}."
        )

    mlflow.set_tracking_uri(
        DATABASE_URI
    )

    client = MlflowClient()

    # Vérifier que la version existe
    model_version = client.get_model_version(
        name=MODEL_NAME,
        version=version,
    )

    print(
        f"Modèle trouvé : {model_version.name}, "
        f"version {model_version.version}"
    )

    # Attribuer l'alias demandé
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=alias,
        version=version,
    )

    if alias == "candidate":
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

    elif alias == "champion":
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=version,
            key="validation_status",
            value="approved",
        )

        client.set_model_version_tag(
            name=MODEL_NAME,
            version=version,
            key="deployment_status",
            value="production",
        )

    print("=" * 60)
    print(
        f"Version {version} définie comme @{alias}."
    )

    if alias == "candidate":
        print(
            "Statut de validation : pending"
        )
        print(
            "Statut de déploiement : not_deployed"
        )

    if alias == "champion":
        print(
            "Statut de validation : approved"
        )
        print(
            "Statut de déploiement : production"
        )

    print("=" * 60)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attribuer un alias candidate ou champion "
            "à une version du modèle de fraude."
        )
    )

    parser.add_argument(
        "--version",
        required=True,
        help="Version MLflow du modèle, par exemple 1.",
    )

    parser.add_argument(
        "--alias",
        required=True,
        choices=sorted(ALLOWED_ALIASES),
        help="Alias à attribuer : candidate ou champion.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    promote_model(
        version=args.version,
        alias=args.alias,
    )