from __future__ import annotations

from typing import Any


ML_WEIGHT = 0.40
VISUAL_WEIGHT = 0.60


def get_fraud_decision(
    score_final: float,
) -> str:
    """
    Convertit le score final en décision métier.
    """

    if score_final < 0.30:
        return "APPROUVE"

    if score_final <= 0.60:
        return "SURVEILLANCE"

    return "BLOQUE"


def fuse_scores(
        
    score_ml: float,
    score_visuel: float | None = None,
    has_image: bool = False,
    visual_reliability: float = 1.0,
) -> dict[str, Any]:
    
    """
    Fusionne le score tabulaire et le score visuel.

    Règles :
    - Sans image, le score final est égal au score ML.
    - Avec image, le poids visuel dépend de sa fiabilité.
    - Une image cohérente ne peut pas réduire le risque déjà
      détecté par Isolation Forest.
    - Le visuel peut uniquement maintenir ou augmenter le risque.
    """

    print("FUSION FILE ACTIVE :", __file__)
    print(
        "VISUAL RELIABILITY RECEIVED :",
        visual_reliability,
    )

    # Sécurisation du score ML
    score_ml = max(
        0.0,
        min(float(score_ml), 1.0),
    )

    # Cas sans image
    if not has_image or score_visuel is None:
        score_final = score_ml

        return {
            "score_final": round(score_final, 4),
            "weighted_score": round(score_final, 4),
            "score_ml_floor_applied": False,
            "decision": get_fraud_decision(
                score_final
            ),
            "fusion_mode": "ml_only",
            "weights": {
                "ml": 1.0,
                "visual": 0.0,
            },
            "visual_reliability": 0.0,
            "thresholds": {
                "approved": "< 0.30",
                "surveillance": "0.30 - 0.60",
                "blocked": "> 0.60",
            },
        }

    # Sécurisation du score visuel
    score_visuel = max(
        0.0,
        min(float(score_visuel), 1.0),
    )

    # Sécurisation de la fiabilité visuelle
    visual_reliability = max(
        0.0,
        min(float(visual_reliability), 1.0),
    )

    # Le poids visuel est réduit si la détection est peu fiable.
    adjusted_visual_weight = (
        VISUAL_WEIGHT * visual_reliability
    )

    # Le poids ML absorbe automatiquement la différence.
    adjusted_ml_weight = (
        1.0 - adjusted_visual_weight
    )

    # Fusion pondérée classique
    weighted_score = (
        score_ml * adjusted_ml_weight
        + score_visuel * adjusted_visual_weight
    )

    # Sécurité métier :
    # une image cohérente ou un score visuel faible
    # ne doit pas annuler une anomalie tabulaire.
    score_final = max(
        score_ml,
        weighted_score,
    )

    # Sécurisation finale entre 0 et 1
    score_final = max(
        0.0,
        min(score_final, 1.0),
    )

    score_ml_floor_applied = (
        score_ml > weighted_score
    )

    return {
        "score_final": round(
            float(score_final),
            4,
        ),
        "weighted_score": round(
            float(weighted_score),
            4,
        ),
        "score_ml_floor_applied": (
            score_ml_floor_applied
        ),
        "decision": get_fraud_decision(
            score_final
        ),
        "fusion_mode": "ml_visual",
        "weights": {
            "ml": round(
                adjusted_ml_weight,
                4,
            ),
            "visual": round(
                adjusted_visual_weight,
                4,
            ),
        },
        "visual_reliability": round(
            visual_reliability,
            4,
        ),
        "scores": {
            "ml": round(
                score_ml,
                4,
            ),
            "visual": round(
                score_visuel,
                4,
            ),
        },
        "thresholds": {
            "approved": "< 0.30",
            "surveillance": "0.30 - 0.60",
            "blocked": "> 0.60",
        },
    }