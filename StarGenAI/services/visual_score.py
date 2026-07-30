from __future__ import annotations

from typing import Any


FRAUD_VISUAL_WEIGHTS = {
    "incoherence_zone_confirmee": 0.40,
    "incoherence_zone_potentielle": 0.15,
    "absence_dommage_declare": 0.35,
    "montant_disproportionne": 0.25,
}

PART_LABELS = {
    "Bonnet": "capot",
    "Bumper": "pare-chocs",
    "Dickey": "coffre",
    "Door": "porte",
    "Fender": "aile",
    "Light": "optique",
    "Windshield": "pare-brise",
}


def build_damage_summary(
    visual_features: dict[str, Any],
) -> list[str]:
    confirmed = visual_features.get(
        "pieces_confirmees",
        [],
    )

    uncertain = visual_features.get(
        "pieces_incertaines",
        [],
    )

    details: list[str] = []

    for part in confirmed:
        label = PART_LABELS.get(part, part)
        details.append(
            f"{label.capitalize()} endommagé avec une "
            "détection confirmée."
        )

    for part in uncertain[:3]:
        label = PART_LABELS.get(part, part)
        details.append(
            f"Dommage possible au niveau du {label}, "
            "à confirmer par un expert."
        )

    if not details:
        details.append(
            "Aucun dommage suffisamment fiable n'a été détecté."
        )

    return details


def get_damage_severity_level(
    score: float,
) -> dict[str, str]:
    if score < 0.20:
        return {
            "code": "LEGERE",
            "label": "Légère",
            "icon": "🟢",
        }

    if score < 0.45:
        return {
            "code": "MODEREE",
            "label": "Modérée",
            "icon": "🟡",
        }

    if score < 0.70:
        return {
            "code": "IMPORTANTE",
            "label": "Importante",
            "icon": "🟠",
        }

    return {
        "code": "CRITIQUE",
        "label": "Critique",
        "icon": "🔴",
    }


def get_visual_confidence_level(
    reliability: float,
    nb_confirmed: int,
    nb_uncertain: int,
) -> dict[str, Any]:
    """
    Transforme la fiabilité technique en niveau métier.
    """

    if (
        reliability >= 0.85
        and nb_confirmed >= 2
    ):
        level = "ELEVEE"
        label = "Élevée"
        icon = "🟢"

    elif (
        reliability >= 0.65
        and nb_confirmed >= 1
    ):
        level = "BONNE"
        label = "Bonne"
        icon = "🟢"

    elif reliability >= 0.45:
        level = "MODEREE"
        label = "Modérée"
        icon = "🟡"

    else:
        level = "FAIBLE"
        label = "Faible"
        icon = "🔴"

    reasons: list[str] = []

    if nb_confirmed == 0:
        reasons.append(
            "Aucune pièce n’a été détectée avec un niveau "
            "de confiance suffisant."
        )

    elif nb_confirmed == 1:
        reasons.append(
            "Une seule pièce a été confirmée par l’analyse visuelle."
        )

    else:
        reasons.append(
            f"{nb_confirmed} pièces ont été confirmées "
            "par l’analyse visuelle."
        )

    if nb_uncertain == 1:
        reasons.append(
            "Une détection supplémentaire reste incertaine."
        )

    elif nb_uncertain >= 2:
        reasons.append(
            f"{nb_uncertain} détections supplémentaires "
            "restent incertaines."
        )

    if reliability < 0.65:
        reasons.append(
            "Une vérification humaine est recommandée."
        )

    return {
        "code": level,
        "label": label,
        "icon": icon,
        "reasons": reasons,
    }


def get_visual_fraud_status(
    score: float,
) -> dict[str, str]:
    if score <= 0.01:
        return {
            "code": "AUCUN_INDICE",
            "label": "Aucun indice visuel de fraude détecté",
            "icon": "✓",
        }

    if score < 0.20:
        return {
            "code": "INDICE_FAIBLE",
            "label": "Indice visuel faible",
            "icon": "◐",
        }

    if score < 0.45:
        return {
            "code": "INDICE_MODERE",
            "label": "Indice visuel à vérifier",
            "icon": "⚠",
        }

    return {
        "code": "INDICE_FORT",
        "label": "Indice visuel important",
        "icon": "!",
    }


def compute_visual_score(
    visual_features: dict[str, Any],
) -> dict[str, Any]:
    """
    Calcule deux scores distincts :

    1. score_visuel_fraude
       Mesure uniquement les anomalies visuelles liées à la fraude.

    2. score_severite_dommages
       Mesure l'importance apparente des dégâts détectés.

    La sévérité n'entre pas directement dans le score de fraude.
    """

    confiance_max = float(
        visual_features.get(
            "score_confiance_max",
            0.0,
        )
    )

    confiance_moyenne = float(
        visual_features.get(
            "score_confiance_moyen",
            0.0,
        )
    )

    nb_confirmed = int(
        visual_features.get(
            "nb_pieces_endommagees",
            0,
        )
    )

    nb_uncertain = int(
        visual_features.get(
            "nb_detections_incertaines",
            0,
        )
    )

    # Une pièce incertaine compte partiellement
    # dans l'évaluation de la gravité.
    effective_parts = (
        nb_confirmed
        + 0.35 * nb_uncertain
    )

    # =========================================================
    # 1. SCORE DE SÉVÉRITÉ DES DOMMAGES
    # =========================================================

    severity_parts_component = min(
        effective_parts / 7.0,
        1.0,
    )

    # La confiance intervient une seule fois.
    severity_reliability = max(
        0.50,
        confiance_moyenne,
        confiance_max,
    )

    score_severite_dommages = min(
        severity_parts_component
        * severity_reliability,
        1.0,
    )

    severity_level = get_damage_severity_level(
        score_severite_dommages
    )

    niveau_severite = severity_level["code"]
    damage_summary = build_damage_summary(
        visual_features
    )

    # =========================================================
    # 2. SCORE VISUEL DE FRAUDE
    # =========================================================

    fraud_contributions = {
        "incoherence_zone_confirmee": (
            FRAUD_VISUAL_WEIGHTS[
                "incoherence_zone_confirmee"
            ]
            if visual_features.get(
                "incoherence_zone_confirmee",
                0,
            )
            else 0.0
        ),
        "incoherence_zone_potentielle": (
            FRAUD_VISUAL_WEIGHTS[
                "incoherence_zone_potentielle"
            ]
            if visual_features.get(
                "incoherence_zone_potentielle",
                0,
            )
            else 0.0
        ),
        "absence_dommage_declare": (
            FRAUD_VISUAL_WEIGHTS[
                "absence_dommage_declare"
            ]
            if visual_features.get(
                "absence_dommage_declare",
                0,
            )
            else 0.0
        ),
        "montant_disproportionne": (
            FRAUD_VISUAL_WEIGHTS[
                "montant_disproportionne"
            ]
            if visual_features.get(
                "montant_disproportionne",
                0,
            )
            else 0.0
        ),
    }

    score_visuel_fraude_brut = sum(
        fraud_contributions.values()
    )

    # Une faible confiance réduit la certitude du score
    # visuel de fraude.
    fraud_reliability = max(
        0.50,
        confiance_max,
    )

    visual_confidence = get_visual_confidence_level(
        reliability=fraud_reliability,
        nb_confirmed=nb_confirmed,
        nb_uncertain=nb_uncertain,
    )

    score_visuel_fraude = min(
        score_visuel_fraude_brut
        * fraud_reliability,
        1.0,
    )

    visual_fraud_status = get_visual_fraud_status(
        score_visuel_fraude
    )

    # =========================================================
    # FLAGS
    # =========================================================

    fraud_flags: list[str] = []
    quality_flags: list[str] = []

    if visual_features.get("incoherence_zone_confirmee"):
        fraud_flags.append("incoherence_zone_confirmee")

    if visual_features.get("incoherence_zone_potentielle"):
        fraud_flags.append("incoherence_zone_potentielle")

    if visual_features.get(
        "absence_dommage_declare"
    ):
        fraud_flags.append(
            "absence_dommage_visible"
        )

    if visual_features.get(
        "montant_disproportionne"
    ):
        fraud_flags.append(
            "montant_disproportionne"
        )

    if visual_features.get(
        "faible_confiance_globale"
    ):
        quality_flags.append(
            "detection_visuelle_incertaine"
        )

    if nb_uncertain >= 2:
        quality_flags.append(
            "plusieurs_detections_incertaines"
        )

    return {
        # Score utilisé dans la fusion antifraude
        "score_visuel": round(
            float(score_visuel_fraude),
            4,
        ),
        "score_visuel_fraude": round(
            float(score_visuel_fraude),
            4,
        ),
        "score_visuel_fraude_brut": round(
            float(score_visuel_fraude_brut),
            4,
        ),

        # Score séparé, uniquement informatif
        "score_severite_dommages": round(
            float(score_severite_dommages),
            4,
        ),
        "niveau_severite": niveau_severite,
        "libelle_severite": severity_level["label"],
        "icone_severite": severity_level["icon"],

        "fiabilite_visuelle": round(
            float(fraud_reliability),
            4,
        ),
        "confiance_analyse_visuelle": {
            "code": visual_confidence["code"],
            "label": visual_confidence["label"],
            "icon": visual_confidence["icon"],
            "reasons": visual_confidence["reasons"],
        },
        "statut_risque_visuel": {
            "code": visual_fraud_status["code"],
            "label": visual_fraud_status["label"],
            "icon": visual_fraud_status["icon"],
        },
        "fiabilite_severite": round(
            float(severity_reliability),
            4,
        ),

        "fraud_contributions": fraud_contributions,

        # Compatibilité temporaire avec ton frontend actuel
        "contributions": fraud_contributions,

        "fraud_flags": fraud_flags,
        "quality_flags": quality_flags,

        # Compatibilité avec le code existant
        "flags": fraud_flags + quality_flags,

        "severity_details": {
            "confirmed_parts": nb_confirmed,
            "uncertain_parts": nb_uncertain,
            "effective_parts": round(
                float(effective_parts),
                2,
            ),
            "normalized_parts": round(
                float(severity_parts_component),
                4,
            ),
        },
        "resume_dommages": damage_summary,

        "weights": FRAUD_VISUAL_WEIGHTS,
        "traceable": True,
    }