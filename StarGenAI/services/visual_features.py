from __future__ import annotations

from typing import Any


# Certaines classes sont clairement associées à une zone.
# Bumper et Light restent ambiguës car le dataset ne distingue
# pas avant/arrière.
FRONT_PARTS = {
    "Bonnet",
    "Windshield",
}

REAR_PARTS = {
    "Dickey",
}

LATERAL_PARTS = {
    "Door",
}

AMBIGUOUS_ZONE_MAP = {
    "Bumper": {"avant", "arriere"},
    "Light": {"avant", "arriere"},
    "Fender": {"avant", "arriere", "lateral"},
}

AMBIGUOUS_PARTS = set(AMBIGUOUS_ZONE_MAP)


DECLARED_ZONE_MAPPING = {
    "coll_av": "avant",
    "collision avant": "avant",
    "avant": "avant",
    "front": "avant",

    "coll_arr": "arriere",
    "collision arrière": "arriere",
    "collision arriere": "arriere",
    "arrière": "arriere",
    "arriere": "arriere",
    "rear": "arriere",

    "choc_lat": "lateral",
    "choc latéral": "lateral",
    "choc lateral": "lateral",
    "latéral": "lateral",
    "lateral": "lateral",
    "side": "lateral",
}


def normalize_declared_zone(value: Any) -> str:
    if value is None:
        return "inconnue"

    normalized = str(value).strip().lower()

    return DECLARED_ZONE_MAPPING.get(
        normalized,
        "inconnue",
    )


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def evaluate_damage_coherence(
    declared_zone: str,
    confirmed_zones: set[str],
    uncertain_zones: set[str],
) -> dict[str, Any]:
    """
    Compare la zone déclarée avec les zones détectées.

    Statuts :
    - COHERENT : la zone déclarée est confirmée visuellement ;
    - PARTIELLEMENT_COHERENT : la zone déclarée est présente,
      mais d'autres zones sont également détectées ;
    - INCOHERENT : une ou plusieurs zones sont confirmées,
      mais aucune ne correspond à la déclaration ;
    - POTENTIELLEMENT_INCOHERENT : seules des zones incertaines
      contredisent la déclaration ;
    - INDETERMINE : les détections ne permettent pas de conclure.
    """

    if declared_zone == "inconnue":
        return {
            "status": "INDETERMINE",
            "is_coherent": False,
            "is_partial": False,
            "is_incoherent": False,
            "message": (
                "La zone déclarée n'est pas suffisamment précise "
                "pour effectuer une comparaison."
            ),
        }

    if confirmed_zones:
        if declared_zone in confirmed_zones:
            if len(confirmed_zones) == 1:
                return {
                    "status": "COHERENT",
                    "is_coherent": True,
                    "is_partial": False,
                    "is_incoherent": False,
                    "message": (
                        f"Les dommages détectés sont cohérents avec "
                        f"la zone déclarée « {declared_zone} »."
                    ),
                }

            other_zones = sorted(
                zone
                for zone in confirmed_zones
                if zone != declared_zone
            )

            return {
                "status": "PARTIELLEMENT_COHERENT",
                "is_coherent": True,
                "is_partial": True,
                "is_incoherent": False,
                "message": (
                    f"La zone déclarée « {declared_zone} » est bien "
                    "détectée, mais des dommages sont également "
                    f"observés dans les zones : {', '.join(other_zones)}."
                ),
            }

        return {
            "status": "INCOHERENT",
            "is_coherent": False,
            "is_partial": False,
            "is_incoherent": True,
            "message": (
                f"La zone déclarée est « {declared_zone} », "
                "alors que les zones confirmées par l'analyse "
                f"visuelle sont : {', '.join(sorted(confirmed_zones))}."
            ),
        }

    if uncertain_zones:
        if declared_zone in uncertain_zones:
            return {
                "status": "INDETERMINE",
                "is_coherent": False,
                "is_partial": False,
                "is_incoherent": False,
                "message": (
                    f"La zone déclarée « {declared_zone} » est compatible "
                    "avec les dommages possibles détectés. Toutefois, "
                    "aucune pièce suffisamment fiable ne permet de "
                    "confirmer précisément cette zone. Une vérification "
                    "humaine est recommandée."
                ),
            }

        return {
            "status": "POTENTIELLEMENT_INCOHERENT",
            "is_coherent": False,
            "is_partial": False,
            "is_incoherent": False,
            "message": (
                f"La zone déclarée est « {declared_zone} », "
                "alors que les détections incertaines suggèrent "
                f"plutôt : {', '.join(sorted(uncertain_zones))}."
            ),
        }

    return {
        "status": "INDETERMINE",
        "is_coherent": False,
        "is_partial": False,
        "is_incoherent": False,
        "message": (
            "Les détections disponibles ne permettent pas de "
            "confirmer précisément la zone du dommage."
        ),
    }


def extract_visual_features(
    damaged_parts: list[str],
    damage_scores: dict[str, float],
    sinistre_data: dict[str, Any],
    uncertain_parts: list[str] | None = None,
) -> dict[str, Any]:
    """
    Transforme les sorties YOLO en features métier structurées.

    YOLO ne calcule aucun score de fraude.
    Il fournit uniquement les pièces détectées et leurs confiances.
    """

    detected_parts = set(damaged_parts)

    zone_avant = int(
        bool(detected_parts.intersection(FRONT_PARTS))
    )

    zone_arriere = int(
        bool(detected_parts.intersection(REAR_PARTS))
    )

    zone_lateral = int(
        bool(detected_parts.intersection(LATERAL_PARTS))
    )

    pieces_ambigues = sorted(
        detected_parts.intersection(AMBIGUOUS_PARTS)
    )

    detected_zones: set[str] = set()

    if zone_avant:
        detected_zones.add("avant")

    if zone_arriere:
        detected_zones.add("arriere")

    if zone_lateral:
        detected_zones.add("lateral")

    declared_value = (
        sinistre_data.get("DESCRIPTION_INCIDENT")
        or sinistre_data.get("description_incident")
        or sinistre_data.get("zone_declaree")
    )

    zone_declaree = normalize_declared_zone(
        declared_value
    )

    print("=" * 60)
    print("ZONE DECLAREE :", zone_declaree)
    print("VALEUR RECUE :", declared_value)
    print("ZONES DETECTEES :", detected_zones)
    print("PIECES DETECTEES :", damaged_parts)
    print("=" * 60)

    uncertain_parts = uncertain_parts or []
    uncertain_detected_parts = set(uncertain_parts)

    uncertain_zones: set[str] = set()

    if uncertain_detected_parts.intersection(FRONT_PARTS):
        uncertain_zones.add("avant")

    if uncertain_detected_parts.intersection(REAR_PARTS):
        uncertain_zones.add("arriere")

    if uncertain_detected_parts.intersection(LATERAL_PARTS):
        uncertain_zones.add("lateral")

    for part in uncertain_detected_parts:
        if part == "Bumper":
            uncertain_zones.update({"avant", "arriere"})

        elif part == "Light":
            uncertain_zones.update({"avant", "arriere"})

        elif part == "Fender":
            uncertain_zones.update(
                {"avant", "arriere", "lateral"}
            )

    damage_coherence = evaluate_damage_coherence(
        declared_zone=zone_declaree,
        confirmed_zones=detected_zones,
        uncertain_zones=uncertain_zones,
    )

    incoherence_zone_confirmee = int(
        damage_coherence["status"] == "INCOHERENT"
    )

    print("INCOHERENCE_ZONE_CONFIRMEE :", incoherence_zone_confirmee)

    incoherence_zone_potentielle = int(
        damage_coherence["status"]
        == "POTENTIELLEMENT_INCOHERENT"
    )

    coherence_zone = int(
        damage_coherence["status"] == "COHERENT"
    )

    coherence_zone_partielle = int(
        damage_coherence["status"]
        == "PARTIELLEMENT_COHERENT"
    )

    if incoherence_zone_confirmee:
        niveau_incoherence_zone = "CONFIRMEE"
    elif incoherence_zone_potentielle:
        niveau_incoherence_zone = "POTENTIELLE"
    else:
        niveau_incoherence_zone = "AUCUNE"

    incoherence_zone = int(
        incoherence_zone_confirmee or incoherence_zone_potentielle
    )

    score_confiance_max = (
        max(damage_scores.values())
        if damage_scores
        else 0.0
    )

    score_confiance_moyen = (
        sum(damage_scores.values()) / len(damage_scores)
        if damage_scores
        else 0.0
    )

    dommage_declare = sinistre_data.get(
        "dommage_declare",
        True,
    )

    uncertain_parts = uncertain_parts or []

    nb_detections_total = (
        len(damaged_parts)
        + len(uncertain_parts)
    )

    absence_dommage_declare = int(
        bool(dommage_declare)
        and nb_detections_total == 0
    )

    montant = safe_float(
        sinistre_data.get(
            "TOTALREGLEMENT",
            sinistre_data.get(
                "montant_reclamation",
                0,
            ),
        )
    )

    nb_pieces = len(damaged_parts)

    # Règle métier initiale.
    # À calibrer plus tard avec les experts assurance.
    montant_disproportionne = int(
        (montant >= 15000 and nb_pieces <= 1)
        or
        (montant >= 25000 and nb_pieces <= 2)
    )

    faible_confiance_globale = int(
        nb_pieces > 0
        and score_confiance_max < 0.50
    )

    couverture_visuelle_incertaine = int(
        len(uncertain_parts) >= 2
    )

    return {
        "nb_pieces_endommagees": nb_pieces,
        "score_confiance_max": round(
            float(score_confiance_max),
            4,
        ),
        "score_confiance_moyen": round(
            float(score_confiance_moyen),
            4,
        ),
        "zone_avant": zone_avant,
        "zone_arriere": zone_arriere,
        "zone_lateral": zone_lateral,
        "zone_declaree": zone_declaree,
        "zones_detectees": sorted(detected_zones),
        "zones_incertaines": sorted(uncertain_zones),
        "zone_detection_conclusive": int(bool(detected_zones)),
        "pieces_confirmees": damaged_parts,
        "pieces_ambigues": pieces_ambigues,
        "pieces_incertaines": uncertain_parts,
        "coherence_zone": coherence_zone,
        "coherence_zone_partielle": coherence_zone_partielle,
        "coherence_dommages": damage_coherence["status"],
        "message_coherence_dommages": (
            damage_coherence["message"]
        ),
        "incoherence_zone": incoherence_zone_confirmee,
        "incoherence_zone_confirmee": (
            incoherence_zone_confirmee
        ),
        "incoherence_zone_potentielle": (
            incoherence_zone_potentielle
        ),
        "niveau_incoherence_zone": (
            niveau_incoherence_zone
        ),
        "coherence_analysis": damage_coherence,
        "zones_possibles_incertaines": sorted(
            uncertain_zones
        ),
        "absence_dommage_declare": (
            absence_dommage_declare
        ),
        "montant_disproportionne": (
            montant_disproportionne
        ),
        "faible_confiance_globale": (
            faible_confiance_globale
        ),
        "montant_reclamation": round(montant, 2),
         "nb_detections_incertaines": len(uncertain_parts),
        "pieces_incertaines": uncertain_parts,
         "couverture_visuelle_incertaine": (
            couverture_visuelle_incertaine
        ),
    }