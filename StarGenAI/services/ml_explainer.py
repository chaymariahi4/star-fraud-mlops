from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURE_LABELS = {
    "delai_declaration": "Délai de déclaration",
    "nb_sinistres_assure": "Historique de sinistres de l’assuré",
    "nb_sinistres_vehicule": "Historique de sinistres du véhicule",
    "nb_sinistres_meme_jour": "Sinistres déclarés le même jour",
    "nb_sinistres_expert": "Nombre de dossiers associés à l’expert",
    "ratio_montant_moyenne_annee": (
        "Montant par rapport à la moyenne annuelle"
    ),
    "montant_zero": "Montant déclaré nul",
    "conducteur_different": (
        "Conducteur différent de l’assuré habituel"
    ),
    "usage_risque": "Usage du véhicule considéré à risque",
    "type_sinistre_encode": "Type de sinistre inhabituel",
    "est_contrat_collectif": "Contrat collectif",
    "age": "Âge de l’assuré",
    "ratio_prime_sinistre": (
        "Rapport entre le montant réclamé et la prime"
    ),
    "delai_souscription_sinistre": (
        "Délai entre la souscription et le sinistre"
    ),
    "PRIME": "Prime annuelle",
    "annee_sinistre": "Année du sinistre",
}


BINARY_FEATURE_MESSAGES = {
    "conducteur_different": {
        1: (
            "Le conducteur impliqué est différent "
            "de l’assuré habituel."
        ),
        0: (
            "Le conducteur impliqué correspond "
            "à l’assuré habituel."
        ),
    },
    "usage_risque": {
        1: (
            "Le véhicule est associé à un usage "
            "considéré à risque."
        ),
        0: "Aucun usage particulier à risque n’est signalé.",
    },
    "est_contrat_collectif": {
        1: "Le dossier concerne un contrat collectif.",
        0: "Le dossier ne concerne pas un contrat collectif.",
    },
    "montant_zero": {
        1: "Le dossier présente un montant déclaré nul.",
        0: "Le dossier comporte un montant déclaré positif.",
    },
}


EXCLUDED_EXPLANATION_FEATURES = {
    "nb_sinistres_expert",
    "type_sinistre_encode",
    "annee_sinistre",
    "delai_souscription_sinistre",
}


def decision_to_score(
    decision_value: float,
) -> float:
    """
    Même transformation que celle utilisée dans ml_pipeline.py.
    """

    return float(
        1.0 / (
            1.0
            + np.exp(8.0 * decision_value)
        )
    )


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def compute_robust_atypicality(
    value: float,
    median: float,
    q25: float,
    q75: float,
    q05: float | None,
    q95: float | None,
) -> tuple[float, str]:
    """
    Calcule un écart robuste par rapport au profil habituel.

    Le score est fondé sur l'IQR :
    abs(value - median) / IQR.
    """

    iqr = q75 - q25

    if abs(iqr) < 1e-9:
        atypicality = (
            1.0
            if abs(value - median) > 1e-9
            else 0.0
        )
    else:
        atypicality = abs(
            value - median
        ) / abs(iqr)

    if q95 is not None and value > q95:
        position = "très élevée"

    elif q05 is not None and value < q05:
        position = "très faible"

    elif value > median:
        position = "supérieure à la valeur habituelle"

    elif value < median:
        position = "inférieure à la valeur habituelle"

    else:
        position = "proche de la valeur habituelle"

    return float(atypicality), position


def build_factor_description(
    feature: str,
    value: float,
    median: float,
    position: str,
) -> str:
    """
    Produit une phrase métier adaptée à la variable.
    """

    if feature in BINARY_FEATURE_MESSAGES:
        return BINARY_FEATURE_MESSAGES[feature].get(
            int(round(value)),
            "La valeur de cette caractéristique est inhabituelle.",
        )

    if feature == "nb_sinistres_assure":
        return (
            f"L’assuré présente {int(value)} sinistre(s), "
            f"contre environ {median:.1f} dans le profil habituel."
        )

    if feature == "nb_sinistres_vehicule":
        return (
            f"Le véhicule est associé à {int(value)} sinistre(s), "
            f"contre environ {median:.1f} habituellement."
        )

    if feature == "ratio_prime_sinistre":
        return (
            f"Le montant réclamé représente {value:.2f} fois "
            f"la prime annuelle, contre une valeur habituelle "
            f"d’environ {median:.2f}."
        )

    if feature == "delai_declaration":
        return (
            f"Le sinistre a été déclaré après {value:.0f} jour(s), "
            f"contre environ {median:.0f} jour(s) habituellement."
        )

    if feature == "delai_souscription_sinistre":
        return (
            f"Le délai entre la souscription et le sinistre "
            f"est de {value:.0f} jour(s), contre environ "
            f"{median:.0f} jour(s) habituellement."
        )

    if feature == "PRIME":
        return (
            f"La prime annuelle est de {value:.2f}, "
            f"contre une valeur habituelle d’environ "
            f"{median:.2f}."
        )

    return (
        f"La valeur observée ({value:.2f}) est {position}, "
        f"par rapport à la référence habituelle "
        f"({median:.2f})."
    )


def get_strength_label(
    model_impact: float,
    atypicality: float,
) -> str:
    """
    Niveau lisible de l'influence du facteur.
    """

    combined = max(
        abs(model_impact) * 5.0,
        atypicality,
    )

    if combined >= 2.0:
        return "FORTE"

    if combined >= 1.0:
        return "MODEREE"

    return "FAIBLE"


def explain_isolation_forest(
    X: pd.DataFrame,
    model: Any,
    scaler: Any,
    reference_path: str | Path,
    max_factors: int = 5,
    min_factors: int = 3,
) -> dict[str, Any]:
    """
    Explication hybride :

    1. Impact modèle :
       remplacement d'une variable par sa médiane.

    2. Atypicalité métier :
       distance robuste à la médiane avec l'IQR.

    Cette méthode garantit une explication même lorsqu'aucun
    facteur ne domine individuellement.
    """

    reference_path = Path(reference_path)

    with reference_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        references = json.load(file)

    medians = references.get("median", {})
    q05_values = references.get("q05", {})
    q25_values = references.get("q25", {})
    q75_values = references.get("q75", {})
    q95_values = references.get("q95", {})

    original_scaled = scaler.transform(X)

    original_decision = float(
        model.decision_function(
            original_scaled
        )[0]
    )

    original_score = decision_to_score(
        original_decision
    )

    factors: list[dict[str, Any]] = []

    for feature in X.columns:
        if feature in EXCLUDED_EXPLANATION_FEATURES:
            continue

        actual_value = safe_float(
            X.iloc[0][feature]
        )

        median = safe_float(
            medians.get(feature, 0.0)
        )

        q25 = safe_float(
            q25_values.get(feature, median),
            median,
        )

        q75 = safe_float(
            q75_values.get(feature, median),
            median,
        )

        q05_raw = q05_values.get(feature)
        q95_raw = q95_values.get(feature)

        q05 = (
            safe_float(q05_raw)
            if q05_raw is not None
            else None
        )

        q95 = (
            safe_float(q95_raw)
            if q95_raw is not None
            else None
        )

        # --------------------------------------------
        # Impact du modèle
        # --------------------------------------------

        modified = X.copy()

        modified.loc[
            modified.index[0],
            feature,
        ] = median

        modified_scaled = scaler.transform(
            modified
        )

        modified_decision = float(
            model.decision_function(
                modified_scaled
            )[0]
        )

        modified_score = decision_to_score(
            modified_decision
        )

        model_impact = (
            original_score
            - modified_score
        )

        if not math.isfinite(model_impact):
            model_impact = 0.0

        # --------------------------------------------
        # Atypicalité statistique robuste
        # --------------------------------------------

        atypicality, position = (
            compute_robust_atypicality(
                value=actual_value,
                median=median,
                q25=q25,
                q75=q75,
                q05=q05,
                q95=q95,
            )
        )

        # Impact positif du modèle :
        # remplacer la variable par la médiane réduit le risque.
        risk_direction = (
            "augmente_risque"
            if model_impact > 0
            else "neutre_ou_protecteur"
        )

        # Score utilisé uniquement pour le classement
        explanation_score = (
            max(model_impact, 0.0) * 4.0
            + min(atypicality, 5.0) * 0.20
        )

        description = build_factor_description(
            feature=feature,
            value=actual_value,
            median=median,
            position=position,
        )

        factors.append({
            "feature": feature,
            "label": FEATURE_LABELS.get(
                feature,
                feature,
            ),
            "value": round(actual_value, 4),
            "reference": round(median, 4),
            "model_impact": round(
                float(model_impact),
                4,
            ),
            # Compatibilité avec l'ancien frontend
            "impact": round(
                float(model_impact),
                4,
            ),
            "atypicality": round(
                float(atypicality),
                4,
            ),
            "explanation_score": round(
                float(explanation_score),
                4,
            ),
            "direction": risk_direction,
            "position": position,
            "strength": get_strength_label(
                model_impact=model_impact,
                atypicality=atypicality,
            ),
            "business_message": description,
        })

    # Priorité aux facteurs qui augmentent réellement le score.
    positive_factors = [
        factor
        for factor in factors
        if factor["model_impact"] > 0
    ]

    positive_factors.sort(
        key=lambda item: item["explanation_score"],
        reverse=True,
    )

    # Variables atypiques de secours si les impacts sont faibles.
    atypical_factors = sorted(
        factors,
        key=lambda item: (
            item["atypicality"],
            item["explanation_score"],
        ),
        reverse=True,
    )

    selected_factors: list[dict[str, Any]] = []
    selected_features: set[str] = set()

    for factor in positive_factors:
        if len(selected_factors) >= max_factors:
            break

        selected_factors.append(factor)
        selected_features.add(factor["feature"])

    # Toujours afficher au moins 3 facteurs lorsque possible.
    for factor in atypical_factors:
        if len(selected_factors) >= min_factors:
            break

        if factor["feature"] in selected_features:
            continue

        # Évite les facteurs parfaitement normaux.
        if factor["atypicality"] <= 0:
            continue

        selected_factors.append(factor)
        selected_features.add(factor["feature"])

    # Compléter jusqu'à max_factors avec des facteurs modérés.
    for factor in atypical_factors:
        if len(selected_factors) >= max_factors:
            break

        if factor["feature"] in selected_features:
            continue

        if factor["atypicality"] < 0.25:
            continue

        selected_factors.append(factor)
        selected_features.add(factor["feature"])

    strong_factor_count = sum(
        factor["strength"] == "FORTE"
        for factor in selected_factors
    )

    if strong_factor_count > 0:
        explanation_mode = "dominant_factors"
        summary = (
            "Le score est principalement expliqué par une "
            "ou plusieurs caractéristiques fortement atypiques."
        )

    elif selected_factors:
        explanation_mode = "combined_moderate_factors"
        summary = (
            "Aucun facteur unique ne suffit à expliquer le score. "
            "Celui-ci résulte de la combinaison de plusieurs "
            "caractéristiques modérément inhabituelles."
        )

    else:
        explanation_mode = "global_profile"
        summary = (
            "Le score résulte du profil global du dossier. "
            "Aucune variable isolée ne présente un écart "
            "suffisamment important."
        )

    return {
        "method": (
            "model_impact_and_robust_atypicality"
        ),
        "score_ml": round(
            float(original_score),
            4,
        ),
        "explanation_mode": explanation_mode,
        "summary": summary,
        "risk_factors": selected_factors,
        "protective_factors": [],
        "used_for_scoring": False,
    }