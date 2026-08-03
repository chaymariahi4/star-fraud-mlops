from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from urllib import response

import hashlib
import json
import uuid

import joblib
import mlflow
import numpy as np
import pandas as pd
import shap

from datetime import datetime
from fastapi import File, Form, UploadFile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from mlflow import MlflowClient
from pydantic import BaseModel
from typing import Any

from StarGenAI.mlops.mlflow_config import DATABASE_URI
from StarGenAI.services.mlflow_model_loader import (
    MODEL_ALIAS,
    MODEL_NAME,
    load_fraud_model,
    load_fraud_scaler,
)

# from StarGenAI.api.fraud_router import router as fraud_router
from StarGenAI.services.vision_pipeline import detect_damages
from StarGenAI.services.visual_features import extract_visual_features
from StarGenAI.services.visual_score import compute_visual_score
from StarGenAI.services.ml_pipelineFraude import compute_ml_score
from StarGenAI.services.fusion import fuse_scores
from StarGenAI.services.visual_explainer import (
    explain_with_phi35,
)




@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Chargement des ressources ML...")

    load_fraud_model()
    load_fraud_scaler()

    print("Ressources ML chargées.")

    yield

    print("Arrêt de l'application.")

app = FastAPI(
    title="STAR AI API",
    description="API cross-selling et détection de fraude",
    lifespan=lifespan,
)

BASE_DIR = Path(__file__).resolve().parent
STAR_DIR = BASE_DIR / "StarGenAI"

ANNOTATED_DIR = STAR_DIR / "data" / "annotated"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/annotated",
    StaticFiles(directory=str(ANNOTATED_DIR)),
    name="annotated",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "STAR Decision Support API",
    }


# app.include_router(fraud_router)

BASE_DIR = Path(__file__).resolve().parent
STAR_DIR = BASE_DIR / "StarGenAI"
from functools import lru_cache


DATA_PATH = (
    BASE_DIR
    / "StarGenAI"
    / "Data"
    / "processed"
    / "feature_dataset.csv"
)


@lru_cache(maxsize=1)
def get_feature_dataset() -> pd.DataFrame:
    """
    Charge le dataset Cross-Sell uniquement lorsqu'une
    fonctionnalité qui en dépend est réellement appelée.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset Cross-Sell introuvable : {DATA_PATH}"
        )

    return pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )
MODELS_PATH = STAR_DIR / "Models" / "recommendation_models.pkl"
FEATURES_PATH = STAR_DIR / "Models" / "model_features.pkl"

@lru_cache(maxsize=1)
def get_recommendation_models() -> dict[str, Any]:
    """
    Charge les modèles Cross-Sell uniquement lors
    du premier appel à l'endpoint de recommandation.
    """

    if not MODELS_PATH.exists():
        raise FileNotFoundError(
            f"Modèles Cross-Sell introuvables : {MODELS_PATH}"
        )

    try:
        loaded_models = joblib.load(MODELS_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Impossible de charger les modèles Cross-Sell : "
            f"{MODELS_PATH}"
        ) from exc

    if not isinstance(loaded_models, dict):
        raise TypeError(
            "Le fichier recommendation_models.pkl doit "
            "contenir un dictionnaire de modèles."
        )

    return loaded_models


@lru_cache(maxsize=1)
def get_model_features() -> list[str]:
    """
    Charge la liste des variables utilisées par les
    modèles Cross-Sell uniquement lors du premier appel.
    """

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Liste des features introuvable : {FEATURES_PATH}"
        )

    try:
        loaded_features = joblib.load(FEATURES_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Impossible de charger les features Cross-Sell : "
            f"{FEATURES_PATH}"
        ) from exc

    return list(loaded_features)

PRODUCT_COLUMNS = [
    "possede_auto",
    "possede_habitation",
    "possede_sante",
    "possede_vie",
    "possede_junior",
    "possede_emprunteur",
    "possede_accident_vie",
]

PRODUCT_NAMES = {
    "possede_auto": "TRIK ESSLAMA",
    "possede_habitation": "DAR ESSLAMA",
    "possede_sante": "STARCARE",
    "possede_vie": "Vie Nouvelle / 7AYYA",
    "possede_junior": "JUNIOR / Avenir Jeunesse",
    "possede_emprunteur": "Emprunteur",
    "possede_accident_vie": "Garantie Accidents de la Vie",
}

FEATURE_LABELS = {
    "age": "Âge",
    "nombre_enfants": "Nombre d’enfants",
    "revenu_estime": "Revenu estimé",
    "anciennete_client": "Ancienneté client",
    "nb_sinistres": "Nombre de sinistres",
    "montant_sinistres": "Montant total des sinistres",
    "nb_produits": "Nombre de produits détenus",
    "cout_moyen_sinistre": "Coût moyen des sinistres",
    "famille": "Profil familial",
    "senior": "Profil senior",
    "jeune": "Profil jeune",
    "multi_equipe": "Client multi-équipé",
    "client_fidele": "Fidélité client",
    "client_risque": "Niveau de risque",
    "proprietaire": "Statut propriétaire",
    "possede_vehicule": "Possession d’un véhicule",
    "score_patrimoine": "Score patrimonial",
    "score_famille": "Score familial",
    "sexe_H": "Sexe masculin",
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
    "ratio_prime_sinistre": (
        "Rapport entre le montant réclamé et la prime"
    ),
    "delai_souscription_sinistre": (
        "Délai entre la souscription et le sinistre"
    ),
    "PRIME": "Prime annuelle",
    "annee_sinistre": "Année du sinistre",
}


@lru_cache(maxsize=None)
def get_product_explainer(product_col: str) -> Any:
    """
    Crée et met en cache l'explainer SHAP d'un produit
    uniquement lorsqu'une explication est demandée.
    """

    models = get_recommendation_models()

    if product_col not in models:
        raise KeyError(
            f"Modèle Cross-Sell absent pour : {product_col}"
        )

    return shap.TreeExplainer(
        models[product_col]
    )


def decode_one_hot(client_row, prefix: str, default_value: str):
    matched_cols = [
        col for col in client_row.index
        if col.startswith(prefix) and int(client_row[col]) == 1
    ]

    if not matched_cols:
        return default_value

    return matched_cols[0].replace(prefix, "")


def get_sexe(client_row):
    return "H" if int(client_row.get("sexe_H", 0)) == 1 else "F"


def get_ville(client_row):
    return decode_one_hot(client_row, "ville_", "Ariana")


def get_situation_familiale(client_row):
    return decode_one_hot(client_row, "situation_familiale_", "Célibataire")


def get_type_logement(client_row):
    return decode_one_hot(client_row, "type_logement_", "Chez famille")


def get_type_vehicule(client_row):
    return decode_one_hot(client_row, "type_vehicule_", "Aucun")


def get_canal_souscription(client_row):
    return decode_one_hot(client_row, "canal_souscription_", "Agence")


def get_age_group(client_row):
    return decode_one_hot(client_row, "age_group_", "18-25")


def get_revenu_classe(client_row):
    return decode_one_hot(client_row, "revenu_classe_", "Faible")


def get_client_products(client_row):
    products = []

    for col in PRODUCT_COLUMNS:
        if int(client_row[col]) == 1:
            products.append(PRODUCT_NAMES[col])

    return products


def normalize_scores(scored_products):
    if not scored_products:
        return []

    raw_scores = [item["raw_score"] for item in scored_products]
    min_score = min(raw_scores)
    max_score = max(raw_scores)

    for item in scored_products:
        if max_score == min_score:
            normalized_score = 100
        else:
            normalized_score = round(
                ((item["raw_score"] - min_score) / (max_score - min_score)) * 100
            )

        item["normalized_score"] = normalized_score
        item["level"] = get_level(normalized_score)

    return scored_products


def get_level(normalized_score: int):
    if normalized_score >= 85:
        return "Très élevée"
    if normalized_score >= 65:
        return "Élevée"
    if normalized_score >= 40:
        return "Moyenne"
    return "Faible"


def format_feature_name(feature_name: str) -> str:
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]

    if feature_name.startswith("ville_"):
        return f"Ville : {feature_name.replace('ville_', '')}"

    if feature_name.startswith("situation_familiale_"):
        value = feature_name.replace("situation_familiale_", "")
        return f"Situation familiale : {value}"

    if feature_name.startswith("type_logement_"):
        value = feature_name.replace("type_logement_", "")
        return f"Type de logement : {value}"

    if feature_name.startswith("type_vehicule_"):
        value = feature_name.replace("type_vehicule_", "")
        return f"Type de véhicule : {value}"

    if feature_name.startswith("canal_souscription_"):
        value = feature_name.replace("canal_souscription_", "")
        return f"Canal de souscription : {value}"

    return feature_name.replace("_", " ").capitalize()


def format_shap_impact_label(value: float) -> str:
    formatted_value = f"{value:+.3f}".replace(".", ",")

    if value > 0:
        return f"Impact positif : {formatted_value}"

    if value < 0:
        return f"Impact négatif : {formatted_value}"

    return f"Impact neutre : {formatted_value}"


def extract_shap_values(
    model: Any,
    client_features: pd.DataFrame,
) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    shap_result = explainer.shap_values(client_features)

    if isinstance(shap_result, list):
        if len(shap_result) == 2:
            values = shap_result[1]
        else:
            values = shap_result[-1]
    else:
        values = shap_result

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[0, :, 1]

    elif values.ndim == 2:
        values = values[0]

    return values.astype(float)


def get_shap_explanation(
    model: Any,
    client_features: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, Any]:
    shap_values = extract_shap_values(
        model=model,
        client_features=client_features,
    )

    feature_names = list(client_features.columns)
    feature_values = client_features.iloc[0].to_dict()

    contributions = []

    for feature_name, shap_value in zip(
        feature_names,
        shap_values,
    ):
        numeric_value = float(shap_value)

        contributions.append(
            {
                "feature": feature_name,
                "label": format_feature_name(feature_name),
                "feature_value": feature_values.get(feature_name),
                "contribution": round(numeric_value, 4),
                "absolute_contribution": round(
                    abs(numeric_value),
                    4,
                ),
                "direction": (
                    "positive"
                    if numeric_value > 0
                    else "negative"
                    if numeric_value < 0
                    else "neutral"
                ),
                "impact_label": format_shap_impact_label(
                    numeric_value
                ),
            }
        )

    positive_factors = sorted(
        [
            item
            for item in contributions
            if item["contribution"] > 0
        ],
        key=lambda item: item["contribution"],
        reverse=True,
    )[:top_n]

    negative_factors = sorted(
        [
            item
            for item in contributions
            if item["contribution"] < 0
        ],
        key=lambda item: item["contribution"],
    )[:top_n]

    return {
        "positive_factors": positive_factors,
        "negative_factors": negative_factors,
    }


def extract_shap_factors(product_col: str, X):
    model = models[product_col]

    try:
        shap_result = get_shap_explanation(
            model=model,
            client_features=X,
            top_n=3,
        )

        return {
            "positive": shap_result.get("positive_factors", []),
            "negative": shap_result.get("negative_factors", []),
        }

    except Exception as error:
        print("Erreur SHAP:", error)
        return {
            "positive": [],
            "negative": [],
        }


def build_business_explanation(product_name, raw_score, level, shap_factors):
    return (
        "Cette recommandation correspond au profil du client et à ses produits "
        "actuellement détenus. STARCARE peut compléter sa couverture avec une "
        "protection santé adaptée."
    )


def build_mistral_payload(client_row, product_name, normalized_score, level, shap_factors):
    return {
        "role": "assistant conseiller assurance STAR",
        "instruction": (
            "Reformuler l'explication du modèle ML sans modifier le classement, "
            "sans inventer de garanties, et en utilisant uniquement les facteurs fournis."
        ),
        "client": {
            "client_id": str(client_row["client_id"]),
            "age": int(client_row["age"]),
            "sexe": get_sexe(client_row),
            "ville": get_ville(client_row),
            "situation_familiale": get_situation_familiale(client_row),
            "nombre_enfants": int(client_row["nombre_enfants"]),
            "produits_possedes": get_client_products(client_row),
        },
        "recommendation": {
            "product": product_name,
            "normalized_score": normalized_score,
            "level": level,
            "positive_factors": shap_factors.get("positive", []),
            "negative_factors": shap_factors.get("negative", []),
        },
    }


@app.get("/")
def home():
    return {
        "message": "STAR Cross-Selling API is running",
        "endpoint": "/api/recommendations/{client_id}",
    }


@app.get("/api/clients")
def get_clients():
    try:
        dataframe = get_feature_dataset()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Le service de recommandation est "
                "temporairement indisponible : "
                "dataset Cross-Sell absent."
            ),
        ) from exc

    return {
        "clients": dataframe["client_id"].astype(str).tolist()
    }


@app.get("/api/recommendations/{client_id}")
def get_recommendations(client_id: str):
    try:
        dataframe = get_feature_dataset()
        models = get_recommendation_models()
        model_features = get_model_features()

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Le service Cross-Sell est temporairement "
                "indisponible : données ou modèles absents."
            ),
        ) from exc

    except (RuntimeError, TypeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Erreur lors du chargement des ressources "
                "du moteur Cross-Sell."
            ),
        ) from exc

    client = dataframe[
        dataframe["client_id"].astype(str) == client_id
    ]

    if client.empty:
        raise HTTPException(
            status_code=404,
            detail="Client introuvable",
        )

    client_row = client.iloc[0]
    X = client[model_features]

    # suite du traitement...

    scored_products = []

    for product_col in PRODUCT_COLUMNS:
        already_has = int(client_row[product_col])

        if already_has == 0:
            model = models[product_col]
            raw_score = float(model.predict_proba(X)[0, 1])
            product_name = PRODUCT_NAMES[product_col]

            shap_factors = extract_shap_factors(product_col, X)

            scored_products.append({
                "product_col": product_col,
                "product": product_name,
                "raw_score": round(raw_score, 4),
                "shap_factors": shap_factors,
            })

    scored_products = sorted(
        scored_products,
        key=lambda item: item["raw_score"],
        reverse=True
    )[:3]

    scored_products = normalize_scores(scored_products)

    recommendations = []

    for index, item in enumerate(scored_products, start=1):
        model = models[item["product_col"]]

        shap_explanation = get_shap_explanation(
            model=model,
            client_features=X,
            top_n=4,
        )

        explanation = build_business_explanation(
            item["product"],
            item["raw_score"],
            get_raw_score_level(item["raw_score"]),
            item["shap_factors"],
        )

        mistral_payload = build_mistral_payload(
            client_row,
            item["product"],
            item["normalized_score"],
            item["level"],
            item["shap_factors"],
        )

        recommendations.append({
            "rank": index,
            "product": item["product"],
            "raw_score": item["raw_score"],
            "score": round(float(item["raw_score"]), 4),
            "score_percent": round(float(item["raw_score"]) * 100, 1),
            "priority": get_raw_score_level(item["raw_score"]),
            "level": get_raw_score_level(item["raw_score"]),
            "shap_factors": item["shap_factors"],
            "positive_factors": shap_explanation["positive_factors"],
            "negative_factors": shap_explanation["negative_factors"],
            "reasons": [
                factor["label"]
                for factor in item["shap_factors"]["positive"]
            ],
            "commercial_summary": explanation,
            "explanation": explanation,
            "mistral_payload": mistral_payload,
        })

    return {
        "client": {
            "client_id": str(client_row["client_id"]),
            "age": int(client_row["age"]),
            "sexe": get_sexe(client_row),
            "ville": get_ville(client_row),
            "situation_familiale": get_situation_familiale(client_row),
            "nombre_enfants": int(client_row["nombre_enfants"]),
            "revenu_estime": int(client_row["revenu_estime"]),
            "revenu_classe": get_revenu_classe(client_row),
            "age_group": get_age_group(client_row),
            "type_logement": get_type_logement(client_row),
            "type_vehicule": get_type_vehicule(client_row),
            "canal_souscription": get_canal_souscription(client_row),
            "anciennete_client": int(client_row["anciennete_client"]),
            "nb_sinistres": int(client_row["nb_sinistres"]),
            "montant_sinistres": float(client_row["montant_sinistres"]),
            "nb_produits": int(client_row["nb_produits"]),
            "score_patrimoine": int(client_row["score_patrimoine"]),
            "score_famille": int(client_row["score_famille"]),
            "produits_possedes": get_client_products(client_row),
        },
        "recommendations": recommendations,
    }


class ModelStatusResponse(BaseModel):
    success: bool
    model_name: str
    alias: str
    version: int
    run_id: str
    source: str


@app.get(
    "/api/mlops/model-status",
    response_model=ModelStatusResponse,
)
def get_model_status() -> ModelStatusResponse:
    mlflow.set_tracking_uri(DATABASE_URI)

    client = MlflowClient()

    model_version = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )

    return ModelStatusResponse(
        success=True,
        model_name=MODEL_NAME,
        alias=MODEL_ALIAS,
        version=int(model_version.version),
        run_id=str(model_version.run_id),
        source=str(model_version.source),
    )


def get_raw_score_level(raw_score: float):
    if raw_score >= 0.75:
        return "Très élevée"
    if raw_score >= 0.50:
        return "Élevée"
    if raw_score >= 0.25:
        return "Moyenne"
    return "Faible"

# ============================================================
# FRAUD DETECTION API
# ============================================================

FRAUD_IMAGES_DIR = STAR_DIR / "data" / "images"
FRAUD_PROCESSED_DIR = STAR_DIR / "data" / "processed"

FRAUD_CLEAN_CSV = FRAUD_PROCESSED_DIR / "sinistres_clean.csv"
FRAUD_SCORED_CSV = FRAUD_PROCESSED_DIR / "sinistres_scored.csv"
IMAGE_HASHES_PATH = FRAUD_IMAGES_DIR / "image_hashes.json"

FRAUD_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
FRAUD_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def validate_sinistre_data(
    sinistre_data: dict[str, Any],
) -> str:
    """
    Vérifie que le payload contient un numéro de sinistre.
    """

    num_sinistre = (
        sinistre_data.get("NUM_SINISTRE")
        or sinistre_data.get("num_sinistre")
    )

    if not num_sinistre:
        raise HTTPException(
            status_code=400,
            detail="NUM_SINISTRE est obligatoire.",
        )

    return str(num_sinistre)


def compute_image_md5(content: bytes) -> str:
    """
    Calcule le hash MD5 d'une image.
    """

    return hashlib.md5(content).hexdigest()


def load_image_hashes() -> dict[str, str]:
    """
    Charge l'index des images déjà reçues.
    """

    if not IMAGE_HASHES_PATH.exists():
        return {}

    try:
        with IMAGE_HASHES_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data

        return {}

    except (OSError, json.JSONDecodeError):
        return {}


def save_image_hashes(
    image_hashes: dict[str, str],
) -> None:
    """
    Sauvegarde l'index MD5 des images.
    """

    with IMAGE_HASHES_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            image_hashes,
            file,
            indent=2,
            ensure_ascii=False,
        )


async def save_fraud_image(
    image: UploadFile,
) -> dict[str, Any]:
    """
    Vérifie le type de fichier, détecte les doublons
    et sauvegarde l'image.
    """

    original_filename = image.filename or "image.jpg"
    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Extension non autorisée : {extension}. "
                f"Extensions acceptées : "
                f"{sorted(ALLOWED_IMAGE_EXTENSIONS)}"
            ),
        )

    content = await image.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Le fichier image est vide.",
        )

    image_md5 = compute_image_md5(content)
    image_hashes = load_image_hashes()

    duplicate_relative_path = image_hashes.get(image_md5)

    if duplicate_relative_path:
        duplicate_absolute_path = (
            STAR_DIR / duplicate_relative_path
        )

        if duplicate_absolute_path.exists():
            return {
                "absolute_path": duplicate_absolute_path,
                "relative_path": duplicate_relative_path,
                "md5": image_md5,
                "duplicate": True,
            }

    filename = f"{uuid.uuid4().hex}{extension}"
    absolute_path = FRAUD_IMAGES_DIR / filename

    absolute_path.write_bytes(content)

    relative_path = str(
        absolute_path.relative_to(STAR_DIR)
    ).replace("\\", "/")

    image_hashes[image_md5] = relative_path
    save_image_hashes(image_hashes)

    return {
        "absolute_path": absolute_path,
        "relative_path": relative_path,
        "md5": image_md5,
        "duplicate": False,
    }


def update_sinistre_image_path(
    num_sinistre: str,
    image_path: str,
) -> bool:
    """
    Met à jour image_path dans sinistres_clean.csv.
    """

    if not FRAUD_CLEAN_CSV.exists():
        return False

    sinistres_df = pd.read_csv(FRAUD_CLEAN_CSV)

    if "NUM_SINISTRE" not in sinistres_df.columns:
        return False

    if "image_path" not in sinistres_df.columns:
        sinistres_df["image_path"] = pd.NA
    else:
        sinistres_df["image_path"] = sinistres_df["image_path"].astype(object)

    matching_rows = (
        sinistres_df["NUM_SINISTRE"].astype(str)
        == str(num_sinistre)
    )

    if not matching_rows.any():
        return False

    sinistres_df.loc[
        matching_rows,
        "image_path",
    ] = image_path

    sinistres_df.to_csv(
        FRAUD_CLEAN_CSV,
        index=False,
    )

    return True


def save_fraud_result(
    result: dict[str, Any],
) -> None:
    """
    Ajoute le résultat dans sinistres_scored.csv.
    """

    visual_analysis = result.get("visual_analysis")

    score_visuel = None
    damaged_parts: list[str] = []

    if visual_analysis:
        score_visuel = visual_analysis["visual_score"]["score_visuel"]
        damaged_parts = visual_analysis["yolo"].get("damaged_parts", [])

    row = {
        "timestamp": result["timestamp"],
        "num_sinistre": result["num_sinistre"],
        "score_ml": result["ml_analysis"]["score_ml"],
        "score_visuel": score_visuel,
        "score_final": result["decision"]["score_final"],
        "decision": result["decision"]["decision"],
        "has_image": result["has_image"],
        "image_path": result.get("image_path"),
        "damaged_parts": json.dumps(damaged_parts, ensure_ascii=False),
    }

    row_df = pd.DataFrame([row])

    if FRAUD_SCORED_CSV.exists():
        row_df.to_csv(
            FRAUD_SCORED_CSV,
            mode="a",
            header=False,
            index=False,
        )
    else:
        row_df.to_csv(
            FRAUD_SCORED_CSV,
            index=False,
        )


# ============================================================
# Route sans image
# ============================================================
@app.post("/api/fraud/analyze")
def analyze_fraud(
    sinistre_data: dict[str, Any],
):
    """
    Analyse un sinistre avec Isolation Forest uniquement.
    """

    try:
        num_sinistre = validate_sinistre_data(sinistre_data)

        montant = float(
            sinistre_data.get("TOTALREGLEMENT", 0) or 0
        )

        prime = float(
            sinistre_data.get("PRIME", 0) or 0
        )

        sinistre_data["ratio_prime_sinistre"] = (
            montant / prime
            if prime > 0
            else 0.0
        )

        ml_analysis = compute_ml_score(sinistre_data)

        print(
                        "SCORE ML calculé (sans image) :",
                        ml_analysis.get("score_ml"),
                )

        decision = fuse_scores(
            score_ml=ml_analysis["score_ml"],
            score_visuel=None,
            has_image=False,
        )

        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "num_sinistre": num_sinistre,
            "has_image": False,
            "image_path": None,
            "ml_analysis": ml_analysis,
            "visual_analysis": None,
            "decision": decision,
        }

        save_fraud_result(response)
        return response

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Erreur pendant l'analyse du sinistre.",
                "error": str(error),
            },
        ) from error


# ============================================================
# Route avec image
# ============================================================
@app.post("/api/fraud/analyze-with-image")
async def analyze_fraud_with_image(
    sinistre_data: str = Form(...),
    image: UploadFile = File(...),
):
    """
    Analyse complète :

    1. Sauvegarde et contrôle MD5 de l'image
    2. Détection YOLOv8
    3. Extraction des features visuelles
    4. Calcul du score visuel
    5. Calcul du score Isolation Forest
    6. Fusion et décision
    """

    try:
        try:
            payload = json.loads(sinistre_data)
            print("=" * 80)
            print("Type :", type(sinistre_data))
            print("Valeur reçue :")
            print(repr(sinistre_data))
            print("=" * 80)

            payload = json.loads(sinistre_data)
        except json.JSONDecodeError as error:
            raise HTTPException(
                status_code=400,
                detail="sinistre_data doit contenir un objet JSON valide.",
            ) from error

        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400,
                detail="sinistre_data doit être un objet JSON.",
            )

        num_sinistre = validate_sinistre_data(payload)
        image_info = await save_fraud_image(image)

        clean_csv_updated = update_sinistre_image_path(
            num_sinistre=num_sinistre,
            image_path=image_info["relative_path"],
        )

        yolo_result = detect_damages(image_info["absolute_path"])

        if not yolo_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Erreur YOLOv8.",
                    "error": yolo_result.get("error"),
                },
            )

        visual_features = extract_visual_features(
            damaged_parts=yolo_result.get("damaged_parts", []),
            damage_scores=yolo_result.get("damage_scores", {}),
            sinistre_data=payload,
            uncertain_parts=yolo_result.get(
        "uncertain_parts",
        [],
    ),
        )

        visual_score = compute_visual_score(visual_features)

        visual_explanation = explain_with_phi35(
            image_path=image_info["absolute_path"],
            sinistre_data=payload,
            yolo_result=yolo_result,
            visual_features=visual_features,
            visual_score=visual_score,
        )

        montant = float(
            payload.get("TOTALREGLEMENT", 0) or 0
        )

        prime = float(
            payload.get("PRIME", 0) or 0
        )

        payload["ratio_prime_sinistre"] = (
            montant / prime
            if prime > 0
            else 0.0
        )

        ml_analysis = compute_ml_score(payload)

        print(
    "FIABILITE CALCULEE DANS APP :",
    visual_score.get("fiabilite_visuelle"),
        )

        print(   
        "FONCTION FUSION IMPORTEE DEPUIS :",
        fuse_scores.__module__,
       )

        decision = fuse_scores(
            score_ml=ml_analysis["score_ml"],
            score_visuel=visual_score["score_visuel_fraude"],
            has_image=True,
            visual_reliability=visual_score["fiabilite_visuelle"],
        )

        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "num_sinistre": num_sinistre,
            "has_image": True,
            "image_path": image_info["relative_path"],
            "duplicate_image": image_info["duplicate"],
            "image_md5": image_info["md5"],
            "clean_csv_updated": clean_csv_updated,
            "ml_analysis": ml_analysis,
            "visual_analysis": {
                "yolo": yolo_result,
                "visual_features": visual_features,
                "visual_score": visual_score,
                "explanation": visual_explanation,
            },
            "decision": decision,
        }

        save_fraud_result(response)
        return response

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Erreur pendant l'analyse complète du sinistre.",
                "error": str(error),
            },
        ) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)