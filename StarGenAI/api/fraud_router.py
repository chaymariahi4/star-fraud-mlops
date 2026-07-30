from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib
import json
import uuid

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from StarGenAI.services.vision_pipeline import detect_damages
from StarGenAI.services.visual_features import extract_visual_features
from StarGenAI.services.visual_score import compute_visual_score
from StarGenAI.services.ml_pipelineFraude import compute_ml_score
from StarGenAI.services.fusion import fuse_scores


router = APIRouter(
    prefix="/api/fraud",
    tags=["Fraud Detection"],
)

STAR_DIR = Path(__file__).resolve().parents[1]

IMAGES_DIR = STAR_DIR / "data" / "images"
PROCESSED_DIR = STAR_DIR / "data" / "processed"

CLEAN_CSV = PROCESSED_DIR / "sinistres_clean.csv"
SCORED_CSV = PROCESSED_DIR / "sinistres_scored.csv"
HASH_FILE = IMAGES_DIR / "image_hashes.json"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def compute_md5(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def load_image_hashes() -> dict[str, str]:
    if not HASH_FILE.exists():
        return {}

    try:
        with HASH_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else {}

    except (OSError, json.JSONDecodeError):
        return {}


def save_image_hashes(hashes: dict[str, str]) -> None:
    with HASH_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            hashes,
            file,
            indent=2,
            ensure_ascii=False,
        )


async def save_uploaded_image(
    image: UploadFile,
) -> dict[str, Any]:
    original_name = image.filename or "image.jpg"
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Extension non autorisée : {extension}. "
                f"Formats acceptés : {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    content = await image.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Le fichier image est vide.",
        )

    image_md5 = compute_md5(content)
    hashes = load_image_hashes()

    duplicate_path = hashes.get(image_md5)

    if duplicate_path:
        absolute_duplicate = STAR_DIR / duplicate_path

        if absolute_duplicate.exists():
            return {
                "absolute_path": absolute_duplicate,
                "relative_path": duplicate_path,
                "md5": image_md5,
                "duplicate": True,
            }

    filename = f"{uuid.uuid4().hex}{extension}"
    absolute_path = IMAGES_DIR / filename

    absolute_path.write_bytes(content)

    relative_path = str(
        absolute_path.relative_to(STAR_DIR)
    ).replace("\\", "/")

    hashes[image_md5] = relative_path
    save_image_hashes(hashes)

    return {
        "absolute_path": absolute_path,
        "relative_path": relative_path,
        "md5": image_md5,
        "duplicate": False,
    }


def update_clean_csv(
    num_sinistre: str,
    image_path: str,
) -> bool:
    if not CLEAN_CSV.exists():
        return False

    df = pd.read_csv(CLEAN_CSV)

    if "NUM_SINISTRE" not in df.columns:
        return False

    if "image_path" not in df.columns:
        df["image_path"] = pd.NA

    mask = (
        df["NUM_SINISTRE"].astype(str)
        == str(num_sinistre)
    )

    if not mask.any():
        return False

    df.loc[mask, "image_path"] = image_path
    df.to_csv(CLEAN_CSV, index=False)

    return True


def append_scored_result(
    result: dict[str, Any],
) -> None:
    visual_analysis = result.get("visual_analysis")
    score_visuel = None
    damaged_parts = []

    if visual_analysis:
        score_visuel = visual_analysis[
            "visual_score"
        ]["score_visuel"]

        damaged_parts = visual_analysis[
            "yolo"
        ].get("damaged_parts", [])

    row = {
        "timestamp": result["timestamp"],
        "num_sinistre": result["num_sinistre"],
        "score_ml": result["ml_analysis"]["score_ml"],
        "score_visuel": score_visuel,
        "score_final": result["decision"]["score_final"],
        "decision": result["decision"]["decision"],
        "has_image": result["has_image"],
        "image_path": result.get("image_path"),
        "damaged_parts": json.dumps(
            damaged_parts,
            ensure_ascii=False,
        ),
    }

    row_df = pd.DataFrame([row])

    if SCORED_CSV.exists():
        row_df.to_csv(
            SCORED_CSV,
            mode="a",
            header=False,
            index=False,
        )
    else:
        row_df.to_csv(
            SCORED_CSV,
            index=False,
        )


def validate_sinistre_data(
    sinistre_data: dict[str, Any],
) -> str:
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


@router.post("/analyze")
def analyze_without_image(
    sinistre_data: dict[str, Any],
):
    """
    Analyse tabulaire uniquement avec Isolation Forest.
    """

    num_sinistre = validate_sinistre_data(
        sinistre_data
    )

    ml_analysis = compute_ml_score(
        sinistre_data
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

    append_scored_result(response)

    return response


@router.post("/analyze-with-image")
async def analyze_with_image(
    sinistre_data: str = Form(...),
    image: UploadFile = File(...),
):
    """
    Analyse tabulaire + YOLOv8 + score visuel.
    """

    try:
        payload = json.loads(sinistre_data)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail="sinistre_data doit contenir un JSON valide.",
        ) from error

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400,
            detail="sinistre_data doit être un objet JSON.",
        )

    num_sinistre = validate_sinistre_data(payload)

    image_info = await save_uploaded_image(image)

    csv_updated = update_clean_csv(
        num_sinistre=num_sinistre,
        image_path=image_info["relative_path"],
    )

    yolo_result = detect_damages(
        image_info["absolute_path"]
    )

    if not yolo_result.get("success", False):
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Échec de l'analyse YOLOv8.",
                "yolo_error": yolo_result.get("error"),
            },
        )

    visual_features = extract_visual_features(
        damaged_parts=yolo_result["damaged_parts"],
        damage_scores=yolo_result["damage_scores"],
        sinistre_data=payload,
    )

    visual_score = compute_visual_score(
        visual_features
    )

    ml_analysis = compute_ml_score(payload)

    decision = fuse_scores(
        score_ml=ml_analysis["score_ml"],
        score_visuel=visual_score["score_visuel"],
        has_image=True,
    )

    response = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "num_sinistre": num_sinistre,
        "has_image": True,
        "image_path": image_info["relative_path"],
        "duplicate_image": image_info["duplicate"],
        "image_md5": image_info["md5"],
        "clean_csv_updated": csv_updated,
        "ml_analysis": ml_analysis,
        "visual_analysis": {
            "yolo": yolo_result,
            "visual_features": visual_features,
            "visual_score": visual_score,
        },
        "decision": decision,
    }

    append_scored_result(response)

    return response