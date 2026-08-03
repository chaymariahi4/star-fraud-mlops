from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "yolov8_damage.pt"
)

ANNOTATED_DIR = (
    BASE_DIR
    / "data"
    / "annotated"
)


CLASS_THRESHOLDS = {
    "Bonnet": 0.35,
    "Bumper": 0.35,
    "Dickey": 0.30,
    "Door": 0.35,
    "Fender": 0.25,
    "Light": 0.30,
    "Windshield": 0.45,
}

UNCERTAIN_MIN_CONFIDENCE = 0.15


@lru_cache(maxsize=1)
def get_yolo_model() -> Any:
    """
    Charge YOLO uniquement lors de la première
    analyse d'une image.

    L'import de FastAPI et les tests légers ne
    nécessitent donc pas le fichier .pt.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle YOLO introuvable : {MODEL_PATH}"
        )

    try:
        # Import différé pour éviter de charger
        # Ultralytics au démarrage de FastAPI.
        from ultralytics import YOLO

    except ImportError as exc:
        raise RuntimeError(
            "La bibliothèque ultralytics "
            "n'est pas installée."
        ) from exc

    try:
        return YOLO(
            str(MODEL_PATH)
        )

    except Exception as exc:
        raise RuntimeError(
            "Impossible de charger le modèle YOLO : "
            f"{MODEL_PATH}"
        ) from exc


def clear_yolo_cache() -> None:
    """
    Vide le cache du modèle YOLO.
    """

    get_yolo_model.cache_clear()


def detect_damages(
    image_path: str | Path,
) -> dict[str, Any]:
    """
    Détecte les pièces endommagées sur une image.

    Les résultats sont séparés en :
    - détections confirmées ;
    - détections incertaines.

    Le modèle YOLO est chargé uniquement lorsque
    cette fonction est réellement appelée.
    """

    image_path = Path(
        image_path
    )

    if not image_path.exists():
        return {
            "damaged_parts": [],
            "damage_scores": {},
            "detections": [],
            "uncertain_parts": [],
            "uncertain_scores": {},
            "uncertain_detections": [],
            "num_damages": 0,
            "num_uncertain": 0,
            "annotated_image_path": None,
            "success": False,
            "error": (
                f"Image introuvable : {image_path}"
            ),
        }

    try:
        model = get_yolo_model()

        results = model.predict(
            source=str(image_path),
            imgsz=960,
            conf=UNCERTAIN_MIN_CONFIDENCE,
            iou=0.45,
            max_det=50,
            verbose=False,
        )

        confirmed_detections: list[
            dict[str, Any]
        ] = []

        uncertain_detections: list[
            dict[str, Any]
        ] = []

        confirmed_scores: dict[
            str,
            float,
        ] = {}

        uncertain_scores: dict[
            str,
            float,
        ] = {}

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(
                    box.cls.item()
                )

                confidence = float(
                    box.conf.item()
                )

                class_name = str(
                    model.names[class_id]
                )

                bbox = [
                    round(
                        float(value),
                        2,
                    )
                    for value in (
                        box.xyxy[0].tolist()
                    )
                ]

                threshold = (
                    CLASS_THRESHOLDS.get(
                        class_name,
                        0.35,
                    )
                )

                detection = {
                    "class": class_name,
                    "confidence": round(
                        confidence,
                        4,
                    ),
                    "threshold": threshold,
                    "bbox": bbox,
                }

                if confidence >= threshold:
                    detection["status"] = (
                        "confirmed"
                    )

                    confirmed_detections.append(
                        detection
                    )

                    previous_score = (
                        confirmed_scores.get(
                            class_name,
                            0.0,
                        )
                    )

                    if confidence > previous_score:
                        confirmed_scores[
                            class_name
                        ] = confidence

                elif (
                    confidence
                    >= UNCERTAIN_MIN_CONFIDENCE
                ):
                    detection["status"] = (
                        "uncertain"
                    )

                    uncertain_detections.append(
                        detection
                    )

                    previous_score = (
                        uncertain_scores.get(
                            class_name,
                            0.0,
                        )
                    )

                    if confidence > previous_score:
                        uncertain_scores[
                            class_name
                        ] = confidence

        damaged_parts = sorted(
            confirmed_scores,
            key=confirmed_scores.get,
            reverse=True,
        )

        damage_scores = {
            part: round(
                confirmed_scores[part],
                4,
            )
            for part in damaged_parts
        }

        uncertain_parts = sorted(
            uncertain_scores,
            key=uncertain_scores.get,
            reverse=True,
        )

        formatted_uncertain_scores = {
            part: round(
                uncertain_scores[part],
                4,
            )
            for part in uncertain_parts
        }

        annotated_path: str | None = None

        if results:
            try:
                import cv2

                ANNOTATED_DIR.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                plotted_image = (
                    results[0].plot()
                )

                annotated_filename = (
                    f"{image_path.stem}"
                    "_annotated.jpg"
                )

                annotated_absolute_path = (
                    ANNOTATED_DIR
                    / annotated_filename
                )

                write_success = cv2.imwrite(
                    str(
                        annotated_absolute_path
                    ),
                    plotted_image,
                )

                if write_success:
                    annotated_path = str(
                        annotated_absolute_path.relative_to(
                            BASE_DIR
                        )
                    ).replace(
                        "\\",
                        "/",
                    )

            except Exception:
                # L'échec de génération de l'image annotée
                # ne doit pas annuler les détections YOLO.
                annotated_path = None

        return {
            "damaged_parts": damaged_parts,
            "damage_scores": damage_scores,
            "detections": confirmed_detections,
            "uncertain_parts": uncertain_parts,
            "uncertain_scores": (
                formatted_uncertain_scores
            ),
            "uncertain_detections": (
                uncertain_detections
            ),
            "num_damages": len(
                damaged_parts
            ),
            "num_uncertain": len(
                uncertain_parts
            ),
            "annotated_image_path": (
                annotated_path
            ),
            "model": "YOLOv8n",
            "image_size": 960,
            "success": True,
        }

    except Exception as error:
        return {
            "damaged_parts": [],
            "damage_scores": {},
            "detections": [],
            "uncertain_parts": [],
            "uncertain_scores": {},
            "uncertain_detections": [],
            "num_damages": 0,
            "num_uncertain": 0,
            "annotated_image_path": None,
            "success": False,
            "error": str(error),
        }