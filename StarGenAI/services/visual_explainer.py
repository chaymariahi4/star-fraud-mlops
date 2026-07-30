from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ENABLE_VLM = (
    os.getenv("ENABLE_VLM", "false").lower()
    == "true"
)


def build_rule_based_explanation(
    visual_features: dict[str, Any],
    visual_score: dict[str, Any],
    yolo_result: dict[str, Any],
) -> dict[str, Any]:
    reasons: list[str] = []
    flags = visual_score.get("flags", [])

    damaged_parts = yolo_result.get(
        "damaged_parts",
        [],
    )

    if damaged_parts:
        reasons.append(
            "Les pièces détectées comme endommagées sont : "
            + ", ".join(damaged_parts)
            + "."
        )
    else:
        reasons.append(
            "Aucune pièce endommagée suffisamment fiable "
            "n'a été détectée."
        )

    coherence_status = visual_features.get(
        "coherence_dommages",
        "INDETERMINE",
    )

    coherence_message = visual_features.get(
        "message_coherence_dommages",
    )

    if coherence_message:
        reasons.append(coherence_message)

    if coherence_status == "INCOHERENT":
        reasons.append(
            "Cette incohérence constitue un signal de contrôle "
            "important, mais ne constitue pas à elle seule une "
            "preuve de fraude."
        )

    elif coherence_status == "POTENTIELLEMENT_INCOHERENT":
        reasons.append(
            "L'incohérence reste incertaine en raison de la "
            "faible confiance des détections. Une validation "
            "humaine est recommandée."
        )

    elif coherence_status == "COHERENT":
        reasons.append(
            "Les dommages observés sont compatibles avec la "
            "déclaration de l'assuré."
        )

    elif coherence_status == "PARTIELLEMENT_COHERENT":
        reasons.append(
            "La zone déclarée est confirmée, mais les dommages "
            "semblent également toucher d'autres parties du véhicule."
        )

    severity_label = visual_score.get(
        "libelle_severite",
        "Indéterminée",
    )

    reasons.append(
        f"La gravité apparente des dommages est évaluée comme "
        f"« {severity_label.lower()} »."
    )

    if visual_features.get(
        "montant_disproportionne"
    ):
        reasons.append(
            "Le montant réclamé paraît élevé au regard du "
            "nombre de pièces détectées."
        )

    if visual_features.get(
        "absence_dommage_declare"
    ):
        reasons.append(
            "Des dommages ont été déclarés, mais aucune "
            "détection visuelle fiable n'a été obtenue."
        )

    if visual_features.get(
        "faible_confiance_globale"
    ):
        reasons.append(
            "La confiance du modèle visuel est faible ; "
            "une vérification humaine est recommandée."
        )

    if not flags:
        reasons.append(
            "Aucune anomalie visuelle métier forte "
            "n'a été identifiée."
        )

    return {
        "raison": " ".join(reasons),
        "flags": flags,
        "source": "regles_metier",
        "used_for_scoring": False,
    }


def explain_with_phi35(
    image_path: str | Path,
    sinistre_data: dict[str, Any],
    yolo_result: dict[str, Any],
    visual_features: dict[str, Any],
    visual_score: dict[str, Any],
) -> dict[str, Any]:
    """
    Produit uniquement une explication.
    Le résultat ne doit jamais modifier les scores.
    """

    if not ENABLE_VLM:
        return build_rule_based_explanation(
            visual_features=visual_features,
            visual_score=visual_score,
            yolo_result=yolo_result,
        )

    try:
        import torch

        from PIL import Image
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor,
        )

        model_id = (
            "microsoft/Phi-3.5-vision-instruct"
        )

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="eager",
        )

        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=4,
        )

        context = {
            "sinistre": {
                "description": sinistre_data.get(
                    "DESCRIPTION_INCIDENT"
                ),
                "montant": sinistre_data.get(
                    "TOTALREGLEMENT"
                ),
                "type_sinistre": sinistre_data.get(
                    "TYPE_SINISTRE"
                ),
            },
            "yolo": {
                "damaged_parts": yolo_result.get(
                    "damaged_parts",
                    [],
                ),
                "damage_scores": yolo_result.get(
                    "damage_scores",
                    {},
                ),
            },
            "visual_features": visual_features,
            "visual_flags": visual_score.get(
                "flags",
                [],
            ),
        }

        instruction = f"""
Tu es un assistant d'analyse de sinistres automobiles.

Analyse uniquement l'image et les informations fournies.
Ne calcule aucun score.
Ne décide pas si le dossier est frauduleux.
Ne contredis pas les détections YOLO sans signaler l'incertitude.

Retourne uniquement un JSON valide sous cette forme :
{{
  "raison": "explication concise en français",
  "flags": ["signal_1", "signal_2"]
}}

Informations structurées :
{json.dumps(context, ensure_ascii=False)}
""".strip()

        messages = [
            {
                "role": "user",
                "content": (
                    "<|image_1|>\n"
                    + instruction
                ),
            }
        ]

        prompt = (
            processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        inputs = processor(
            prompt,
            [image],
            return_tensors="pt",
        ).to(device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=220,
            temperature=0.0,
            do_sample=False,
            eos_token_id=(
                processor.tokenizer.eos_token_id
            ),
        )

        generated_ids = generated_ids[
            :,
            inputs["input_ids"].shape[1]:,
        ]

        generated_text = (
            processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        ).strip()

        try:
            parsed = json.loads(generated_text)

            return {
                "raison": str(
                    parsed.get(
                        "raison",
                        generated_text,
                    )
                ),
                "flags": parsed.get(
                    "flags",
                    [],
                ),
                "source": "phi_3_5_vision",
                "used_for_scoring": False,
            }

        except json.JSONDecodeError:
            return {
                "raison": generated_text,
                "flags": visual_score.get(
                    "flags",
                    [],
                ),
                "source": "phi_3_5_vision",
                "used_for_scoring": False,
                "json_parse_success": False,
            }

    except Exception as error:
        fallback = build_rule_based_explanation(
            visual_features=visual_features,
            visual_score=visual_score,
            yolo_result=yolo_result,
        )

        fallback["vlm_error"] = str(error)
        fallback["fallback_used"] = True

        return fallback