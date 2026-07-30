from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "StarGenAI"
    / "Data"
    / "processed"
    / "features_finales.csv"
)

REPORTS_DIR = Path(__file__).resolve().parent / "monitoring_reports"


FEATURES = [
    "delai_declaration",
    "nb_sinistres_assure",
    "nb_sinistres_vehicule",
    "nb_sinistres_meme_jour",
    "nb_sinistres_expert",
    "ratio_montant_moyenne_annee",
    "montant_zero",
    "conducteur_different",
    "usage_risque",
    "type_sinistre_encode",
    "est_contrat_collectif",
    "age",
    "ratio_prime_sinistre",
    "delai_souscription_sinistre",
    "PRIME",
    "annee_sinistre",
]


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}"
        )

    dataframe = pd.read_csv(DATA_PATH)

    missing_columns = [
        column
        for column in FEATURES
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans le dataset : "
            + ", ".join(missing_columns)
        )

    monitoring_data = dataframe[FEATURES].copy()

    monitoring_data = monitoring_data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return monitoring_data


def split_reference_current(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(dataframe) < 100:
        raise ValueError(
            "Le dataset doit contenir au moins 100 lignes."
        )

    split_index = int(len(dataframe) * 0.80)

    reference_data = dataframe.iloc[:split_index].copy()
    current_data = dataframe.iloc[split_index:].copy()

    if reference_data.empty or current_data.empty:
        raise ValueError(
            "Impossible de créer les jeux de référence et courant."
        )

    return reference_data, current_data


def simulate_data_drift(
    current_data: pd.DataFrame,
) -> pd.DataFrame:
    drifted_data = current_data.copy()

    drifted_data["age"] = drifted_data["age"] + 12

    drifted_data["PRIME"] = drifted_data["PRIME"] * 1.40

    drifted_data["delai_declaration"] = (
        drifted_data["delai_declaration"] * 1.75
    )

    drifted_data["nb_sinistres_assure"] = (
        drifted_data["nb_sinistres_assure"] + 2
    )

    return drifted_data


def generate_monitoring_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = Report(
        [
            DataDriftPreset(),
            DataSummaryPreset(),
        ],
        include_tests=True,
    )

    result = report.run(
        current_data=current_data,
        reference_data=reference_data,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    html_path = (
        REPORTS_DIR
        / f"fraud_monitoring_report_{timestamp}.html"
    )

    json_path = (
        REPORTS_DIR
        / f"fraud_monitoring_report_{timestamp}.json"
    )

    result.save_html(str(html_path))
    result.save_json(str(json_path))

    return html_path, json_path


def main() -> None:
    print("Chargement du dataset...")

    dataframe = load_dataset()

    print(f"Nombre total de lignes : {len(dataframe)}")
    print(f"Nombre de variables surveillées : {len(FEATURES)}")

    reference_data, current_data = split_reference_current(
        dataframe
    )

    current_data = simulate_data_drift(current_data)

    print(
        "Données de référence : "
        f"{len(reference_data)} lignes"
    )

    print(
        "Données courantes : "
        f"{len(current_data)} lignes"
    )

    html_path, json_path = generate_monitoring_report(
        reference_data=reference_data,
        current_data=current_data,
    )

    print()
    print("Monitoring Evidently terminé avec succès.")
    print(f"Rapport HTML : {html_path}")
    print(f"Rapport JSON : {json_path}")


if __name__ == "__main__":
    main()