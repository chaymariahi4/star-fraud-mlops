from __future__ import annotations

import importlib


REQUIRED_MODULES = [
    "fastapi",
    "pandas",
    "numpy",
    "sklearn",
    "mlflow",
    "dvc",
    "evidently",
]


def test_required_modules_are_available() -> None:
    missing_modules: list[str] = []

    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing_modules.append(module_name)

    assert not missing_modules, (
        "Modules Python absents : "
        + ", ".join(missing_modules)
    )