# conftest.py
import os
from pathlib import Path

import great_expectations as ge
import pandas as pd
import pytest
import yaml


def pytest_addoption(parser):
    parser.addoption(
        "--dataset-loc",
        action="store",
        default=None,
        help="Path or URL to the CSV dataset file (optional — fallback: config/config.yaml).",
    )


def _find_config(relative_path="config/config.yaml", max_depth=5):
    """Search upward from this file for config/config.yaml (up to max_depth parents)."""
    start = Path(__file__).resolve()
    for depth in range(max_depth):
        candidate = start.parents[depth] / relative_path
        if candidate.exists():
            return candidate
    return None


def _load_config_yaml(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load YAML config at {path}: {e}")


@pytest.fixture(scope="module")
def df(request):
    # 1) CLI option has highest precedence
    dataset_loc = request.config.getoption("--dataset-loc")

    # 2) then environment variable
    if not dataset_loc:
        dataset_loc = os.getenv("DATASET_LOC")

    # 3) then config/config.yaml
    if not dataset_loc:
        config_path = _find_config()
        if config_path:
            cfg = _load_config_yaml(config_path)
            # expecting structure: data: { dataset_loc: "..." }
            dataset_loc = cfg.get("data", {}).get("dataset_loc") if isinstance(cfg, dict) else None

    # If still missing, raise a helpful error
    if not dataset_loc:
        raise pytest.UsageError(
            "Dataset location not provided. Provide it with:\n"
            "  1) CLI: --dataset-loc=PATH_OR_URL\n"
            "  2) env var: export DATASET_LOC=PATH_OR_URL\n"
            "  3) config file: config/config.yaml with 'data: dataset_loc: \"...\"'\n\n"
            "Examples:\n"
            "  pytest --dataset-loc=https://raw.githubusercontent.com/.../dataset.csv tests/data -q\n"
            "  export DATASET_LOC=tests/data/my_dataset.csv && pytest tests/data -q"
        )

    # Try to read the CSV (pandas can read local file paths and http(s) URLs)
    try:
        df_pd = pd.read_csv(dataset_loc)
    except Exception as e:
        raise pytest.UsageError(f"Failed to read CSV from '{dataset_loc}': {e}")

    # Wrap into Great Expectations PandasDataset and return
    try:
        return ge.dataset.PandasDataset(df_pd)
    except Exception as e:
        raise pytest.UsageError(f"Failed to create Great Expectations DataSet from DataFrame: {e}")
