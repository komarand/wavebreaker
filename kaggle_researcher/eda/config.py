from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


DEFAULT_EDA_RUNS_DIR = "./data/eda_runs"
DEFAULT_KAGGLE_DATASETS_DIR = "./data/kaggle_datasets"
DEFAULT_EDA_SCHEMA_VERSION = "1.0"
DEFAULT_EDA_PROFILE_SAMPLE_ROWS = 200_000
DEFAULT_EDA_MAX_PROFILE_ROWS_FULL_SCAN = 2_000_000
DEFAULT_EDA_MAX_ADVERSARIAL_ROWS = 500_000
DEFAULT_EDA_MAX_BASELINE_ROWS = 1_000_000
DEFAULT_EDA_RANDOM_SEED = 42

load_dotenv(".env")


class EdaConfigError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class EdaSettings:
    eda_runs_dir: str = DEFAULT_EDA_RUNS_DIR
    kaggle_datasets_dir: str = DEFAULT_KAGGLE_DATASETS_DIR
    eda_schema_version: str = DEFAULT_EDA_SCHEMA_VERSION
    eda_profile_sample_rows: int = DEFAULT_EDA_PROFILE_SAMPLE_ROWS
    eda_max_profile_rows_full_scan: int = DEFAULT_EDA_MAX_PROFILE_ROWS_FULL_SCAN
    eda_max_adversarial_rows: int = DEFAULT_EDA_MAX_ADVERSARIAL_ROWS
    eda_max_baseline_rows: int = DEFAULT_EDA_MAX_BASELINE_ROWS
    eda_random_seed: int = DEFAULT_EDA_RANDOM_SEED
    kaggle_username: str | None = None
    kaggle_key: str | None = field(default=None, repr=False)


def load_eda_config() -> EdaSettings:
    return EdaSettings(
        eda_runs_dir=os.getenv("EDA_RUNS_DIR", DEFAULT_EDA_RUNS_DIR),
        kaggle_datasets_dir=os.getenv("KAGGLE_DATASETS_DIR", DEFAULT_KAGGLE_DATASETS_DIR),
        eda_schema_version=os.getenv("EDA_SCHEMA_VERSION", DEFAULT_EDA_SCHEMA_VERSION),
        eda_profile_sample_rows=_get_positive_int_env(
            "EDA_PROFILE_SAMPLE_ROWS",
            DEFAULT_EDA_PROFILE_SAMPLE_ROWS,
        ),
        eda_max_profile_rows_full_scan=_get_positive_int_env(
            "EDA_MAX_PROFILE_ROWS_FULL_SCAN",
            DEFAULT_EDA_MAX_PROFILE_ROWS_FULL_SCAN,
        ),
        eda_max_adversarial_rows=_get_positive_int_env(
            "EDA_MAX_ADVERSARIAL_ROWS",
            DEFAULT_EDA_MAX_ADVERSARIAL_ROWS,
        ),
        eda_max_baseline_rows=_get_positive_int_env(
            "EDA_MAX_BASELINE_ROWS",
            DEFAULT_EDA_MAX_BASELINE_ROWS,
        ),
        eda_random_seed=_get_positive_int_env("EDA_RANDOM_SEED", DEFAULT_EDA_RANDOM_SEED),
        kaggle_username=os.getenv("KAGGLE_USERNAME"),
        kaggle_key=os.getenv("KAGGLE_KEY"),
    )


def _get_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise EdaConfigError(f"{name} must be a positive integer") from exc

    if value <= 0:
        raise EdaConfigError(f"{name} must be a positive integer")

    return value
