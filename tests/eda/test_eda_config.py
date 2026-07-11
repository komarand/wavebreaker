from __future__ import annotations

import pytest

from kaggle_researcher.eda.config import (
    DEFAULT_EDA_MAX_ADVERSARIAL_ROWS,
    DEFAULT_EDA_MAX_BASELINE_ROWS,
    DEFAULT_EDA_MAX_COLUMN_CARDINALITY_SCAN_ROWS,
    DEFAULT_EDA_MAX_PROFILE_ROWS_FULL_SCAN,
    DEFAULT_EDA_MAX_TABLE_BYTES,
    DEFAULT_EDA_MODULE_TIMEOUT_SEC,
    DEFAULT_EDA_PROFILE_SAMPLE_ROWS,
    DEFAULT_EDA_RANDOM_SEED,
    DEFAULT_EDA_RUNS_DIR,
    DEFAULT_EDA_SCHEMA_VERSION,
    DEFAULT_KAGGLE_DATASETS_DIR,
    EdaConfigError,
    load_eda_config,
)


EDA_ENV_NAMES = [
    "EDA_RUNS_DIR",
    "KAGGLE_DATASETS_DIR",
    "EDA_SCHEMA_VERSION",
    "EDA_PROFILE_SAMPLE_ROWS",
    "EDA_MAX_PROFILE_ROWS_FULL_SCAN",
    "EDA_MAX_ADVERSARIAL_ROWS",
    "EDA_MAX_BASELINE_ROWS",
    "EDA_MAX_TABLE_BYTES",
    "EDA_MAX_COLUMN_CARDINALITY_SCAN_ROWS",
    "EDA_MODULE_TIMEOUT_SEC",
    "EDA_RANDOM_SEED",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "DEEPSEEK_API_KEY",
]


@pytest.fixture(autouse=True)
def isolated_eda_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in EDA_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_eda_config_defaults_do_not_require_deepseek_or_kaggle_credentials() -> None:
    settings = load_eda_config()

    assert settings.eda_runs_dir == DEFAULT_EDA_RUNS_DIR
    assert settings.kaggle_datasets_dir == DEFAULT_KAGGLE_DATASETS_DIR
    assert settings.eda_schema_version == DEFAULT_EDA_SCHEMA_VERSION
    assert settings.eda_profile_sample_rows == DEFAULT_EDA_PROFILE_SAMPLE_ROWS
    assert settings.eda_max_profile_rows_full_scan == DEFAULT_EDA_MAX_PROFILE_ROWS_FULL_SCAN
    assert settings.eda_max_adversarial_rows == DEFAULT_EDA_MAX_ADVERSARIAL_ROWS
    assert settings.eda_max_baseline_rows == DEFAULT_EDA_MAX_BASELINE_ROWS
    assert settings.eda_max_table_bytes == DEFAULT_EDA_MAX_TABLE_BYTES
    assert settings.eda_max_column_cardinality_scan_rows == DEFAULT_EDA_MAX_COLUMN_CARDINALITY_SCAN_ROWS
    assert settings.eda_module_timeout_sec == DEFAULT_EDA_MODULE_TIMEOUT_SEC
    assert settings.eda_random_seed == DEFAULT_EDA_RANDOM_SEED
    assert settings.kaggle_username is None
    assert settings.kaggle_key is None


def test_eda_config_env_overrides_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDA_RUNS_DIR", "./custom/eda_runs")
    monkeypatch.setenv("KAGGLE_DATASETS_DIR", "./custom/kaggle_datasets")
    monkeypatch.setenv("EDA_SCHEMA_VERSION", "1.1")
    monkeypatch.setenv("EDA_PROFILE_SAMPLE_ROWS", "123")
    monkeypatch.setenv("EDA_MAX_PROFILE_ROWS_FULL_SCAN", "456")
    monkeypatch.setenv("EDA_MAX_ADVERSARIAL_ROWS", "789")
    monkeypatch.setenv("EDA_MAX_BASELINE_ROWS", "321")
    monkeypatch.setenv("EDA_MAX_TABLE_BYTES", "654")
    monkeypatch.setenv("EDA_MAX_COLUMN_CARDINALITY_SCAN_ROWS", "987")
    monkeypatch.setenv("EDA_MODULE_TIMEOUT_SEC", "111")
    monkeypatch.setenv("EDA_RANDOM_SEED", "7")
    monkeypatch.setenv("KAGGLE_USERNAME", "kaggle-user")
    monkeypatch.setenv("KAGGLE_KEY", "super-secret-kaggle-key")

    settings = load_eda_config()

    assert settings.eda_runs_dir == "./custom/eda_runs"
    assert settings.kaggle_datasets_dir == "./custom/kaggle_datasets"
    assert settings.eda_schema_version == "1.1"
    assert settings.eda_profile_sample_rows == 123
    assert settings.eda_max_profile_rows_full_scan == 456
    assert settings.eda_max_adversarial_rows == 789
    assert settings.eda_max_baseline_rows == 321
    assert settings.eda_max_table_bytes == 654
    assert settings.eda_max_column_cardinality_scan_rows == 987
    assert settings.eda_module_timeout_sec == 111
    assert settings.eda_random_seed == 7
    assert settings.kaggle_username == "kaggle-user"
    assert settings.kaggle_key == "super-secret-kaggle-key"


def test_eda_secret_values_are_not_printed_in_settings_repr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAGGLE_KEY", "super-secret-kaggle-key")

    settings = load_eda_config()

    assert settings.kaggle_key == "super-secret-kaggle-key"
    assert "super-secret-kaggle-key" not in repr(settings)
    assert "kaggle_key" not in repr(settings)


@pytest.mark.parametrize(
    "name",
    [
        "EDA_PROFILE_SAMPLE_ROWS",
        "EDA_MAX_PROFILE_ROWS_FULL_SCAN",
        "EDA_MAX_ADVERSARIAL_ROWS",
        "EDA_MAX_BASELINE_ROWS",
        "EDA_MAX_TABLE_BYTES",
        "EDA_MAX_COLUMN_CARDINALITY_SCAN_ROWS",
        "EDA_MODULE_TIMEOUT_SEC",
        "EDA_RANDOM_SEED",
    ],
)
def test_positive_integer_eda_env_values_are_validated(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setenv(name, "0")

    with pytest.raises(EdaConfigError, match=f"{name} must be a positive integer"):
        load_eda_config()


def test_non_integer_eda_env_value_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDA_RANDOM_SEED", "not-an-int")

    with pytest.raises(EdaConfigError, match="EDA_RANDOM_SEED must be a positive integer"):
        load_eda_config()
