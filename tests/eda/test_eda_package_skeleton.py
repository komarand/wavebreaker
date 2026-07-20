from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from kaggle_researcher.eda.main import build_parser
from kaggle_researcher.eda.schemas import EdaRunConfig


def test_eda_package_imports() -> None:
    import kaggle_eda_engine
    import kaggle_researcher.eda

    assert kaggle_eda_engine is not None
    assert kaggle_researcher.eda is not None


def test_eda_cli_help_works() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "kaggle_eda_engine.main", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--competition-id" in result.stdout
    assert "--no-download-dataset" in result.stdout
    assert "--enable-baseline" in result.stdout
    assert "--enable-baseline-ablations" in result.stdout
    assert "--enable-slice-diagnostics" in result.stdout


def test_eda_cli_slice_diagnostics_flag_defaults_and_parses() -> None:
    parser = build_parser()
    defaults = parser.parse_args([])
    enabled = parser.parse_args(["--enable-slice-diagnostics"])

    assert defaults.enable_slice_diagnostics is False
    assert enabled.enable_slice_diagnostics is True


def test_slice_diagnostics_config_field_accepts_cli_value() -> None:
    config = EdaRunConfig(
        competition_id="fixture",
        hypotheses_path=Path("hypotheses.json"),
        task_plan_path=Path("task_plan.json"),
        enable_slice_diagnostics=True,
    )

    assert config.enable_slice_diagnostics is True
