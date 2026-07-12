from __future__ import annotations

import subprocess
import sys


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
