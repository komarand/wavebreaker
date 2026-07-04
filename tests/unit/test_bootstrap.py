from __future__ import annotations

import subprocess
import sys


def test_import_package() -> None:
    import kaggle_researcher

    assert kaggle_researcher.__version__ == "0.1.0"


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "kaggle_researcher.main", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
