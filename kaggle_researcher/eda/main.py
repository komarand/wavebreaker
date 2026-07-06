from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle_eda_engine",
        description="Kaggle EDA Engine data evidence layer.",
    )
    parser.add_argument("--competition-id", help="Kaggle competition slug or identifier")
    parser.add_argument("--competition-url", help="Kaggle competition URL")
    parser.add_argument("--hypotheses-path", type=Path, help="Path to research_hypotheses.json")
    parser.add_argument("--task-plan-path", type=Path, help="Path to eda_task_plan.json")
    parser.add_argument("--local-dataset-path", type=Path, help="Path to a local dataset directory")
    parser.add_argument("--output-dir", type=Path, default=Path("./data/eda_runs"), help="EDA run output directory")
    parser.add_argument(
        "--download-dataset",
        dest="download_dataset",
        action="store_true",
        default=False,
        help="Download Kaggle competition data when no local dataset path is provided",
    )
    parser.add_argument(
        "--no-download-dataset",
        dest="download_dataset",
        action="store_false",
        help="Disable Kaggle dataset download",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=[],
        help="Specific EDA modules to run when execution is implemented",
    )
    parser.add_argument(
        "--skip-modules",
        nargs="+",
        default=[],
        help="Specific EDA modules to skip when execution is implemented",
    )
    parser.add_argument(
        "--enable-p1-modules",
        action="store_true",
        help="Enable optional P1 EDA modules when implemented",
    )
    parser.add_argument(
        "--enable-baseline",
        action="store_true",
        help="Enable the optional honest baseline module when implemented",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first module failure when execution is implemented",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    print("Kaggle EDA Engine CLI is not implemented yet.")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
