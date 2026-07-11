from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from kaggle_researcher.eda.config import load_eda_config
from kaggle_researcher.eda.orchestrator import run_eda
from kaggle_researcher.eda.schemas import EdaRunConfig


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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/eda_runs"),
        help="EDA run output directory",
    )
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
        "--enable-notebook-static-analysis",
        action="store_true",
        help="Enable optional static notebook pattern extraction when P1 modules run",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first module failure when execution is implemented",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _require_args(parser, args)
    settings = load_eda_config()
    config = EdaRunConfig(
        competition_id=args.competition_id,
        competition_url=args.competition_url,
        hypotheses_path=args.hypotheses_path,
        task_plan_path=args.task_plan_path,
        local_dataset_path=args.local_dataset_path,
        output_dir=args.output_dir or Path(settings.eda_runs_dir),
        download_dataset=args.download_dataset,
        force_download=False,
        modules=args.modules or None,
        skip_modules=args.skip_modules or [],
        enable_p1_modules=args.enable_p1_modules,
        enable_baseline=args.enable_baseline,
        enable_notebook_static_analysis=args.enable_notebook_static_analysis,
        fail_fast=args.fail_fast,
        profile_sample_rows=settings.eda_profile_sample_rows,
        max_profile_rows_full_scan=settings.eda_max_profile_rows_full_scan,
        max_adversarial_rows=settings.eda_max_adversarial_rows,
        max_baseline_rows=settings.eda_max_baseline_rows,
        max_table_bytes=settings.eda_max_table_bytes,
        max_column_cardinality_scan_rows=settings.eda_max_column_cardinality_scan_rows,
        module_timeout_sec=settings.eda_module_timeout_sec,
        random_seed=settings.eda_random_seed,
    )
    result = asyncio.run(run_eda(config))
    print(f"EDA evidence pack: {result.evidence_pack_path}")
    print(f"EDA summary: {result.summary_path}")
    return 0


def _require_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    required = {
        "--competition-id": args.competition_id,
        "--hypotheses-path": args.hypotheses_path,
        "--task-plan-path": args.task_plan_path,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
