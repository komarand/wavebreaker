from __future__ import annotations

import argparse
from collections.abc import Sequence


OBJECTIVES = ("medal", "top_percent", "learn", "fast_baseline")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wave",
        description="Wavebreaker B5 competition research pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    facts_parser = subparsers.add_parser(
        "facts",
        help="Collect deterministic competition facts.",
    )
    _add_competition_arguments(facts_parser)

    brief_parser = subparsers.add_parser(
        "brief",
        help="Generate a grounded competition brief.",
    )
    _add_competition_arguments(brief_parser)
    brief_parser.add_argument("--vram", type=float, help="Available VRAM in GB.")
    brief_parser.add_argument("--hours", type=float, help="Available hours per week.")
    brief_parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        default="medal",
        help="Primary competition objective.",
    )
    brief_parser.add_argument(
        "--facts-from",
        help="Read previously collected facts from this path.",
    )

    journal_parser = subparsers.add_parser(
        "journal",
        help="Record the outcome of a competition.",
    )
    journal_parser.add_argument("slug", help="Kaggle competition slug.")
    journal_parser.add_argument("--used-validation", help="Validation scheme used.")
    journal_parser.add_argument("--final-rank", type=int, help="Final leaderboard rank.")
    journal_parser.add_argument("--num-teams", type=int, help="Number of competing teams.")
    journal_parser.add_argument(
        "--brief-was-useful",
        choices=("yes", "no"),
        help="Whether the generated brief was useful.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raise NotImplementedError(f"wave {args.command} is not implemented yet")


def _add_competition_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("slug", help="Kaggle competition slug.")
    parser.add_argument("--max-notebooks", type=int, help="Maximum notebooks to collect.")
    parser.add_argument("--max-discussions", type=int, help="Maximum discussions to collect.")
    parser.add_argument("--similar", help="Comma-separated similar competition slugs.")
    parser.add_argument("--out", help="Output directory.")


if __name__ == "__main__":
    main()
