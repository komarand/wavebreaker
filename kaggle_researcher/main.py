from __future__ import annotations

import argparse
import asyncio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle_researcher",
        description="KaggleResearcher CLI (bootstrap placeholder).",
    )
    parser.add_argument("competition_url", nargs="?", help="Kaggle competition URL")
    parser.add_argument("competition_desc", nargs="?", help="Competition description")
    parser.add_argument("--competition-id", dest="competition_id", help="Optional competition identifier")
    return parser


async def run() -> int:
    parser = build_parser()
    parser.parse_args()
    print("NotImplemented")
    return 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
