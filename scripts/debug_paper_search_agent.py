from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kaggle_researcher.agents.paper_search_agent import search_paper_sources


async def main() -> None:
    warnings: list[str] = []
    docs = await search_paper_sources(
        queries=[
            "credit default prediction machine learning",
            "Gini coefficient optimization",
        ],
        max_results=10,
        warnings=warnings,
    )

    print("num_docs:", len(docs))
    print("warnings:", warnings)

    for doc in docs[:10]:
        print("---")
        print("source:", doc.get("source"))
        print("id:", doc.get("id"))
        print("title:", doc.get("title"))
        print("url:", doc.get("url"))
        print("content_len:", len(doc.get("content") or ""))


if __name__ == "__main__":
    asyncio.run(main())
