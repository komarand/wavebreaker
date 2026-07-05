from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kaggle_researcher.agents.kaggle_agent import get_notebook_content, search_notebooks


def main() -> int:
    competition_id = "home-credit-credit-risk-model-stability"
    queries = ["home credit credit risk model stability"]

    notebooks = search_notebooks(queries, competition_id=competition_id)
    print(f"num_notebooks={len(notebooks)}")

    for notebook in notebooks[:5]:
        print(
            "notebook "
            f"ref={notebook.get('id')} "
            f"title={notebook.get('title')} "
            f"votes={notebook.get('total_votes')}"
        )

    if not notebooks:
        return 1

    first_ref = str(notebooks[0]["id"])
    content = get_notebook_content(first_ref)
    print(f"first_ref={first_ref}")
    print(f"content_len={len(content)}")
    print("preview:")
    print(content[:1000])
    return 0 if content else 1


if __name__ == "__main__":
    raise SystemExit(main())
