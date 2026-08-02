from __future__ import annotations

import importlib.util
import os

import pytest

from kaggle_researcher.facts.discussions import fetch_competition_discussions


@pytest.mark.network
def test_titanic_discussion_live_sanity() -> None:
    if importlib.util.find_spec("kagglesdk") is None:
        pytest.skip("kagglesdk is required for the live discussion test")
    if not (
        os.getenv("KAGGLE_API_TOKEN")
        or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    ):
        pytest.skip("Kaggle credentials are required for the live discussion test")

    facts = fetch_competition_discussions("titanic", max_topics=1)

    assert len(facts) == 1
    assert facts[0].competition_id == "titanic"
    assert facts[0].source_type == "discussion"
