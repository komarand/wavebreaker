from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kaggle_researcher.facts import discussions


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "facts" / "forum_topics.json"
)


class FakeForumsClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.list_calls: list[dict[str, Any]] = []
        self.get_calls: list[str] = []

    def list_topics(self, **kwargs: Any) -> SimpleNamespace:
        self.list_calls.append(kwargs)
        slug = kwargs["forum_slug"]
        if slug == "listing-fails":
            raise RuntimeError("list unavailable")
        if kwargs["category"] == "competition_write_ups":
            page = self.payload["writeup_pages"][slug]
        else:
            token = kwargs["page_token"] or "first"
            page = self.payload["competition_pages"][token]
        return SimpleNamespace(**page)

    def get_topic(self, topic_id: str) -> SimpleNamespace:
        self.get_calls.append(topic_id)
        if int(topic_id) in self.payload["failed_topic_ids"]:
            raise RuntimeError("thread unavailable")
        return SimpleNamespace(**self.payload["threads"][topic_id])


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeForumsClient:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    client = FakeForumsClient(payload)
    monkeypatch.setattr(discussions, "_forums_client", lambda: client)
    return client


def test_competition_discussions_paginate_to_limit_and_join_thread_text(
    fake_client: FakeForumsClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    facts = discussions.fetch_competition_discussions("current-comp", max_topics=3)

    assert [fact.topic_id for fact in facts] == ["101", "103"]
    assert [call["page_token"] for call in fake_client.list_calls] == [None, "page-2"]
    assert [call["page_size"] for call in fake_client.list_calls] == [3, 1]
    assert fake_client.get_calls == ["101", "102", "103"]
    assert facts[0].author == "competition-host"
    assert facts[0].author_is_host is True
    assert facts[0].votes == 35
    assert facts[0].created_at is not None
    assert facts[0].source_type == "discussion"
    assert facts[0].competition_id == "current-comp"
    assert facts[0].text == (
        "Use grouped validation.\n\n"
        "Groups must not cross folds.\n\n"
        "This removed the leakage."
    )
    assert "Failed to fetch discussion topic 102" in caplog.text


def test_zero_limit_does_not_create_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        discussions,
        "_forums_client",
        lambda: pytest.fail("client must not be created"),
    )

    assert discussions.fetch_competition_discussions("current-comp", 0) == []
    assert discussions.fetch_winner_writeups(["past-one"], 0) == []


def test_winner_writeups_use_category_top_sort_and_competition_ids(
    fake_client: FakeForumsClient,
) -> None:
    facts = discussions.fetch_winner_writeups(
        ["past-one", "past-two"],
        per_competition=2,
    )

    assert [fact.topic_id for fact in facts] == ["201", "202", "203"]
    assert [fact.competition_id for fact in facts] == [
        "past-one",
        "past-one",
        "past-two",
    ]
    assert all(fact.source_type == "winner_writeup" for fact in facts)
    assert all(call["category"] == "competition_write_ups" for call in fake_client.list_calls)
    assert all(call["sort_by"] == "top" for call in fake_client.list_calls)


def test_listing_failure_for_one_competition_does_not_abort_later_writeups(
    fake_client: FakeForumsClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    facts = discussions.fetch_winner_writeups(
        ["listing-fails", "past-two"],
        per_competition=1,
    )

    assert [fact.topic_id for fact in facts] == ["203"]
    assert "Failed to list discussion topics for listing-fails" in caplog.text


def test_topic_missing_required_source_fields_is_skipped(
    fake_client: FakeForumsClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_client.payload["competition_pages"]["first"]["topics"] = [
        {"id": 105, "title": "Missing host flag", "votes": 4}
    ]
    fake_client.payload["competition_pages"]["first"]["next_page_token"] = None
    fake_client.payload["threads"]["105"] = {
        "topic": {"id": 105, "content": "No host flag in the full topic either."},
        "comments": [],
    }

    facts = discussions.fetch_competition_discussions("current-comp", max_topics=1)

    assert facts == []
    assert "missing valid author_is_host" in caplog.text


def test_fixture_is_valid_json_with_required_sections() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert payload["competition_pages"]["first"]["topics"]
    assert payload["writeup_pages"]["past-one"]["topics"]
    assert payload["threads"]


def test_module_does_not_use_subprocess_or_kaggle_cli() -> None:
    source = Path(discussions.__file__).read_text(encoding="utf-8")

    assert "subprocess" not in source
    assert "kaggle.api" not in source
    assert "kaggle competitions" not in source
