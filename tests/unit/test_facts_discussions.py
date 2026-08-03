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
    assert facts.status == "collected"
    assert facts.error is None


def test_empty_discussion_response_is_distinct_from_forbidden(
    fake_client: FakeForumsClient,
) -> None:
    fake_client.payload["competition_pages"]["first"] = {
        "topics": [],
        "next_page_token": None,
    }

    facts = discussions.fetch_competition_discussions("current-comp", max_topics=3)

    assert facts == []
    assert facts.status == "empty"
    assert facts.error is None


def test_discussion_403_is_nonfatal_forbidden_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForbiddenError(RuntimeError):
        status = 403

    class ForbiddenClient:
        def list_topics(self, **kwargs: Any) -> Any:
            raise ForbiddenError("credential=must-not-leak")

    monkeypatch.setattr(discussions, "_forums_client", ForbiddenClient)

    facts = discussions.fetch_competition_discussions("restricted", max_topics=3)

    assert facts == []
    assert facts.status == "forbidden"
    assert facts.error == "Kaggle Discussion API returned HTTP 403."
    assert facts.limitation == "Kaggle Discussion API returned HTTP 403."
    assert "credential" not in facts.error


def test_unknown_discussion_error_is_failed_not_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedClient:
        def list_topics(self, **kwargs: Any) -> Any:
            raise RuntimeError("secret-value")

    monkeypatch.setattr(discussions, "_forums_client", FailedClient)

    facts = discussions.fetch_competition_discussions("broken", max_topics=3)

    assert facts == []
    assert facts.status == "failed"
    assert facts.error == "Kaggle Discussion API collection failed (RuntimeError)."
    assert "secret-value" not in facts.error


def test_discussion_pagination_accepts_camel_case_page_token(
    fake_client: FakeForumsClient,
) -> None:
    first_page = fake_client.payload["competition_pages"]["first"]
    first_page["nextPageToken"] = first_page.pop("next_page_token")

    discussions.fetch_competition_discussions("current-comp", max_topics=3)

    assert [call["page_token"] for call in fake_client.list_calls] == [None, "page-2"]


def test_zero_limit_does_not_create_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        discussions,
        "_forums_client",
        lambda: pytest.fail("client must not be created"),
    )

    assert discussions.fetch_competition_discussions("current-comp", 0) == []
    assert discussions.fetch_winner_writeups(["past-one"], 0) == []


def test_comment_pagination_uses_next_page_token() -> None:
    class LowLevelClient:
        def __init__(self) -> None:
            self.tokens: list[str | None] = []

        def get_topic(self, request: Any) -> Any:
            return SimpleNamespace(topic=SimpleNamespace(id=request.id))

        def list_comments(self, request: Any) -> Any:
            token = request.page_token or None
            self.tokens.append(token)
            if token is None:
                return SimpleNamespace(
                    comments=[SimpleNamespace(content="first")],
                    next_page_token="page-2",
                )
            return SimpleNamespace(
                comments=[SimpleNamespace(content="second")],
                next_page_token=None,
            )

    low_level = LowLevelClient()
    client = object.__new__(discussions._KaggleForumsClient)
    client._client = low_level
    client._request_policy = discussions.KaggleRequestPolicy(min_interval_seconds=0)

    thread = client.get_topic("101")

    assert low_level.tokens == [None, "page-2"]
    assert [comment.content for comment in thread.comments] == ["first", "second"]


def test_low_level_forums_calls_retry_http_429() -> None:
    class RateLimited(RuntimeError):
        status = 429

    class LowLevelClient:
        def __init__(self) -> None:
            self.calls = 0

        def list_topics(self, request: Any) -> Any:
            self.calls += 1
            if self.calls < 3:
                raise RateLimited("rate limited")
            return SimpleNamespace(topics=[], next_page_token=None)

    low_level = LowLevelClient()
    client = object.__new__(discussions._KaggleForumsClient)
    client._client = low_level
    client._request_policy = discussions.KaggleRequestPolicy(
        base_delay_seconds=0,
        min_interval_seconds=0,
    )

    response = client.list_topics(
        forum_slug="competition",
        page_size=10,
        page_token=None,
        category=None,
        sort_by=None,
    )

    assert response.topics == []
    assert low_level.calls == 3


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
