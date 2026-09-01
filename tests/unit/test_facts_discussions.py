from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kaggle_researcher.facts import discussions


class HttpError(RuntimeError):
    def __init__(self, status: int) -> None:
        super().__init__(f"secret response body for HTTP {status}")
        self.status = status


class FakeCompetitionClient:
    def __init__(
        self,
        *,
        topic_pages: dict[int, Any],
        messages: dict[int, Any],
    ) -> None:
        self.topic_pages = topic_pages
        self.message_responses = messages
        self.topic_calls: list[tuple[str, int | None]] = []
        self.message_calls: list[tuple[str, int, int | None]] = []

    def list_topics(self, slug: str, *, page: int | None = None) -> Any:
        self.topic_calls.append((slug, page))
        response = self.topic_pages.get(page or 1, SimpleNamespace(topics=[]))
        if isinstance(response, BaseException):
            raise response
        return response

    def list_topic_messages(
        self,
        slug: str,
        topic_id: int,
        *,
        page_size: int | None = None,
    ) -> Any:
        self.message_calls.append((slug, topic_id, page_size))
        response = self.message_responses.get(
            topic_id,
            SimpleNamespace(messages=[]),
        )
        if isinstance(response, BaseException):
            raise response
        return response


def _topic(
    topic_id: int,
    title: str = "Useful discussion",
    **overrides: Any,
) -> SimpleNamespace:
    payload = {
        "id": topic_id,
        "title": title,
        "topic_url": "",
        "author_name": "topic-author",
        "comment_count": 1,
        "votes": 4,
        "post_date": None,
        "last_comment_post_date": None,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _message(
    message_id: int,
    content: str = "<p>Hello</p>",
    **overrides: Any,
) -> SimpleNamespace:
    payload = {
        "id": message_id,
        "content": content,
        "author_name": None,
        "post_date": None,
        "votes": 1,
        "replies": [],
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _install_client(
    monkeypatch: pytest.MonkeyPatch,
    client: FakeCompetitionClient,
) -> None:
    monkeypatch.setattr(discussions, "_competition_discussions_client", lambda: client)
    monkeypatch.setattr(discussions, "_between_topics_sleep", lambda seconds: None)


def test_competition_route_paginates_topics_and_collects_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html = (
        "<p>A beginner-friendly tour of computer vision...</p>"
        '<p>Read it here: <a href="https://medium.com/@author/example">'
        "Article</a></p>"
    )
    client = FakeCompetitionClient(
        topic_pages={
            1: SimpleNamespace(topics=[_topic(101), _topic(102)], total_count=3),
            2: SimpleNamespace(topics=[_topic(102), _topic(103)], total_count=3),
        },
        messages={
            101: SimpleNamespace(messages=[_message(1001, html)]),
            102: SimpleNamespace(messages=[]),
            103: SimpleNamespace(messages=[_message(1003, "<p>Third</p>")]),
        },
    )
    _install_client(monkeypatch, client)

    facts = discussions.fetch_competition_discussions("jaguar-re-id", 10)

    assert facts.status == "collected"
    assert [fact.topic_id for fact in facts] == ["101", "102", "103"]
    assert client.topic_calls == [("jaguar-re-id", 1), ("jaguar-re-id", 2)]
    assert client.message_calls == [
        ("jaguar-re-id", 101, -1),
        ("jaguar-re-id", 102, -1),
        ("jaguar-re-id", 103, -1),
    ]
    assert facts[0].evidence_id == "discussion-topic:jaguar-re-id:101"
    assert facts[0].messages[0].evidence_id == ("discussion-message:jaguar-re-id:101:1001")
    assert "<p>" not in facts[0].text
    assert facts[0].messages[0].links[0].kind == "external"
    assert facts[1].collection_status == "empty"


def test_empty_topic_response_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeCompetitionClient(
        topic_pages={1: SimpleNamespace(topics=[], total_count=0)},
        messages={},
    )
    _install_client(monkeypatch, client)

    facts = discussions.fetch_competition_discussions("empty", 10)

    assert facts == []
    assert facts.status == "empty"
    assert facts.error is None


def test_repeated_topic_ids_stop_page_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeCompetitionClient(
        topic_pages={
            1: SimpleNamespace(topics=[_topic(1)], total_count=99),
            2: SimpleNamespace(topics=[_topic(1)], total_count=99),
            3: pytest.fail,
        },
        messages={1: SimpleNamespace(messages=[])},
    )
    _install_client(monkeypatch, client)

    facts = discussions.fetch_competition_discussions("repeat", 20)

    assert [fact.topic_id for fact in facts] == ["1"]
    assert client.topic_calls == [("repeat", 1), ("repeat", 2)]


def test_topic_page_safety_cap_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeCompetitionClient(
        topic_pages={
            page: SimpleNamespace(topics=[_topic(page)], total_count=999) for page in range(1, 5)
        },
        messages={1: SimpleNamespace(messages=[]), 2: SimpleNamespace(messages=[])},
    )
    _install_client(monkeypatch, client)
    monkeypatch.setattr(discussions, "TOPICS_PAGE_SAFETY_CAP", 2)

    facts = discussions.fetch_competition_discussions("bounded", 20)

    assert [fact.topic_id for fact in facts] == ["1", "2"]
    assert facts.status == "partial"
    assert "safety cap" in (facts.limitation or "")
    assert client.topic_calls == [("bounded", 1), ("bounded", 2)]


def test_max_topics_stops_without_requesting_second_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeCompetitionClient(
        topic_pages={
            1: SimpleNamespace(topics=[_topic(1), _topic(2)], total_count=10),
        },
        messages={1: SimpleNamespace(messages=[])},
    )
    _install_client(monkeypatch, client)

    facts = discussions.fetch_competition_discussions("limited", 1)

    assert [fact.topic_id for fact in facts] == ["1"]
    assert client.topic_calls == [("limited", 1)]


def test_nested_and_duplicate_messages_are_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = _message(2, "<p>Reply</p>")
    root = _message(1, "<p>Root</p>", replies=[reply, reply])
    client = FakeCompetitionClient(
        topic_pages={1: SimpleNamespace(topics=[_topic(10)], total_count=1)},
        messages={10: SimpleNamespace(messages=[root, root])},
    )
    _install_client(monkeypatch, client)

    facts = discussions.fetch_competition_discussions("nested", 10)

    assert [message.message_id for message in facts[0].messages] == ["1", "2"]
    assert facts[0].text == "Root\n\nReply"


def test_one_failed_topic_preserves_other_topics_and_marks_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeCompetitionClient(
        topic_pages={
            1: SimpleNamespace(topics=[_topic(1), _topic(2)], total_count=2),
        },
        messages={
            1: HttpError(503),
            2: SimpleNamespace(messages=[_message(20)]),
        },
    )
    _install_client(monkeypatch, client)

    facts = discussions.fetch_competition_discussions("partial", 10)

    assert facts.status == "partial"
    assert [fact.topic_id for fact in facts] == ["1", "2"]
    assert facts[0].collection_status == "failed"
    assert facts[1].collection_status == "collected"
    assert "secret response body" not in (facts.error or "")
    assert "HTTP 503" in (facts.limitation or "")


@pytest.mark.parametrize(
    ("status_code", "expected_status"),
    [(429, "rate_limited"), (403, "forbidden"), (500, "failed")],
)
def test_initial_topic_error_has_specific_status_and_no_generic_fallback(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    expected_status: str,
) -> None:
    client = FakeCompetitionClient(
        topic_pages={1: HttpError(status_code)},
        messages={},
    )
    _install_client(monkeypatch, client)

    facts = discussions.fetch_competition_discussions("broken", 10)

    assert facts == []
    assert facts.status == expected_status
    assert "secret response body" not in (facts.error or "")


def test_inter_topic_delay_is_injectable(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeCompetitionClient(
        topic_pages={1: SimpleNamespace(topics=[_topic(1), _topic(2), _topic(3)], total_count=3)},
        messages={},
    )
    sleeps: list[float] = []
    monkeypatch.setattr(discussions, "_competition_discussions_client", lambda: client)
    monkeypatch.setattr(discussions, "_between_topics_sleep", sleeps.append)

    discussions.fetch_competition_discussions("paced", 10)

    assert sleeps == [discussions.BETWEEN_TOPICS_DELAY_SECONDS] * 2


def test_html_parser_normalizes_blocks_entities_and_unsafe_links() -> None:
    raw_html = (
        "<p>Hello &amp; welcome<br>Read it<a href='https://medium.com/x'>here</a></p>"
        "<div><a href='/competitions/example'>Kaggle</a></div>"
        "<a href='https://medium.com/x'>duplicate</a>"
        "<a href='javascript:alert(1)'>unsafe</a>"
        "<a href='data:text/plain,bad'>bad</a>"
        "<script>secret()</script><style>.hidden{}</style>"
    )

    text, links = discussions._parse_discussion_html(raw_html)

    assert text == "Hello & welcome\nRead it here\n\nKaggle\nduplicate unsafe bad"
    assert [(link.url, link.kind) for link in links] == [
        ("https://medium.com/x", "external"),
        ("/competitions/example", "relative"),
    ]
    assert links[0].competition_slug is None
    assert links[1].competition_slug == "example"
    assert "secret" not in text
    assert "hidden" not in text


@pytest.mark.parametrize(
    ("href", "expected_slug"),
    [
        ("https://www.kaggle.com/c/happywhale/discussion/12", "happywhale"),
        ("https://www.kaggle.com/competitions/foo-bar", "foo-bar"),
        ("/competitions/foo-bar/data", "foo-bar"),
        ("https://www.kaggle.com/username", None),
    ],
)
def test_competition_slug_is_read_from_kaggle_link_path(
    href: str,
    expected_slug: str | None,
) -> None:
    _, links = discussions._parse_discussion_html(f'<a href="{href}">source</a>')

    assert links[0].competition_slug == expected_slug


@pytest.mark.parametrize(
    ("raw_html", "expected"),
    [
        ("<p>Hello</p><p>World</p>", "Hello\n\nWorld"),
        ("<p>One<br>Two", "One\nTwo"),
        ("<div><strong>Malformed", "Malformed"),
        ("", ""),
    ],
)
def test_html_parser_is_deterministic_and_tolerant(
    raw_html: str,
    expected: str,
) -> None:
    assert discussions._parse_discussion_html(raw_html)[0] == expected
    assert discussions._parse_discussion_html(raw_html)[0] == expected


def test_long_html_is_explicitly_truncated_and_hashes_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(discussions, "MAX_DISCUSSION_HTML_CHARS", 12)
    raw_html = "<p>abcdefghijklmnopqrstuvwxyz</p>"

    result = discussions._normalize_messages(
        [_message(1, raw_html)],
        slug="example",
        topic_id="10",
    )[0]

    assert result.content_truncated is True
    assert result.content_original_length == len(raw_html)
    assert result.content_html == raw_html[:12]
    assert result.content_sha256 == hashlib.sha256(raw_html.encode()).hexdigest()


def test_message_author_uses_confirmed_author_name_only() -> None:
    with_author = _message(1, author_name="actual-author")
    missing_author = {"id": 2, "content": "<p>None</p>", "author": "guessed"}

    facts = discussions._normalize_messages(
        [with_author, missing_author],
        slug="example",
        topic_id="10",
    )

    assert facts[0].author_name == "actual-author"
    assert facts[1].author_name is None
    assert facts[1].author_slug is None
    assert facts[1].author_id is None


@pytest.mark.parametrize(
    ("title", "candidate"),
    [
        ("1st Place Solution", True),
        ("4th Place Solution WriteUp", True),
        ("Winning solution", True),
        ("Model improvement plan", False),
        ("How do I load the data?", False),
        ("SOLUTION", True),
    ],
)
def test_writeup_candidate_classification(title: str, candidate: bool) -> None:
    actual, signals = discussions._writeup_candidate(title)

    assert actual is candidate
    assert bool(signals) is candidate


@pytest.mark.parametrize(
    ("title", "placement_kind", "placement_value"),
    [
        ("Top 5% Solution (0.80143)", "top_percent", 5),
        ("Top 10 % writeup", "top_percent", 10),
        ("1st Place Solution", "rank", 1),
        ("2nd place solution", "rank", 2),
        ("3rd Place", "rank", 3),
        ("12th place", "rank", 12),
        ("#1 solution", "rank", 1),
        ("Rank 4 solution", "rank", 4),
        ("winner", "rank", 1),
        ("Winning solution", "rank", 1),
        ("Solution notes", "unspecified", None),
    ],
)
def test_extract_placement(
    title: str,
    placement_kind: str,
    placement_value: int | None,
) -> None:
    assert discussions.extract_placement(title) == (placement_kind, placement_value)


def test_solution_without_placement_remains_candidate_with_unspecified_placement() -> None:
    fact = discussions._competition_topic_fact(
        _topic(1, "Feature engineering solution"),
        "example",
    )

    assert fact.is_writeup_candidate is True
    assert fact.placement_kind == "unspecified"
    assert fact.placement_value is None


def test_winner_writeups_filter_regular_competition_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def fetch(slug: str, limit: int) -> discussions.DiscussionCollection:
        calls.append((slug, limit))
        return discussions.DiscussionCollection(
            [
                discussions._competition_topic_fact(
                    _topic(1, "Round 1 and 2 solution: 1st place"),
                    slug,
                ),
                discussions._competition_topic_fact(
                    _topic(2, "How do I load the data?"),
                    slug,
                ),
            ],
            status="collected",
        )

    monkeypatch.setattr(discussions, "fetch_competition_discussions", fetch)

    facts = discussions.fetch_winner_writeups(["round-2-jaguar"], 10)

    assert calls == [("round-2-jaguar", 10)]
    assert [fact.topic_id for fact in facts] == ["1"]
    assert facts[0].source_type == "winner_writeup"
    assert facts[0].is_writeup_candidate is True
    assert facts[0].writeup_signals == ["solution", "place", "placement"]
    assert facts[0].placement_kind == "rank"
    assert facts[0].placement_value == 1


def test_zero_writeup_limit_does_not_collect_discussions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        discussions,
        "fetch_competition_discussions",
        lambda slug, limit: pytest.fail("discussion collection must not start"),
    )

    assert discussions.fetch_winner_writeups(["example"], 0) == []


def test_low_level_competition_requests_use_confirmed_fields() -> None:
    class LowLevelClient:
        def __init__(self) -> None:
            self.topic_request: Any = None
            self.message_request: Any = None

        def list_competition_topics(self, request: Any) -> Any:
            self.topic_request = request
            return SimpleNamespace(topics=[], total_count=0)

        def list_topic_messages(self, request: Any) -> Any:
            self.message_request = request
            return SimpleNamespace(messages=[])

    low_level = LowLevelClient()
    client = object.__new__(discussions._KaggleCompetitionDiscussionsClient)
    client._client = low_level
    client._request_policy = discussions.KaggleRequestPolicy(min_interval_seconds=0)

    client.list_topics("jaguar-re-id", page=2)
    client.list_topic_messages("jaguar-re-id", 727017, page_size=-1)

    assert low_level.topic_request.competition_name == "jaguar-re-id"
    assert low_level.topic_request.page == 2
    assert low_level.message_request.competition_name == "jaguar-re-id"
    assert low_level.message_request.topic_id == 727017
    assert low_level.message_request.page_size == -1


@pytest.mark.parametrize(
    ("method_name", "failures"),
    [
        ("list_competition_topics", [429]),
        ("list_topic_messages", [429, 429]),
        ("list_topic_messages", [503]),
    ],
)
def test_low_level_competition_requests_retry_transient_errors(
    method_name: str,
    failures: list[int],
) -> None:
    class LowLevelClient:
        def __init__(self) -> None:
            self.calls = 0

        def _call(self) -> Any:
            self.calls += 1
            if self.calls <= len(failures):
                raise HttpError(failures[self.calls - 1])
            return SimpleNamespace(topics=[], total_count=0, messages=[])

        def list_competition_topics(self, request: Any) -> Any:
            return self._call()

        def list_topic_messages(self, request: Any) -> Any:
            return self._call()

    low_level = LowLevelClient()
    client = object.__new__(discussions._KaggleCompetitionDiscussionsClient)
    client._client = low_level
    client._request_policy = discussions.KaggleRequestPolicy(
        max_attempts=5,
        base_delay_seconds=0,
        min_interval_seconds=0,
    )

    if method_name == "list_competition_topics":
        client.list_topics("example", page=1)
    else:
        client.list_topic_messages("example", 1, page_size=-1)

    assert low_level.calls == len(failures) + 1


def test_403_is_not_retried_and_five_429s_are_bounded() -> None:
    class LowLevelClient:
        def __init__(self, status: int) -> None:
            self.status = status
            self.calls = 0

        def list_competition_topics(self, request: Any) -> Any:
            self.calls += 1
            raise HttpError(self.status)

    for status, expected_calls in ((403, 1), (429, 5)):
        low_level = LowLevelClient(status)
        client = object.__new__(discussions._KaggleCompetitionDiscussionsClient)
        client._client = low_level
        client._request_policy = discussions.KaggleRequestPolicy(
            max_attempts=5,
            base_delay_seconds=0,
            min_interval_seconds=0,
        )
        with pytest.raises(HttpError):
            client.list_topics("example", page=1)
        assert low_level.calls == expected_calls


def test_competition_collector_source_has_no_generic_fallback() -> None:
    source = Path(discussions.__file__).read_text(encoding="utf-8")
    collector_source = source.split("def _collect_competition_topics", 1)[1].split(
        "def _competition_topic_fact", 1
    )[0]

    assert "discussion_api_client" not in collector_source
    assert "forum_slug" not in collector_source
    assert "list_forums" not in collector_source
    assert "TopicListCategory" not in source
    assert "competition_write_ups" not in source


def test_zero_limit_does_not_create_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        discussions,
        "_competition_discussions_client",
        lambda: pytest.fail("client must not be created"),
    )

    assert discussions.fetch_competition_discussions("example", 0) == []


def test_module_does_not_scrape_or_download_external_links() -> None:
    source = Path(discussions.__file__).read_text(encoding="utf-8")

    assert "BeautifulSoup" not in source
    assert "selenium" not in source.lower()
    assert "playwright" not in source.lower()
    assert "requests.get" not in source
