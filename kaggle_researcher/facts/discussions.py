from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from kaggle_researcher.facts.kaggle_api import (
    KaggleRequestPolicy,
    extract_http_status,
)
from kaggle_researcher.facts.models import DiscussionFacts

LOGGER = logging.getLogger(__name__)
PAGE_SIZE = 200
_FORUMS_REQUEST_POLICY = KaggleRequestPolicy(min_interval_seconds=0.5)
DiscussionStatus = Literal["collected", "empty", "forbidden", "unavailable", "failed"]


class DiscussionCollection(list[DiscussionFacts]):
    def __init__(
        self,
        items: Iterable[DiscussionFacts] = (),
        *,
        status: DiscussionStatus,
        error: str | None = None,
        limitation: str | None = None,
    ) -> None:
        super().__init__(items)
        self.status = status
        self.error = error
        self.limitation = limitation


def fetch_competition_discussions(
    slug: str,
    max_topics: int,
) -> DiscussionCollection:
    if max_topics <= 0:
        return DiscussionCollection(status="empty")
    return _collect_topics(
        slug=slug,
        limit=max_topics,
        source_type="discussion",
    )


def fetch_winner_writeups(
    competition_slugs: list[str],
    per_competition: int,
) -> list[DiscussionFacts]:
    if per_competition <= 0:
        return []

    writeups: list[DiscussionFacts] = []
    for slug in competition_slugs:
        writeups.extend(
            _collect_topics(
                slug=slug,
                limit=per_competition,
                source_type="winner_writeup",
                category="competition_write_ups",
                sort_by="top",
                sort_by_votes=True,
            )
        )
    return writeups


def _collect_topics(
    *,
    slug: str,
    limit: int,
    source_type: Literal["discussion", "winner_writeup"],
    category: str | None = None,
    sort_by: str | None = None,
    sort_by_votes: bool = False,
) -> DiscussionCollection:
    try:
        client = _forums_client()
    except Exception as exc:
        error = _discussion_error("unavailable", exc)
        LOGGER.warning("Discussion client is unavailable (%s)", type(exc).__name__)
        return DiscussionCollection(status="unavailable", error=error)

    topic_summaries: list[Any] = []
    page_token: str | None = None

    while len(topic_summaries) < limit:
        try:
            response = client.list_topics(
                forum_slug=slug,
                page_size=min(PAGE_SIZE, limit - len(topic_summaries)),
                page_token=page_token,
                category=category,
                sort_by=sort_by,
            )
        except Exception as exc:
            status: DiscussionStatus = (
                "forbidden" if extract_http_status(exc) == 403 else "failed"
            )
            error = _discussion_error(status, exc)
            LOGGER.warning(
                "Failed to list discussion topics for %s (%s)",
                slug,
                type(exc).__name__,
            )
            return DiscussionCollection(
                status=status,
                error=error,
                limitation=error if status == "forbidden" else None,
            )

        page_topics = list(_items(response, "topics"))
        topic_summaries.extend(page_topics[: limit - len(topic_summaries)])
        next_page_token = _text(_value(response, "next_page_token", "nextPageToken"))
        if not next_page_token or next_page_token == page_token or not page_topics:
            break
        page_token = next_page_token

    if sort_by_votes:
        topic_summaries.sort(
            key=lambda topic: _integer(_value(topic, "votes")) or -1,
            reverse=True,
        )

    facts: list[DiscussionFacts] = []
    topic_failures = 0
    for topic_summary in topic_summaries[:limit]:
        topic_id = _text(_value(topic_summary, "id", "topic_id", "topicId"))
        if not topic_id:
            continue
        try:
            thread = client.get_topic(topic_id)
            facts.append(
                _to_discussion_fact(
                    summary=topic_summary,
                    thread=thread,
                    source_type=source_type,
                    competition_id=slug,
                )
            )
        except Exception as exc:
            topic_failures += 1
            detail = f": {exc}" if isinstance(exc, ValueError) else ""
            LOGGER.warning(
                "Failed to fetch discussion topic %s (%s)%s",
                topic_id,
                type(exc).__name__,
                detail,
            )

    if facts:
        return DiscussionCollection(facts, status="collected")
    if topic_summaries and topic_failures:
        return DiscussionCollection(
            status="failed",
            error="Kaggle Discussion API topic retrieval failed.",
        )
    return DiscussionCollection(status="empty")


def discussion_auth_mode() -> Literal["legacy", "oauth", "unknown"]:
    kaggle_dir = Path.home() / ".kaggle"
    if (
        os.getenv("KAGGLE_API_TOKEN")
        or (kaggle_dir / "credentials.json").is_file()
        or (kaggle_dir / "access_token").is_file()
        or (kaggle_dir / "access_token.txt").is_file()
    ):
        return "oauth"
    if (
        (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
        or (kaggle_dir / "kaggle.json").is_file()
    ):
        return "legacy"
    return "unknown"


def _discussion_error(status: DiscussionStatus, exc: BaseException) -> str:
    status_code = extract_http_status(exc)
    if status == "forbidden":
        return "Kaggle Discussion API returned HTTP 403."
    if status_code is not None:
        return f"Kaggle Discussion API returned HTTP {status_code}."
    if status == "unavailable":
        return "Kaggle Discussion API client is unavailable."
    return f"Kaggle Discussion API collection failed ({type(exc).__name__})."


def _to_discussion_fact(
    *,
    summary: Any,
    thread: Any,
    source_type: Literal["discussion", "winner_writeup"],
    competition_id: str,
) -> DiscussionFacts:
    thread_topic, comments = _thread_parts(thread)
    topic = thread_topic if thread_topic is not None else summary
    topic_id = _required_text(
        _first_value(topic, summary, names=("id", "topic_id", "topicId")),
        "topic id",
    )
    title = _required_text(_first_value(topic, summary, names=("title",)), "title")
    author = _optional_text(
        _first_value(topic, summary, names=("author_name", "authorName", "author"))
    )
    author_is_host = _required_boolean(
        _first_value(
            topic,
            summary,
            names=("author_is_host", "authorIsHost", "is_author_host", "isAuthorHost"),
        ),
        "author_is_host",
    )
    votes = _required_integer(
        _first_value(topic, summary, names=("votes", "vote_count", "voteCount")),
        "votes",
    )
    created_at = _first_value(
        topic,
        summary,
        names=("created_at", "createdAt", "post_date", "postDate"),
    )
    text_parts = [
        text
        for text in (
            _optional_text(_value(topic, "content", "text", "message")),
            *_comment_texts(comments),
        )
        if text
    ]
    return DiscussionFacts(
        topic_id=topic_id,
        title=title,
        author=author,
        author_is_host=author_is_host,
        votes=votes,
        created_at=created_at,
        source_type=source_type,
        competition_id=competition_id,
        text="\n\n".join(text_parts),
    )


def _thread_parts(thread: Any) -> tuple[Any, list[Any]]:
    if isinstance(thread, tuple):
        topic = thread[0] if thread else None
        comments = thread[1] if len(thread) > 1 else []
        return topic, list(comments or [])
    topic = _value(thread, "topic")
    comments = _value(thread, "comments", "messages")
    return topic, list(comments or [])


def _comment_texts(comments: Iterable[Any]) -> list[str]:
    texts: list[str] = []
    for comment in comments:
        text = _optional_text(_value(comment, "content", "text", "message"))
        if text:
            texts.append(text)
        replies = _value(comment, "replies", "children")
        if replies:
            texts.extend(_comment_texts(replies))
    return texts


def _items(response: Any, name: str) -> Iterable[Any]:
    value = _value(response, name)
    return value if isinstance(value, Iterable) and not isinstance(value, str | bytes) else []


def _first_value(primary: Any, fallback: Any, *, names: tuple[str, ...]) -> Any:
    primary_value = _value(primary, *names)
    return primary_value if primary_value is not None else _value(fallback, *names)


def _value(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        try:
            attribute = getattr(value, name)
        except (AttributeError, TypeError):
            continue
        if attribute is not None:
            return attribute
    return None


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _optional_text(value: Any) -> str | None:
    text = _text(value).strip()
    return text or None


def _required_text(value: Any, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"Discussion topic is missing {field_name}")
    return text


def _required_boolean(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"Discussion topic is missing valid {field_name}")


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _required_integer(value: Any, field_name: str) -> int:
    number = _integer(value)
    if number is None:
        raise ValueError(f"Discussion topic is missing valid {field_name}")
    return number


class _KaggleForumsClient:
    def __init__(self) -> None:
        from kagglesdk import KaggleClient, KaggleEnv

        self._kaggle = KaggleClient(
            env=KaggleEnv.PROD,
            username=os.getenv("KAGGLE_USERNAME"),
            password=os.getenv("KAGGLE_KEY"),
            api_token=os.getenv("KAGGLE_API_TOKEN") or _stored_oauth_access_token(),
        )
        self._client = self._kaggle.discussions.discussion_api_client
        self._request_policy = _FORUMS_REQUEST_POLICY

    def list_topics(
        self,
        *,
        forum_slug: str,
        page_size: int,
        page_token: str | None,
        category: str | None,
        sort_by: str | None,
    ) -> Any:
        from kagglesdk.discussions.types.discussions_api_service import ApiListTopicsRequest
        from kagglesdk.discussions.types.discussions_enums import (
            TopicListCategory,
            TopicListSortBy,
        )

        request = ApiListTopicsRequest()
        request.forum_slug = forum_slug
        request.page_size = page_size
        if page_token:
            request.page_token = page_token
        if category:
            request.category = TopicListCategory[
                f"TOPIC_LIST_CATEGORY_{category.upper()}"
            ]
        if sort_by:
            request.sort_by = TopicListSortBy[f"TOPIC_LIST_SORT_BY_{sort_by.upper()}"]
        return self._request(lambda: self._client.list_topics(request))

    def get_topic(self, topic_id: str) -> Any:
        from kagglesdk.discussions.types.discussions_api_service import (
            ApiGetTopicRequest,
            ApiListCommentsRequest,
        )

        get_request = ApiGetTopicRequest()
        get_request.id = int(topic_id)
        topic_response = self._request(lambda: self._client.get_topic(get_request))

        comments: list[Any] = []
        page_token: str | None = None
        while True:
            comments_request = ApiListCommentsRequest()
            comments_request.topic_id = int(topic_id)
            comments_request.page_size = PAGE_SIZE
            if page_token:
                comments_request.page_token = page_token
            response = self._request(
                lambda request=comments_request: self._client.list_comments(request)
            )
            comments.extend(response.comments or [])
            next_page_token = _text(response.next_page_token)
            if not next_page_token or next_page_token == page_token:
                break
            page_token = next_page_token
        return SimpleNamespace(topic=topic_response.topic, comments=comments)

    def _request(self, operation: Any) -> Any:
        policy = getattr(self, "_request_policy", _FORUMS_REQUEST_POLICY)
        return policy.call(operation)


def _forums_client() -> Any:
    return _KaggleForumsClient()


def _stored_oauth_access_token() -> str | None:
    credentials_path = Path.home() / ".kaggle" / "credentials.json"
    try:
        payload = json.loads(credentials_path.read_text(encoding="utf-8"))
        token = payload.get("access_token")
        expiration = payload.get("access_token_expiration")
        if not isinstance(token, str) or not token or not isinstance(expiration, str):
            return None
        expires_at = datetime.fromisoformat(expiration)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        return token if expires_at > datetime.now(timezone.utc) else None
    except (OSError, ValueError, TypeError):
        return None
