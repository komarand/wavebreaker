from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from urllib.parse import urlparse

from kaggle_researcher.facts.kaggle_api import (
    KaggleRequestPolicy,
    extract_http_status,
)
from kaggle_researcher.facts.models import (
    DiscussionFacts,
    DiscussionLink,
    DiscussionMessageFacts,
)

LOGGER = logging.getLogger(__name__)
PAGE_SIZE = 200
TOPICS_PAGE_SAFETY_CAP = 100
MAX_DISCUSSION_HTML_CHARS = 200_000
MESSAGE_PAGE_SIZE_ALL = -1
BETWEEN_TOPICS_DELAY_SECONDS = 0.4
_between_topics_sleep = time.sleep
_FORUMS_REQUEST_POLICY = KaggleRequestPolicy(min_interval_seconds=0.5)
_COMPETITION_REQUEST_POLICY = KaggleRequestPolicy(
    max_attempts=5,
    min_interval_seconds=0.4,
    jitter_fraction=0.1,
)
DiscussionStatus = Literal[
    "collected",
    "partial",
    "empty",
    "rate_limited",
    "forbidden",
    "unavailable",
    "failed",
]


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
    return _collect_competition_topics(slug=slug, limit=max_topics)


def fetch_winner_writeups(
    competition_slugs: list[str],
    per_competition: int,
) -> list[DiscussionFacts]:
    if per_competition <= 0:
        return []

    writeups: list[DiscussionFacts] = []
    for slug in competition_slugs:
        writeups.extend(
            _collect_forum_topics(
                slug=slug,
                limit=per_competition,
                source_type="winner_writeup",
                category="competition_write_ups",
                sort_by="top",
                sort_by_votes=True,
            )
        )
    return writeups


def _collect_competition_topics(*, slug: str, limit: int) -> DiscussionCollection:
    try:
        client = _competition_discussions_client()
    except Exception as exc:
        error = _discussion_error("unavailable", exc)
        LOGGER.warning(
            "Competition discussion client is unavailable (%s)",
            type(exc).__name__,
        )
        return DiscussionCollection(status="unavailable", error=error)

    topic_summaries: list[Any] = []
    seen_topic_ids: set[str] = set()
    seen_pages: set[int] = set()
    listing_failure: tuple[DiscussionStatus, str] | None = None
    reported_total_count: int | None = None
    page = 1

    while len(topic_summaries) < limit and len(seen_pages) < TOPICS_PAGE_SAFETY_CAP:
        if page in seen_pages:
            break
        seen_pages.add(page)
        try:
            response = client.list_topics(slug, page=page)
        except Exception as exc:
            status = _exception_status(exc)
            error = _competition_operation_error("topic listing", exc)
            listing_failure = (status, error)
            LOGGER.warning(
                "Failed to list competition discussion topics for %s on page %s " "(%s, HTTP %s)",
                slug,
                page,
                type(exc).__name__,
                extract_http_status(exc),
            )
            break

        page_topics = list(_items(response, "topics"))
        if not page_topics:
            break

        new_topic_ids = 0
        for topic in page_topics:
            topic_id = _optional_text(_value(topic, "id", "topic_id", "topicId"))
            if topic_id is None or topic_id in seen_topic_ids:
                continue
            seen_topic_ids.add(topic_id)
            topic_summaries.append(topic)
            new_topic_ids += 1
            if len(topic_summaries) >= limit:
                break

        if len(topic_summaries) >= limit or new_topic_ids == 0:
            break
        reported_total_count = _integer(_value(response, "total_count", "totalCount"))
        if reported_total_count is not None and len(seen_topic_ids) >= reported_total_count:
            break
        page += 1

    if (
        listing_failure is None
        and len(seen_pages) >= TOPICS_PAGE_SAFETY_CAP
        and len(topic_summaries) < limit
        and (reported_total_count is None or len(seen_topic_ids) < reported_total_count)
    ):
        listing_failure = (
            "failed",
            "Kaggle competition discussions topic listing reached the page safety cap.",
        )

    if not topic_summaries:
        if listing_failure is None:
            return DiscussionCollection(status="empty")
        status, error = listing_failure
        return DiscussionCollection(
            status=status,
            error=error,
            limitation=error if status in {"rate_limited", "forbidden"} else None,
        )

    facts: list[DiscussionFacts] = []
    topic_failures: list[tuple[DiscussionStatus, str]] = []
    for index, summary in enumerate(topic_summaries):
        if index:
            _between_topics_sleep(BETWEEN_TOPICS_DELAY_SECONDS)
        fact = _competition_topic_fact(summary, slug)
        try:
            response = client.list_topic_messages(
                slug,
                int(fact.topic_id),
                page_size=MESSAGE_PAGE_SIZE_ALL,
            )
        except Exception as exc:
            status = _exception_status(exc)
            error = _competition_operation_error(
                f"messages for topic {fact.topic_id}",
                exc,
            )
            fact.collection_status = _topic_status(status)
            fact.collection_error = error
            topic_failures.append((status, error))
            LOGGER.warning(
                "Failed to list messages for competition discussion topic %s " "(%s, HTTP %s)",
                fact.topic_id,
                type(exc).__name__,
                extract_http_status(exc),
            )
            facts.append(fact)
            continue

        try:
            messages = _normalize_messages(
                _items(response, "messages"),
                slug=slug,
                topic_id=fact.topic_id,
            )
        except Exception as exc:
            error = (
                f"Kaggle competition discussions messages for topic "
                f"{fact.topic_id} failed ({type(exc).__name__})."
            )
            fact.collection_status = "failed"
            fact.collection_error = error
            topic_failures.append(("failed", error))
            LOGGER.warning(
                "Failed to normalize messages for competition discussion topic %s " "(%s)",
                fact.topic_id,
                type(exc).__name__,
            )
            facts.append(fact)
            continue
        fact.messages = messages
        fact.text = "\n\n".join(
            message.content_text for message in messages if message.content_text
        )
        fact.collection_status = "collected" if messages else "empty"
        facts.append(fact)

    failures = ([listing_failure] if listing_failure is not None else []) + topic_failures
    if not failures:
        return DiscussionCollection(facts, status="collected")

    successful_topics = any(fact.collection_status in {"collected", "empty"} for fact in facts)
    failure_statuses = {status for status, _ in failures}
    if successful_topics:
        status: DiscussionStatus = "partial"
    elif failure_statuses == {"rate_limited"}:
        status = "rate_limited"
    elif failure_statuses == {"forbidden"}:
        status = "forbidden"
    else:
        status = "failed"
    error = "; ".join(dict.fromkeys(error for _, error in failures))
    limitation = f"Competition discussion corpus is partial: {error}"
    return DiscussionCollection(
        facts,
        status=status,
        error=error,
        limitation=limitation,
    )


def _competition_topic_fact(topic: Any, slug: str) -> DiscussionFacts:
    topic_id = _required_text(
        _value(topic, "id", "topic_id", "topicId"),
        "topic id",
    )
    title = _optional_text(_value(topic, "title")) or ""
    raw_url = _optional_text(_value(topic, "topic_url", "topicUrl"))
    if raw_url:
        url = f"https://www.kaggle.com{raw_url}" if raw_url.startswith("/") else raw_url
        url_constructed = False
    else:
        url = f"https://www.kaggle.com/competitions/{slug}/discussion/{topic_id}"
        url_constructed = True
    is_candidate, signals = _writeup_candidate(title)
    return DiscussionFacts(
        topic_id=topic_id,
        title=title,
        author=_optional_text(_value(topic, "author_name", "authorName")),
        author_is_host=None,
        votes=_integer(_value(topic, "votes")) or 0,
        created_at=_value(topic, "post_date", "postDate"),
        updated_at=_value(
            topic,
            "last_comment_post_date",
            "lastCommentPostDate",
        ),
        source_type="discussion",
        competition_id=slug,
        evidence_id=f"discussion-topic:{slug}:{topic_id}",
        url=url,
        url_constructed=url_constructed,
        comment_count=_integer(_value(topic, "comment_count", "commentCount")),
        is_writeup_candidate=is_candidate,
        writeup_signals=signals,
        collection_status="empty",
    )


def _normalize_messages(
    messages: Iterable[Any],
    *,
    slug: str,
    topic_id: str,
) -> list[DiscussionMessageFacts]:
    normalized: list[DiscussionMessageFacts] = []
    seen_message_ids: set[str] = set()
    for message in _flatten_messages(messages):
        message_id = _optional_text(_value(message, "id", "message_id", "messageId"))
        if message_id is None or message_id in seen_message_ids:
            continue
        seen_message_ids.add(message_id)
        raw_html = _text(_value(message, "content"))
        content_original_length = len(raw_html)
        content_truncated = content_original_length > MAX_DISCUSSION_HTML_CHARS
        stored_html = raw_html[:MAX_DISCUSSION_HTML_CHARS]
        content_text, links = _parse_discussion_html(stored_html)
        normalized.append(
            DiscussionMessageFacts(
                evidence_id=(f"discussion-message:{slug}:{topic_id}:{message_id}"),
                message_id=message_id,
                topic_id=topic_id,
                author_name=_optional_text(_value(message, "author_name", "authorName")),
                created_at=_value(message, "post_date", "postDate"),
                votes=_integer(_value(message, "votes")),
                content_html=stored_html,
                content_text=content_text,
                content_sha256=hashlib.sha256(raw_html.encode("utf-8")).hexdigest(),
                content_truncated=content_truncated,
                content_original_length=content_original_length,
                links=links,
            )
        )
    return normalized


def _flatten_messages(messages: Iterable[Any]) -> Iterable[Any]:
    stack = list(reversed(list(messages)))
    seen_objects: set[int] = set()
    while stack and len(seen_objects) < 10_000:
        message = stack.pop()
        object_id = id(message)
        if object_id in seen_objects:
            continue
        seen_objects.add(object_id)
        yield message
        replies = _value(message, "replies")
        if replies:
            stack.extend(reversed(list(replies)))


def _writeup_candidate(title: str) -> tuple[bool, list[str]]:
    signal_patterns = (
        ("solution", r"\bsolution\b"),
        ("writeup", r"\bwrite[ -]?up\b"),
        ("winner", r"\bwinn(?:er|ing)\b"),
        ("place", r"\bplace\b"),
        ("placement", r"\b\d+(?:st|nd|rd|th)\b"),
    )
    signals = [name for name, pattern in signal_patterns if re.search(pattern, title, re.I)]
    return bool(signals), signals


def _exception_status(exc: BaseException) -> DiscussionStatus:
    status_code = extract_http_status(exc)
    if status_code == 429:
        return "rate_limited"
    if status_code == 403:
        return "forbidden"
    return "failed"


def _topic_status(status: DiscussionStatus) -> Literal["rate_limited", "forbidden", "failed"]:
    if status == "rate_limited":
        return "rate_limited"
    if status == "forbidden":
        return "forbidden"
    return "failed"


def _competition_operation_error(operation: str, exc: BaseException) -> str:
    status_code = extract_http_status(exc)
    if status_code is not None:
        return f"Kaggle competition discussions {operation} returned HTTP {status_code}."
    return f"Kaggle competition discussions {operation} failed " f"({type(exc).__name__})."


def _collect_forum_topics(
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
            status: DiscussionStatus = "forbidden" if extract_http_status(exc) == 403 else "failed"
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
    if (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) or (
        kaggle_dir / "kaggle.json"
    ).is_file():
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


class _DiscussionHtmlParser(HTMLParser):
    _BLOCK_TAGS = frozenset(
        {
            "p",
            "div",
            "br",
            "li",
            "ul",
            "ol",
            "blockquote",
            "pre",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }
    )
    _IGNORED_TAGS = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.links: list[DiscussionLink] = []
        self._seen_urls: set[str] = set()
        self._ignored_depth = 0
        self._active_href: str | None = None
        self._active_link_text: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag in self._BLOCK_TAGS:
            self.text_parts.append("\n")
        if tag == "a":
            self._finish_link()
            self.text_parts.append(" ")
            self._active_href = next(
                (value for name, value in attrs if name.lower() == "href"),
                None,
            )
            self._active_link_text = []

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if self._ignored_depth:
            return
        if tag == "a":
            self._finish_link()
            self.text_parts.append(" ")
        if tag in self._BLOCK_TAGS and tag != "br":
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        self.text_parts.append(data)
        if self._active_href is not None:
            self._active_link_text.append(data)

    def finish(self) -> None:
        self._finish_link()

    def _finish_link(self) -> None:
        if self._active_href is None:
            return
        classified = _classify_link(self._active_href)
        if classified is not None:
            url, kind = classified
            if url not in self._seen_urls:
                self._seen_urls.add(url)
                text = _normalize_inline_text("".join(self._active_link_text)) or None
                self.links.append(DiscussionLink(url=url, text=text, kind=kind))
        self._active_href = None
        self._active_link_text = []


def _parse_discussion_html(raw_html: str) -> tuple[str, list[DiscussionLink]]:
    parser = _DiscussionHtmlParser()
    try:
        parser.feed(raw_html)
        parser.close()
    except Exception:
        LOGGER.warning("Malformed discussion HTML could not be parsed completely")
    parser.finish()
    return _normalize_block_text("".join(parser.text_parts)), parser.links


def _classify_link(
    raw_href: str,
) -> tuple[str, Literal["kaggle", "external", "relative"]] | None:
    href = unescape(raw_href).strip()
    if not href:
        return None
    parsed = urlparse(href)
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        hostname = (parsed.hostname or "").lower()
        kind: Literal["kaggle", "external", "relative"] = (
            "kaggle" if hostname == "kaggle.com" or hostname.endswith(".kaggle.com") else "external"
        )
        return href, kind
    if parsed.scheme or parsed.netloc:
        return None
    return href, "relative"


def _normalize_inline_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_block_text(value: str) -> str:
    normalized_lines: list[str] = []
    previous_blank = True
    for raw_line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[\t\f\v ]+", " ", raw_line.replace("\xa0", " ")).strip()
        if line:
            normalized_lines.append(line)
            previous_blank = False
        elif not previous_blank:
            normalized_lines.append("")
            previous_blank = True
    while normalized_lines and not normalized_lines[-1]:
        normalized_lines.pop()
    return "\n".join(normalized_lines)


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


class _KaggleCompetitionDiscussionsClient:
    def __init__(self) -> None:
        from kagglesdk import KaggleClient, KaggleEnv

        self._kaggle = KaggleClient(
            env=KaggleEnv.PROD,
            username=os.getenv("KAGGLE_USERNAME"),
            password=os.getenv("KAGGLE_KEY"),
            api_token=os.getenv("KAGGLE_API_TOKEN") or _stored_oauth_access_token(),
        )
        self._client = self._kaggle.competitions.competition_api_client
        self._request_policy = _COMPETITION_REQUEST_POLICY

    def list_topics(
        self,
        competition_slug: str,
        *,
        page: int | None = None,
    ) -> Any:
        from kagglesdk.competitions.types.competition_api_service import (
            ApiListCompetitionTopicsRequest,
        )

        request = ApiListCompetitionTopicsRequest()
        request.competition_name = competition_slug
        if page is not None:
            request.page = page
        return self._request(
            "list_competition_topics",
            lambda: self._client.list_competition_topics(request),
        )

    def list_topic_messages(
        self,
        competition_slug: str,
        topic_id: int,
        *,
        page_size: int | None = None,
    ) -> Any:
        from kagglesdk.competitions.types.competition_api_service import (
            ApiListTopicMessagesRequest,
        )

        request = ApiListTopicMessagesRequest()
        request.competition_name = competition_slug
        request.topic_id = topic_id
        if page_size is not None:
            request.page_size = page_size
        return self._request(
            "list_topic_messages",
            lambda: self._client.list_topic_messages(request),
        )

    def _request(self, operation_name: str, operation: Any) -> Any:
        policy = getattr(self, "_request_policy", _COMPETITION_REQUEST_POLICY)

        def logged_operation() -> Any:
            try:
                return operation()
            except Exception as exc:
                LOGGER.warning(
                    "Kaggle competition discussion operation %s failed " "(%s, HTTP %s)",
                    operation_name,
                    type(exc).__name__,
                    extract_http_status(exc),
                )
                raise

        return policy.call(logged_operation)


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
            request.category = TopicListCategory[f"TOPIC_LIST_CATEGORY_{category.upper()}"]
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


def _competition_discussions_client() -> Any:
    return _KaggleCompetitionDiscussionsClient()


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
