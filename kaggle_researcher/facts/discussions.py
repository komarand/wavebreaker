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
from typing import Any, Literal
from urllib.parse import urlparse

from kaggle_researcher.facts.kaggle_api import (
    GLOBAL_KAGGLE_POLICY,
    extract_http_status,
)
from kaggle_researcher.facts.kaggle_api import (
    KaggleRequestPolicy as KaggleRequestPolicy,
)
from kaggle_researcher.facts.models import (
    DiscussionFacts,
    DiscussionLink,
    DiscussionMessageFacts,
)

LOGGER = logging.getLogger(__name__)
TOPICS_PAGE_SAFETY_CAP = 100
MAX_DISCUSSION_HTML_CHARS = 200_000
MESSAGE_PAGE_SIZE_ALL = -1
BETWEEN_TOPICS_DELAY_SECONDS = 0.4
_between_topics_sleep = time.sleep
_COMPETITION_REQUEST_POLICY = GLOBAL_KAGGLE_POLICY
COMPETITION_PATH = re.compile(
    r"^/(?:c|competitions)/(?P<slug>[a-z0-9][a-z0-9-]{1,80})(?:/|$)"
)
PLACEMENT_PATTERN = re.compile(
    r"\btop\s*(?P<top_percent>\d{1,3})\s*%"
    r"|(?<!\w)#\s*(?P<hash_rank>\d+)\b"
    r"|\brank\s*(?P<named_rank>\d+)\b"
    r"|\b(?P<ordinal_rank>\d+)(?:st|nd|rd|th)(?:\s+place)?\b"
    r"|\bwinn(?:er|ing)(?:\s+solution)?\b",
    re.IGNORECASE,
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
        discussions = fetch_competition_discussions(slug, per_competition)
        candidates = (
            discussion
            for discussion in discussions
            if discussion.is_writeup_candidate
        )
        writeups.extend(
            discussion.model_copy(update={"source_type": "winner_writeup"})
            for discussion in candidates
        )
    return writeups


def extract_placement(title: str) -> tuple[str | None, int | None]:
    match = PLACEMENT_PATTERN.search(title)
    if match is None:
        return "unspecified", None
    if match.group("top_percent") is not None:
        return "top_percent", int(match.group("top_percent"))
    for group_name in ("hash_rank", "named_rank", "ordinal_rank"):
        value = match.group(group_name)
        if value is not None:
            return "rank", int(value)
    return "rank", 1


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
    placement_kind, placement_value = extract_placement(title)
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
        placement_kind=placement_kind,
        placement_value=placement_value,
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
            url, kind, competition_slug = classified
            if url not in self._seen_urls:
                self._seen_urls.add(url)
                text = _normalize_inline_text("".join(self._active_link_text)) or None
                self.links.append(
                    DiscussionLink(
                        url=url,
                        text=text,
                        kind=kind,
                        competition_slug=competition_slug,
                    )
                )
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
) -> tuple[str, Literal["kaggle", "external", "relative"], str | None] | None:
    href = unescape(raw_href).strip()
    if not href:
        return None
    parsed = urlparse(href)
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        hostname = (parsed.hostname or "").lower()
        kind: Literal["kaggle", "external", "relative"] = (
            "kaggle" if hostname == "kaggle.com" or hostname.endswith(".kaggle.com") else "external"
        )
        return href, kind, _competition_slug(parsed.path) if kind == "kaggle" else None
    if parsed.scheme or parsed.netloc:
        return None
    return href, "relative", _competition_slug(parsed.path)


def _competition_slug(path: str) -> str | None:
    match = COMPETITION_PATH.match(path.lower())
    return match.group("slug") if match is not None else None


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


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
