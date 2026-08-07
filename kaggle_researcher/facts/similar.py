from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from kaggle_researcher.facts.competition import fetch_competition_metadata
from kaggle_researcher.facts.models import (
    CompetitionMetadata,
    DiscussionFacts,
    SimilarCompetition,
    SimilarityEvidence,
    SimilarSearchDiagnostics,
)
from kaggle_researcher.facts.notebook_ast import canonicalize_metric_label

BARE_SLUG = re.compile(
    r"(?<![\w/-])(?:kaggle\.com/)?(?:c|competitions)/"
    r"(?P<slug>[a-z0-9][a-z0-9-]{2,80})(?![\w-])",
    re.IGNORECASE,
)
_METADATA_FIELDS = frozenset(
    {
        "title",
        "metric_name",
        "is_code_competition",
        "submissions_per_day",
        "max_team_size",
        "deadline",
        "reward",
        "category",
        "num_teams",
    }
)
_VERIFICATION_ORDER = {
    "verified": 0,
    "unchecked": 1,
    "rejected": 2,
    "not_found": 3,
}
_MetadataFetcher = Callable[..., CompetitionMetadata]


def collect_mentions(
    discussions: list[DiscussionFacts],
    competition_id: str,
) -> list[SimilarCompetition]:
    competition_key = competition_id.casefold()
    topic_ids: dict[str, set[str]] = {}
    mention_counts: dict[str, int] = {}

    def record(slug: str, topic_id: str) -> None:
        normalized = slug.casefold()
        if normalized == competition_key:
            return
        topic_ids.setdefault(normalized, set()).add(topic_id)
        mention_counts[normalized] = mention_counts.get(normalized, 0) + 1

    for discussion in discussions:
        topic_id = str(discussion.topic_id)
        for message in discussion.messages:
            for link in message.links:
                if link.competition_slug:
                    record(link.competition_slug, topic_id)
        for text in _discussion_texts(discussion):
            for match in BARE_SLUG.finditer(text):
                record(match.group("slug"), topic_id)

    candidates = [
        SimilarCompetition(
            slug=slug,
            discovered_by="discussion_mention",
            evidence_topic_ids=sorted(topic_ids[slug]),
            mention_topic_count=len(topic_ids[slug]),
            mention_total=mention_counts[slug],
        )
        for slug in topic_ids
    ]
    return sorted(candidates, key=lambda item: (-item.mention_topic_count, item.slug))


def verify_candidate(
    candidate: SimilarCompetition,
    self_metadata: CompetitionMetadata,
    self_submission_columns: list[str],
    api: Any | None = None,
    *,
    metadata_fetcher: _MetadataFetcher | None = None,
) -> SimilarCompetition:
    del self_submission_columns
    fetcher = metadata_fetcher or fetch_competition_metadata
    candidate_metadata = (
        fetcher(candidate.slug, api) if api is not None else fetcher(candidate.slug)
    )
    if _metadata_not_found(candidate_metadata):
        return candidate.model_copy(
            update={
                "title": candidate_metadata.title,
                "verification": "not_found",
                "rejection_reason": "competition not found",
            }
        )

    metric_self = canonicalize_metric_label(self_metadata.metric_name)
    metric_candidate = canonicalize_metric_label(candidate_metadata.metric_name)
    same_metric = _same_metric(
        self_metadata.metric_name,
        candidate_metadata.metric_name,
        metric_self,
        metric_candidate,
    )
    same_code_competition = _same_optional(
        self_metadata.is_code_competition,
        candidate_metadata.is_code_competition,
    )
    same_category = _same_optional_text(
        self_metadata.category,
        candidate_metadata.category,
    )
    feature_values = {
        "same_metric": same_metric,
        "same_code_competition": same_code_competition,
        "same_category": same_category,
    }
    evidence = SimilarityEvidence(
        same_metric=same_metric,
        same_submission_shape=None,
        same_code_competition=same_code_competition,
        same_category=same_category,
        matched_features=[name for name, value in feature_values.items() if value is True],
        metric_self=metric_self,
        metric_candidate=metric_candidate,
    )
    if same_metric is True:
        verification = "verified"
        rejection_reason = None
    else:
        verification = "rejected"
        rejection_reason = _metric_rejection_reason(
            self_metadata.metric_name,
            candidate_metadata.metric_name,
            metric_self,
            metric_candidate,
        )
    return candidate.model_copy(
        update={
            "title": candidate_metadata.title,
            "verification": verification,
            "evidence": evidence,
            "rejection_reason": rejection_reason,
        }
    )


def build_similar_candidates(
    discussions: list[DiscussionFacts],
    self_metadata: CompetitionMetadata,
    self_submission_columns: list[str],
    manual: list[str],
    max_verifications: int,
    api: Any | None = None,
    *,
    metadata_fetcher: _MetadataFetcher | None = None,
) -> tuple[list[SimilarCompetition], SimilarSearchDiagnostics]:
    if max_verifications < 0:
        raise ValueError("max_verifications must not be negative")

    mentioned = collect_mentions(discussions, self_metadata.competition_id)
    by_slug = {candidate.slug: candidate for candidate in mentioned}
    manual_order: list[str] = []
    for raw_slug in manual:
        slug = raw_slug.strip().casefold()
        if not slug or slug == self_metadata.competition_id.casefold():
            continue
        if slug not in manual_order:
            manual_order.append(slug)
        existing = by_slug.get(slug)
        if existing is None:
            by_slug[slug] = SimilarCompetition(
                slug=slug,
                discovered_by="manual",
                confirmed=True,
            )
        else:
            by_slug[slug] = existing.model_copy(
                update={"discovered_by": "manual", "confirmed": True}
            )

    metadata_lookups = 0
    for slug in manual_order:
        by_slug[slug] = verify_candidate(
            by_slug[slug],
            self_metadata,
            self_submission_columns,
            api,
            metadata_fetcher=metadata_fetcher,
        )
        metadata_lookups += 1

    automatic = [candidate for candidate in mentioned if candidate.slug not in manual_order]
    for candidate in automatic[:max_verifications]:
        by_slug[candidate.slug] = verify_candidate(
            candidate,
            self_metadata,
            self_submission_columns,
            api,
            metadata_fetcher=metadata_fetcher,
        )
        metadata_lookups += 1

    candidates = sorted(
        by_slug.values(),
        key=lambda item: (
            not item.confirmed,
            _VERIFICATION_ORDER[item.verification],
            -item.mention_topic_count,
            item.slug,
        ),
    )
    verified = sum(item.verification == "verified" for item in candidates)
    rejected = sum(item.verification == "rejected" for item in candidates)
    not_found = sum(item.verification == "not_found" for item in candidates)
    status = (
        "found_by_mention"
        if verified
        else "all_rejected"
        if candidates
        else "no_candidates"
    )
    diagnostics = SimilarSearchDiagnostics(
        status=status,
        candidates_seen=len(candidates),
        verified=verified,
        rejected=rejected,
        not_found=not_found,
        metadata_lookups=metadata_lookups,
    )
    return candidates, diagnostics


def _discussion_texts(discussion: DiscussionFacts) -> Iterable[str]:
    message_texts = [
        message.content_text for message in discussion.messages if message.content_text
    ]
    joined_messages = "\n\n".join(message_texts)
    if discussion.text and discussion.text != joined_messages:
        yield discussion.text
    yield from message_texts


def _metadata_not_found(metadata: CompetitionMetadata) -> bool:
    return (
        metadata.metric_status == "unavailable"
        and _METADATA_FIELDS <= set(metadata.unavailable_fields)
    )


def _same_metric(
    self_raw: str | None,
    candidate_raw: str | None,
    self_canonical: str | None,
    candidate_canonical: str | None,
) -> bool | None:
    if self_raw is None or candidate_raw is None:
        return None
    if self_canonical is not None and candidate_canonical is not None:
        return self_canonical == candidate_canonical
    return _normalized_text(self_raw) == _normalized_text(candidate_raw)


def _same_optional(left: Any | None, right: Any | None) -> bool | None:
    return None if left is None or right is None else left == right


def _same_optional_text(left: str | None, right: str | None) -> bool | None:
    if left is None or right is None:
        return None
    return _normalized_text(left) == _normalized_text(right)


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _metric_rejection_reason(
    self_raw: str | None,
    candidate_raw: str | None,
    self_canonical: str | None,
    candidate_canonical: str | None,
) -> str:
    if candidate_raw is None:
        return "metric unknown on candidate"
    if self_raw is None:
        return "metric unknown on current competition"
    self_name = self_canonical or _normalized_text(self_raw)
    candidate_name = candidate_canonical or _normalized_text(candidate_raw)
    return f"metric mismatch: {self_name} vs {candidate_name}"
