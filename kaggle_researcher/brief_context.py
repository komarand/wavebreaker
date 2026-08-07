from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from typing import Any, Literal

from pydantic import BaseModel, Field

from kaggle_researcher.facts.cv_lb import summarize_cv_lb
from kaggle_researcher.facts.models import (
    CompetitionFacts,
    DatasetReference,
    DiscussionFacts,
    NotebookFacts,
    ScoreObservation,
)

FACTS_SOURCE_ID = "facts"
CV_LB_SOURCE_ID = "cv_lb"
NOTEBOOK_AST_SOURCE_ID = "notebook_ast"
TRUSTED_SOURCE_IDS = frozenset({FACTS_SOURCE_ID, CV_LB_SOURCE_ID, NOTEBOOK_AST_SOURCE_ID})
TRUSTED_CATEGORIES = frozenset({"official", "cv_lb", "notebook_ast"})
SCORE_EXAMPLES_PER_CLUSTER = 3
RAW_TEXT_EXAMPLE_LIMIT = 160
DATASET_REFS_IN_CONTEXT = 15
_CONTEXT_SEPARATOR = "\n\n"
_TRUSTED_CATEGORY_ORDER = ("official", "cv_lb", "notebook_ast")


class ContextBudgetError(ValueError):
    """Raised when trusted evidence alone exceeds the configured context budget."""


class ContextPackingStats(BaseModel):
    token_budget: int
    estimated_tokens_used: int
    trusted_estimated_tokens: int
    trusted_block_tokens: dict[str, int]
    trusted_total_tokens: int
    optional_estimated_tokens: int
    included_source_ids: list[str]
    omitted_source_ids: list[str]
    omitted_by_reason: dict[str, int]
    omitted_source_reasons: dict[str, str] = Field(default_factory=dict)
    truncation_applied: bool
    included_current_discussions: int
    omitted_current_discussions: int
    included_writeups: int
    omitted_writeups: int
    similar_writeup_counts: dict[str, int]
    leaderboard_statuses: dict[str, str]
    limitations: list[str]


class PackedBriefContext(BaseModel):
    competition_id: str
    text: str
    stats: ContextPackingStats


@dataclass(frozen=True, slots=True)
class _ContextUnit:
    text: str
    source_ids: tuple[str, ...]
    category: Literal["official", "notebook_ast", "cv_lb", "writeup", "discussion"]
    current_discussion: bool = False


def estimated_tokens(text: str) -> int:
    """Estimate tokens deterministically at one token per three UTF-8 bytes."""
    byte_count = len(text.encode("utf-8"))
    return (byte_count + 2) // 3


def pack_brief_context(
    facts: CompetitionFacts,
    max_context_tokens: int,
) -> PackedBriefContext:
    if max_context_tokens <= 0:
        raise ValueError("max_context_tokens must be a positive integer")

    units = _ordered_context_units(facts)
    trusted_units = [unit for unit in units if unit.category in TRUSTED_CATEGORIES]
    optional_units = [unit for unit in units if unit.category not in TRUSTED_CATEGORIES]
    trusted_bytes = _joined_context_bytes(trusted_units)
    trusted_tokens = _tokens_for_bytes(trusted_bytes)
    trusted_block_tokens = _trusted_block_token_counts(trusted_units)
    trusted_total_tokens = sum(trusted_block_tokens.values())
    if trusted_tokens > max_context_tokens:
        block_sizes = ", ".join(
            f"{category}={trusted_block_tokens[category]}"
            for category in _TRUSTED_CATEGORY_ORDER
            if category in trusted_block_tokens
        )
        raise ContextBudgetError(
            f"Trusted context requires {trusted_tokens} tokens but "
            f"max_context_tokens is {max_context_tokens} ({block_sizes})."
        )

    included_units: list[_ContextUnit] = list(trusted_units)
    omitted_units: list[_ContextUnit] = []
    context_parts: list[str] = []
    context_parts.extend(unit.text for unit in trusted_units)
    used_bytes = trusted_bytes

    for index, unit in enumerate(optional_units):
        added_bytes = _context_unit_added_bytes(unit, has_previous=bool(context_parts))
        if _tokens_for_bytes(used_bytes + added_bytes) <= max_context_tokens:
            context_parts.append(unit.text)
            included_units.append(unit)
            used_bytes += added_bytes
            continue
        omitted_units.extend(optional_units[index:])
        break

    context_note = _context_omission_note(omitted_units)
    while context_note and _tokens_for_bytes(
        used_bytes
        + len(_CONTEXT_SEPARATOR.encode("utf-8"))
        + len(context_note.encode("utf-8"))
    ) > max_context_tokens:
        if len(included_units) == len(trusted_units):
            required_tokens = _tokens_for_bytes(
                trusted_bytes
                + len(_CONTEXT_SEPARATOR.encode("utf-8"))
                + len(context_note.encode("utf-8"))
            )
            raise ContextBudgetError(
                "Trusted context plus its required omission note requires "
                f"{required_tokens} tokens but max_context_tokens is "
                f"{max_context_tokens}."
            )
        removed = included_units.pop()
        context_parts.pop()
        used_bytes -= _context_unit_added_bytes(
            removed,
            has_previous=bool(context_parts),
        )
        omitted_units.insert(0, removed)
        context_note = _context_omission_note(omitted_units)

    if context_note:
        context_parts.append(context_note)

    text = _join_context(context_parts)
    included_source_ids = _ordered_unique(
        source_id for unit in included_units for source_id in unit.source_ids
    )
    omitted_source_ids = _ordered_unique(
        source_id for unit in omitted_units for source_id in unit.source_ids
    )
    omitted_source_reasons = {source_id: "context_budget" for source_id in omitted_source_ids}
    omitted_count = sum(max(1, len(unit.source_ids)) for unit in omitted_units)
    omitted_by_reason = {"context_budget": omitted_count} if omitted_count else {}

    included_current_discussions = sum(unit.current_discussion for unit in included_units)
    omitted_current_discussions = sum(unit.current_discussion for unit in omitted_units)
    included_writeups = sum(unit.category == "writeup" for unit in included_units)
    omitted_writeups = sum(unit.category == "writeup" for unit in omitted_units)

    limitations = _packing_limitations(
        facts=facts,
        omitted_units=omitted_units,
        omitted_current_discussions=omitted_current_discussions,
        omitted_writeups=omitted_writeups,
    )
    stats = ContextPackingStats(
        token_budget=max_context_tokens,
        estimated_tokens_used=estimated_tokens(text),
        trusted_estimated_tokens=trusted_tokens,
        trusted_block_tokens=trusted_block_tokens,
        trusted_total_tokens=trusted_total_tokens,
        optional_estimated_tokens=_tokens_for_bytes(max(0, used_bytes - trusted_bytes)),
        included_source_ids=included_source_ids,
        omitted_source_ids=omitted_source_ids,
        omitted_by_reason=omitted_by_reason,
        omitted_source_reasons=omitted_source_reasons,
        truncation_applied=bool(omitted_units),
        included_current_discussions=included_current_discussions,
        omitted_current_discussions=omitted_current_discussions,
        included_writeups=included_writeups,
        omitted_writeups=omitted_writeups,
        similar_writeup_counts=_similar_writeup_counts(facts),
        leaderboard_statuses={
            item.competition_id: item.status
            for item in sorted(
                facts.similar_competitions,
                key=lambda item: item.competition_id,
            )
        },
        limitations=limitations,
    )
    return PackedBriefContext(
        competition_id=facts.competition_id,
        text=text,
        stats=stats,
    )


def _ordered_context_units(facts: CompetitionFacts) -> list[_ContextUnit]:
    units = [_official_facts_unit(facts)]
    if facts.cv_lb_pairs or facts.similar_competitions:
        units.append(_cv_lb_unit(facts))
    if facts.dataset_references:
        units.append(_dataset_references_unit(facts.dataset_references))
    units.extend(_notebook_ast_units(facts.notebooks))

    discussions = sorted(facts.discussions, key=_discussion_sort_key)
    writeups = [item for item in discussions if item.source_type == "winner_writeup"]
    current_host = [
        item
        for item in discussions
        if item.source_type == "discussion"
        and item.competition_id == facts.competition_id
        and item.author_is_host
    ]
    current_other = [
        item
        for item in discussions
        if item.source_type == "discussion"
        and item.competition_id == facts.competition_id
        and not item.author_is_host
    ]
    other = [
        item
        for item in discussions
        if item.source_type == "discussion" and item.competition_id != facts.competition_id
    ]
    units.extend(_untrusted_unit(item, facts.competition_id) for item in writeups)
    units.extend(_untrusted_unit(item, facts.competition_id) for item in current_host)
    units.extend(_untrusted_unit(item, facts.competition_id) for item in current_other)
    units.extend(_untrusted_unit(item, facts.competition_id) for item in other)
    return units


def _official_facts_unit(facts: CompetitionFacts) -> _ContextUnit:
    payload = {
        "collection_errors": facts.collection_errors,
        "collected_at": facts.collected_at.isoformat(),
        "competition_id": facts.competition_id,
        "files": facts.files.model_dump(mode="json"),
        "metadata": facts.metadata.model_dump(mode="json"),
        "public_leaderboard_shape": (
            facts.public_leaderboard.shape.model_dump(mode="json")
            if facts.public_leaderboard.shape is not None
            else None
        ),
        "schema_version": facts.schema_version,
        "score_diagnostics": facts.score_diagnostics.model_dump(mode="json"),
        "user_constraints": facts.user_constraints.model_dump(mode="json"),
    }
    return _ContextUnit(
        text=_trusted_block(
            "TRUSTED_OFFICIAL_FACTS",
            payload,
            source_id=FACTS_SOURCE_ID,
        ),
        source_ids=(FACTS_SOURCE_ID,),
        category="official",
    )


def _dataset_references_unit(
    references: list[DatasetReference],
) -> _ContextUnit:
    ordered = sorted(
        references,
        key=lambda item: (-item.cluster_count, item.slug.casefold()),
    )
    included = ordered[:DATASET_REFS_IN_CONTEXT]
    payload = {
        "dataset_references": [item.model_dump(mode="json") for item in included],
    }
    omitted_count = len(ordered) - len(included)
    if omitted_count:
        payload["dataset_references_omitted"] = omitted_count
    notebook_refs = _ordered_unique(
        notebook_ref for item in included for notebook_ref in item.notebook_refs
    )
    return _ContextUnit(
        text=_trusted_block(
            "TRUSTED_NOTEBOOK_AST",
            payload,
            source_id=NOTEBOOK_AST_SOURCE_ID,
        ),
        source_ids=(NOTEBOOK_AST_SOURCE_ID, *notebook_refs),
        category="notebook_ast",
    )


def _notebook_ast_units(notebooks: list[NotebookFacts]) -> list[_ContextUnit]:
    clusters: dict[str, list[NotebookFacts]] = {}
    for notebook in notebooks:
        clusters.setdefault(notebook.lineage_cluster_id, []).append(notebook)

    units: list[_ContextUnit] = []
    for cluster_id in sorted(clusters):
        members = sorted(clusters[cluster_id], key=lambda item: item.ref)
        notebook_source_ids = tuple(item.ref for item in members)
        payload = {
            "cluster_id": cluster_id,
            "declared_cv": sorted({value for item in members for value in item.declared_cv}),
            "feature_ops": _aggregate_observations(members, "feature_ops"),
            "metrics": _aggregate_observations(members, "metrics"),
            "models": _aggregate_observations(members, "models"),
            "notebook_count": len(members),
            "parse_status_counts": dict(
                sorted(Counter(item.parse_status for item in members).items())
            ),
            "score_observations": _score_observation_summary(members),
            "source_ids": list(notebook_source_ids),
            "splitters": _aggregate_observations(members, "splitters"),
        }
        units.append(
            _ContextUnit(
                text=_trusted_block(
                    "TRUSTED_NOTEBOOK_AST",
                    payload,
                    lineage_cluster_id=cluster_id,
                    source_id=NOTEBOOK_AST_SOURCE_ID,
                ),
                source_ids=(NOTEBOOK_AST_SOURCE_ID, *notebook_source_ids),
                category="notebook_ast",
            )
        )
    return units


def _score_observation_summary(notebooks: list[NotebookFacts]) -> dict[str, Any]:
    observations = [
        (notebook.ref, observation)
        for notebook in notebooks
        for observation in notebook.score_observations
        if observation.plausible
    ]
    by_split: dict[str, list[tuple[str, ScoreObservation]]] = {
        split: [item for item in observations if item[1].split == split]
        for split in ("cv", "lb", "unknown")
    }
    summary: dict[str, Any] = {}
    for split in ("cv", "lb"):
        values = [observation.value for _, observation in by_split[split]]
        if values:
            summary[split] = {
                "count": len(values),
                "median": round(statistics.median(values), 6),
                "min": round(min(values), 6),
                "max": round(max(values), 6),
            }
    summary["unknown"] = {"count": len(by_split["unknown"])}

    ordered_examples = sorted(
        observations,
        key=lambda item: (
            {"cv": 0, "lb": 1, "unknown": 2}[item[1].split],
            item[0],
            item[1].locator,
            item[1].value,
        ),
    )[:SCORE_EXAMPLES_PER_CLUSTER]
    summary["examples"] = [
        _score_observation_example(notebook_ref, observation)
        for notebook_ref, observation in ordered_examples
    ]
    return summary


def _score_observation_example(
    notebook_ref: str,
    observation: ScoreObservation,
) -> dict[str, Any]:
    return {
        "notebook_ref": notebook_ref,
        "split": observation.split,
        "value": observation.value,
        "metric_raw": observation.metric_raw,
        "metric_canonical": observation.metric_canonical,
        "locator": observation.locator,
        "raw_text": _truncate_example_text(observation.raw_text),
    }


def _truncate_example_text(value: str) -> str:
    if len(value) <= RAW_TEXT_EXAMPLE_LIMIT:
        return value
    return f"{value[:RAW_TEXT_EXAMPLE_LIMIT]}…"


def _aggregate_observations(
    notebooks: list[NotebookFacts],
    field_name: Literal["splitters", "models", "metrics", "feature_ops"],
) -> list[dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {}
    for notebook in notebooks:
        for observation in getattr(notebook, field_name):
            compact = {"kwargs": observation.kwargs, "name": observation.name}
            observations[_canonical_json(compact)] = compact
    return [observations[key] for key in sorted(observations)]


def _cv_lb_unit(facts: CompetitionFacts) -> _ContextUnit:
    payload = {
        "summary": summarize_cv_lb(facts.cv_lb_pairs),
        "cv_lb_pairs": [
            item.model_dump(mode="json")
            for item in sorted(
                facts.cv_lb_pairs,
                key=lambda item: (
                    item.lineage_cluster_id,
                    item.notebook_ref,
                    item.declared_cv,
                    item.public_score,
                ),
            )
        ],
        "leaderboard_stability": [
            item.model_dump(mode="json")
            for item in sorted(
                facts.similar_competitions,
                key=lambda item: item.competition_id,
            )
        ],
    }
    return _ContextUnit(
        text=_trusted_block(
            "TRUSTED_CV_LB",
            payload,
            source_id=CV_LB_SOURCE_ID,
        ),
        source_ids=(CV_LB_SOURCE_ID,),
        category="cv_lb",
    )


def _untrusted_unit(
    discussion: DiscussionFacts,
    current_competition_id: str,
) -> _ContextUnit:
    header = (
        "<UNTRUSTED_SOURCE\n"
        f'  source_id="{escape(discussion.topic_id, quote=True)}"\n'
        f'  source_type="{discussion.source_type}"\n'
        f'  author="{escape(discussion.author or "", quote=True)}"\n'
        f'  competition_id="{escape(discussion.competition_id, quote=True)}"\n'
        ">"
    )
    return _ContextUnit(
        text=f"{header}\n{discussion.text}\n</UNTRUSTED_SOURCE>",
        source_ids=(discussion.topic_id,),
        category=("writeup" if discussion.source_type == "winner_writeup" else "discussion"),
        current_discussion=(
            discussion.source_type == "discussion"
            and discussion.competition_id == current_competition_id
        ),
    )


def _discussion_sort_key(item: DiscussionFacts) -> tuple[Any, ...]:
    return (
        -item.votes,
        -_datetime_timestamp(item.created_at),
        item.topic_id,
        item.competition_id,
    )


def _datetime_timestamp(value: datetime | None) -> float:
    if value is None:
        return float("-inf")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).timestamp()


def _trusted_block(name: str, payload: object, **attributes: str) -> str:
    attribute_text = "".join(
        f' {key}="{escape(value, quote=True)}"' for key, value in sorted(attributes.items())
    )
    return f"<{name}{attribute_text}>\n{_canonical_json(payload)}\n</{name}>"


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _join_context(parts: list[str]) -> str:
    return _CONTEXT_SEPARATOR.join(parts)


def _joined_context_bytes(units: list[_ContextUnit]) -> int:
    if not units:
        return 0
    return sum(len(unit.text.encode("utf-8")) for unit in units) + len(
        _CONTEXT_SEPARATOR.encode("utf-8")
    ) * (len(units) - 1)


def _trusted_block_token_counts(units: list[_ContextUnit]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for unit in units:
        counts[unit.category] += estimated_tokens(unit.text)
    return {category: counts[category] for category in _TRUSTED_CATEGORY_ORDER if counts[category]}


def _context_unit_added_bytes(unit: _ContextUnit, *, has_previous: bool) -> int:
    separator_bytes = len(_CONTEXT_SEPARATOR.encode("utf-8")) if has_previous else 0
    return separator_bytes + len(unit.text.encode("utf-8"))


def _tokens_for_bytes(byte_count: int) -> int:
    return (byte_count + 2) // 3


def _ordered_unique(values: Any) -> list[str]:
    return list(dict.fromkeys(values))


def _similar_writeup_counts(facts: CompetitionFacts) -> dict[str, int]:
    counts = Counter(
        item.competition_id for item in facts.discussions if item.source_type == "winner_writeup"
    )
    return dict(sorted(counts.items()))


def _packing_limitations(
    *,
    facts: CompetitionFacts,
    omitted_units: list[_ContextUnit],
    omitted_current_discussions: int,
    omitted_writeups: int,
) -> list[str]:
    limitations: list[str] = []
    if facts.files.sample_submission_source == "unavailable":
        limitations.append("Sample submission metadata is unavailable.")

    parse_counts = Counter(item.parse_status for item in facts.notebooks)
    incomplete_notebooks = parse_counts["partial"] + parse_counts["failed"]
    if incomplete_notebooks:
        limitations.append(
            f"Notebook parsing was partial or failed for {incomplete_notebooks} notebooks."
        )

    for item in sorted(
        facts.similar_competitions,
        key=lambda value: value.competition_id,
    ):
        if item.status == "not_computable":
            reason = item.not_computable_reason or "The required records are unavailable."
            limitations.append(
                f"Leaderboard stability for {item.competition_id} is not computable: " f"{reason}"
            )

    if not facts.notebooks:
        limitations.append("No notebook AST observations were available.")
    if not facts.cv_lb_pairs:
        limitations.append("No CV/LB observations were available.")
    if not facts.similar_competitions:
        limitations.append("No similar-competition leaderboard stability records were available.")

    writeup_count = sum(item.source_type == "winner_writeup" for item in facts.discussions)
    current_discussion_count = sum(
        item.source_type == "discussion" and item.competition_id == facts.competition_id
        for item in facts.discussions
    )
    if not writeup_count:
        limitations.append("No winner writeups were collected.")
    if not current_discussion_count:
        limitations.append("No current-competition discussions were collected.")
    if omitted_writeups:
        limitations.append(
            f"{omitted_writeups} winner writeups were omitted whole because of the "
            "context budget."
        )
    if omitted_current_discussions:
        limitations.append(
            f"{omitted_current_discussions} current-competition discussions were "
            "omitted whole because of the context budget."
        )
    omitted_other_discussions = sum(
        unit.category == "discussion" and not unit.current_discussion for unit in omitted_units
    )
    if omitted_other_discussions:
        limitations.append(
            f"{omitted_other_discussions} other discussions were omitted whole because "
            "of the context budget."
        )

    return limitations


def _context_omission_note(omitted_units: list[_ContextUnit]) -> str | None:
    if not omitted_units:
        return None

    writeups = sum(unit.category == "writeup" for unit in omitted_units)
    discussions = sum(unit.category == "discussion" for unit in omitted_units)
    descriptions: list[str] = []
    if writeups:
        descriptions.append(
            f"{writeups} writeup document{'s' if writeups != 1 else ''}"
        )
    if discussions:
        descriptions.append(
            f"{discussions} discussion document{'s' if discussions != 1 else ''}"
        )
    return f"CONTEXT_NOTE: omitted {' and '.join(descriptions)} due to budget"
