from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation

from kaggle_researcher.brief_context import CV_LB_SOURCE_ID, TRUSTED_SOURCE_IDS
from kaggle_researcher.brief_schemas import Claim, ClaimStats, CompetitionBrief
from kaggle_researcher.facts.cv_lb import summarize_cv_lb
from kaggle_researcher.facts.models import CompetitionFacts

CLAIM_SECTIONS = (
    "validation",
    "metric_notes",
    "leakage_risks",
    "what_works",
    "time_wasters",
)
_NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)" r"(?:[eE][-+]?\d+)?(?!\w)"
)


def validate_brief(
    brief: CompetitionBrief,
    facts: CompetitionFacts,
) -> CompetitionBrief:
    """Return a source-validated copy using drop, move, and record operations only."""
    valid_source_ids = _valid_source_ids(facts)
    discussion_source_ids = {discussion.topic_id for discussion in facts.discussions}
    limitations = list(brief.limitations)
    unknowns = list(brief.unknowns)
    cv_lb_summary = summarize_cv_lb(facts.cv_lb_pairs)
    if cv_lb_summary["count"] and cv_lb_summary["reliability"] != "sufficient":
        limitations.append(
            f"CV/LB reliability is {cv_lb_summary['reliability']} "
            f"(n={cv_lb_summary['count']}): {cv_lb_summary['note']}."
        )
    validated_sections: dict[str, list[Claim]] = {}
    removed_claim_ids: set[str] = set()

    for section_name in CLAIM_SECTIONS:
        validated_claims: list[Claim] = []
        for claim in getattr(brief, section_name):
            retained_source_ids = [
                source_id for source_id in claim.source_ids if source_id in valid_source_ids
            ]
            invalid_source_ids = list(
                dict.fromkeys(
                    source_id for source_id in claim.source_ids if source_id not in valid_source_ids
                )
            )
            limitations.extend(
                _invalid_source_limitation(claim, source_id) for source_id in invalid_source_ids
            )

            if not retained_source_ids:
                unknowns.append(f"unsupported: {claim.text}")
                removed_claim_ids.add(claim.claim_id)
                continue

            validated_claim = claim.model_copy(update={"source_ids": retained_source_ids})
            if validated_claim.kind == "fact" and set(retained_source_ids) <= discussion_source_ids:
                validated_claim = validated_claim.model_copy(update={"kind": "claim"})
                limitations.append(
                    f"Claim {claim.claim_id} was downgraded from fact to claim because "
                    "it is supported only by discussion or writeup sources."
                )
            validated_claims.append(validated_claim)
        validated_sections[section_name] = validated_claims

    thesis_support = [
        claim_id for claim_id in brief.thesis_support if claim_id not in removed_claim_ids
    ]
    thesis = brief.thesis
    validated_claims_by_id = {
        claim.claim_id: claim
        for section_name in CLAIM_SECTIONS
        for claim in validated_sections[section_name]
    }
    if thesis.strip():
        thesis_support = _filter_unreliable_numeric_thesis_support(
            thesis=thesis,
            thesis_support=thesis_support,
            claims_by_id=validated_claims_by_id,
            cv_lb_summary=cv_lb_summary,
            limitations=limitations,
        )
        thesis_rejected = False
        if not thesis_support:
            limitations.append(
                "Thesis was moved to unknowns after all supporting claims were removed."
            )
            thesis_rejected = True

        unsupported_numbers = _unsupported_thesis_numbers(
            thesis=thesis,
            thesis_support=thesis_support,
            claims_by_id=validated_claims_by_id,
            facts=facts,
        )
        if unsupported_numbers:
            limitations.append(
                "Thesis was moved to unknowns because numeric literals were absent "
                "from its remaining supporting claims and CompetitionFacts: "
                f"{', '.join(unsupported_numbers)}."
            )
            thesis_rejected = True

        if thesis_rejected:
            unknowns.append(f"unsupported: {thesis}")
            thesis_support = []
            thesis = ""

    if not thesis.strip():
        thesis = ""

    payload = brief.model_dump(mode="python")
    payload.update(validated_sections)
    payload.update(
        {
            "thesis": thesis,
            "thesis_support": thesis_support,
            "unknowns": unknowns,
            "limitations": limitations,
        }
    )
    validated = CompetitionBrief.model_validate(payload)
    return validated.model_copy(update={"claim_stats": _claim_stats(validated)})


def _claim_stats(brief: CompetitionBrief) -> ClaimStats:
    claims = [
        claim
        for section_name in CLAIM_SECTIONS
        for claim in getattr(brief, section_name)
    ]
    kind_counts = {
        kind: sum(claim.kind == kind for claim in claims)
        for kind in ("fact", "claim", "inference")
    }
    grounded = sum(bool(claim.source_ids) for claim in claims)
    total = len(claims)
    return ClaimStats(
        fact=kind_counts["fact"],
        claim=kind_counts["claim"],
        inference=kind_counts["inference"],
        total=total,
        grounded=grounded,
        ungrounded=total - grounded,
        grounding_rate=round(grounded / total, 4) if total else 0.0,
        distinct_sources=len(
            {
                source_id
                for claim in claims
                for source_id in claim.source_ids
            }
        ),
    )


def _valid_source_ids(facts: CompetitionFacts) -> set[str]:
    return {
        *TRUSTED_SOURCE_IDS,
        *(notebook.ref for notebook in facts.notebooks),
        *(discussion.topic_id for discussion in facts.discussions),
    }


def _invalid_source_limitation(claim: Claim, source_id: str) -> str:
    return (
        f"Claim {claim.claim_id} references invalid source_id {source_id!r}; "
        "the source ID was removed."
    )


def _filter_unreliable_numeric_thesis_support(
    *,
    thesis: str,
    thesis_support: list[str],
    claims_by_id: dict[str, Claim],
    cv_lb_summary: dict[str, int | float | str | None],
    limitations: list[str],
) -> list[str]:
    if not _numeric_literals(thesis) or cv_lb_summary["reliability"] != "insufficient":
        return thesis_support

    retained: list[str] = []
    for claim_id in thesis_support:
        claim = claims_by_id.get(claim_id)
        is_unreliable_numeric_cv_lb_claim = (
            claim is not None
            and CV_LB_SOURCE_ID in claim.source_ids
            and bool(_numeric_literals(claim.text))
        )
        if not is_unreliable_numeric_cv_lb_claim:
            retained.append(claim_id)
            continue
        limitations.append(
            f"Thesis support claim {claim_id} was removed because numeric claims from "
            f"source {CV_LB_SOURCE_ID!r} have reliability=insufficient "
            f"(n={cv_lb_summary['count']})."
        )
    return retained


def _unsupported_thesis_numbers(
    *,
    thesis: str,
    thesis_support: list[str],
    claims_by_id: dict[str, Claim],
    facts: CompetitionFacts,
) -> list[str]:
    thesis_numbers = _numeric_literals(thesis)
    if not thesis_numbers:
        return []

    supporting_text = "\n".join(
        claims_by_id[claim_id].text for claim_id in thesis_support if claim_id in claims_by_id
    )
    facts_text = json.dumps(
        facts.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    available_numbers = {
        *_numeric_literals(supporting_text),
        *_numeric_literals(facts_text),
    }
    return [
        rendered
        for canonical, rendered in thesis_numbers.items()
        if canonical not in available_numbers
    ]


def _numeric_literals(text: str) -> dict[str, str]:
    literals: dict[str, str] = {}
    for match in _NUMBER_PATTERN.finditer(text):
        rendered = match.group(0)
        try:
            value = Decimal(rendered.replace(",", ""))
        except InvalidOperation:
            continue
        canonical = format(value.normalize(), "f")
        if value == 0:
            canonical = "0"
        literals.setdefault(canonical, rendered)
    return literals
