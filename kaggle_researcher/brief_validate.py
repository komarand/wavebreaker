from __future__ import annotations

from kaggle_researcher.brief_schemas import Claim, CompetitionBrief
from kaggle_researcher.facts.models import CompetitionFacts


CLAIM_SECTIONS = (
    "validation",
    "metric_notes",
    "leakage_risks",
    "what_works",
    "time_wasters",
)


def validate_brief(
    brief: CompetitionBrief,
    facts: CompetitionFacts,
) -> CompetitionBrief:
    """Return a source-validated copy using drop, move, and record operations only."""
    valid_source_ids = _valid_source_ids(facts)
    limitations = list(brief.limitations)
    unknowns = list(brief.unknowns)
    validated_sections: dict[str, list[Claim]] = {}
    removed_claim_ids: set[str] = set()

    for section_name in CLAIM_SECTIONS:
        validated_claims: list[Claim] = []
        for claim in getattr(brief, section_name):
            retained_source_ids = [
                source_id
                for source_id in claim.source_ids
                if source_id in valid_source_ids
            ]
            invalid_source_ids = list(
                dict.fromkeys(
                    source_id
                    for source_id in claim.source_ids
                    if source_id not in valid_source_ids
                )
            )
            limitations.extend(
                _invalid_source_limitation(claim, source_id)
                for source_id in invalid_source_ids
            )

            if not retained_source_ids:
                unknowns.append(f"unsupported: {claim.text}")
                removed_claim_ids.add(claim.claim_id)
                continue

            validated_claims.append(
                Claim.model_validate(
                    {
                        **claim.model_dump(mode="python"),
                        "source_ids": retained_source_ids,
                    }
                )
            )
        validated_sections[section_name] = validated_claims

    thesis_support = [
        claim_id
        for claim_id in brief.thesis_support
        if claim_id not in removed_claim_ids
    ]
    thesis = brief.thesis
    if thesis.strip() and not thesis_support:
        unknowns.append(f"unsupported: {thesis}")
        limitations.append(
            "Thesis was moved to unknowns after all supporting claims were removed."
        )
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
    return CompetitionBrief.model_validate(payload)


def _valid_source_ids(facts: CompetitionFacts) -> set[str]:
    return {
        "facts",
        *(notebook.ref for notebook in facts.notebooks),
        *(discussion.topic_id for discussion in facts.discussions),
    }


def _invalid_source_limitation(claim: Claim, source_id: str) -> str:
    return (
        f"Claim {claim.claim_id} references invalid source_id {source_id!r}; "
        "the source ID was removed."
    )
