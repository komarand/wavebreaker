from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from kaggle_researcher.research_scout_schemas import EdaTask, ResearchHypothesis


class Claim(BaseModel):
    claim_id: str = Field(pattern=r"^claim_[a-zA-Z0-9_-]+$")
    text: str
    source_ids: list[str]
    kind: Literal["fact", "claim", "inference"]
    evidence_strength: Literal[
        "official",
        "measured_with_protocol",
        "reported_score",
        "prevalence",
        "inference",
    ]


class ClaimStats(BaseModel):
    fact: int
    claim: int
    inference: int
    total: int
    grounded: int
    ungrounded: int
    grounding_rate: float
    distinct_sources: int
    by_evidence_strength: dict[str, int]
    hypotheses_total: int
    hypotheses_dropped_unverifiable: int


class CompetitionBrief(BaseModel):
    schema_version: str = "1.0"
    competition_id: str
    prompt_version: str | None = None
    claim_stats: ClaimStats | None = None

    thesis: str
    thesis_support: list[str]

    validation: list[Claim]
    metric_notes: list[Claim]
    leakage_risks: list[Claim]
    what_works: list[Claim]
    time_wasters: list[Claim]

    hypotheses: list[ResearchHypothesis]
    eda_tasks: list[EdaTask]

    first_moves: list[str]
    unknowns: list[str]
    limitations: list[str]

    @model_validator(mode="after")
    def validate_thesis_grounding(self) -> CompetitionBrief:
        claims = [
            *self.validation,
            *self.metric_notes,
            *self.leakage_risks,
            *self.what_works,
            *self.time_wasters,
        ]
        claims_by_id: dict[str, Claim] = {}
        duplicate_ids: set[str] = set()
        for claim in claims:
            if claim.claim_id in claims_by_id:
                duplicate_ids.add(claim.claim_id)
            claims_by_id[claim.claim_id] = claim

        if duplicate_ids:
            duplicates = ", ".join(sorted(duplicate_ids))
            raise ValueError(f"claim_id values must be globally unique: {duplicates}")

        if self.thesis.strip() and not self.thesis_support:
            raise ValueError("a non-empty thesis requires thesis_support claim identifiers")

        unknown_ids = sorted(set(self.thesis_support) - claims_by_id.keys())
        if unknown_ids:
            unknown = ", ".join(unknown_ids)
            raise ValueError(f"thesis_support contains unknown claim_id values: {unknown}")

        unsupported_ids = sorted(
            claim_id
            for claim_id in set(self.thesis_support)
            if not claims_by_id[claim_id].source_ids
        )
        if unsupported_ids:
            unsupported = ", ".join(unsupported_ids)
            raise ValueError(
                "thesis-supporting claims require at least one source_id: "
                f"{unsupported}"
            )

        return self
