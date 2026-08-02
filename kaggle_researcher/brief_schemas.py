from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from kaggle_researcher.research_scout_schemas import EdaTask, ResearchHypothesis


class Claim(BaseModel):
    text: str
    source_ids: list[str]
    kind: Literal["fact", "claim", "inference"]


class CompetitionBrief(BaseModel):
    schema_version: str = "1.0"
    competition_id: str

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
