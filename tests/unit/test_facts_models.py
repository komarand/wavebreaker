from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from kaggle_researcher.facts.models import (
    CodeObservation,
    CompetitionFacts,
    CompetitionMetadata,
    CvLbPair,
    DiscussionFacts,
    FileInfo,
    FileManifest,
    LeaderboardStability,
    NotebookFacts,
    UserConstraints,
)


def test_optional_fact_fields_default_to_none() -> None:
    metadata = CompetitionMetadata(
        competition_id="example",
        unavailable_fields=["title", "metric_name"],
    )
    file_info = FileInfo(name="unknown.bin", role_hint="auxiliary")
    constraints = UserConstraints()

    assert metadata.title is None
    assert metadata.metric_name is None
    assert metadata.deadline is None
    assert file_info.size_bytes is None
    assert constraints.vram_gb is None
    assert constraints.hours_per_week is None
    assert constraints.cloud_budget_usd is None
    assert constraints.objective == "medal"


def test_notebook_votes_default_to_zero() -> None:
    notebook = NotebookFacts(
        ref="author/notebook",
        title="Notebook",
        ast_fingerprint="fingerprint",
        lineage_cluster_id="lineage",
        splitters=[],
        models=[],
        metrics=[],
        feature_ops=[],
        declared_cv=[],
        parse_status="ok",
    )

    assert notebook.votes == 0


def test_competition_facts_preserve_collected_values_and_errors() -> None:
    collected_at = datetime(2026, 7, 29, 10, 30, tzinfo=UTC)
    observation = CodeObservation(
        name="GroupKFold",
        kwargs={"n_splits": "5", "groups": "customer_id"},
        locator="cell_18",
    )
    notebook = NotebookFacts(
        ref="author/example-notebook",
        title="Example notebook",
        votes=42,
        public_score=0.8123,
        ast_fingerprint="abc123",
        lineage_cluster_id="lineage-1",
        splitters=[observation],
        models=[],
        metrics=[],
        feature_ops=[],
        declared_cv=["0.8012"],
        parse_status="ok",
    )
    facts = CompetitionFacts(
        competition_id="example",
        collected_at=collected_at,
        metadata=CompetitionMetadata(
            competition_id="example",
            title="Example Competition",
            unavailable_fields=["reward"],
        ),
        files=FileManifest(
            files=[
                FileInfo(
                    name="train.csv",
                    size_bytes=100,
                    role_hint="train",
                )
            ],
            sample_submission_columns=[],
            sample_submission_source="unavailable",
            limitations=["sample submission is unavailable"],
        ),
        notebooks=[notebook],
        discussions=[
            DiscussionFacts(
                topic_id="123",
                title="Host update",
                author_is_host=True,
                votes=5,
                source_type="discussion",
                competition_id="example",
                text="Competition update.",
            )
        ],
        similar_competitions=[
            LeaderboardStability(
                competition_id="similar-example",
                status="not_computable",
                matched_teams=0,
                source="unavailable",
                not_computable_reason="Meta Kaggle is not configured",
            )
        ],
        cv_lb_pairs=[
            CvLbPair(
                notebook_ref=notebook.ref,
                declared_cv=0.8012,
                public_score=0.8123,
                lineage_cluster_id=notebook.lineage_cluster_id,
            )
        ],
        user_constraints=UserConstraints(vram_gb=12),
        collection_errors=["discussion collection failed: 403"],
    )

    payload = facts.model_dump(mode="json")

    assert payload["schema_version"] == "1.0"
    assert payload["collected_at"] == "2026-07-29T10:30:00Z"
    assert payload["notebooks"][0]["splitters"][0]["kwargs"]["groups"] == "customer_id"
    assert payload["similar_competitions"][0]["match_fraction"] is None
    assert payload["similar_competitions"][0]["limitations"] == []
    assert payload["collection_errors"] == ["discussion collection failed: 403"]


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (FileInfo, {"name": "train.csv", "role_hint": "unknown"}),
        (
            FileManifest,
            {
                "files": [],
                "sample_submission_columns": [],
                "sample_submission_source": "header_download",
                "limitations": [],
            },
        ),
        (
            UserConstraints,
            {"objective": "win"},
        ),
    ],
)
def test_literal_fields_reject_unknown_values(
    model: type[object],
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        model(**kwargs)


def test_facts_models_have_no_forbidden_pipeline_imports() -> None:
    model_source = (
        Path(__file__).resolve().parents[2] / "kaggle_researcher" / "facts" / "models.py"
    ).read_text(encoding="utf-8")

    for forbidden_module in ("deepseek_client", "retriever", "embedder", "store"):
        assert forbidden_module not in model_source
