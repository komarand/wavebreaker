from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_researcher import wave
from kaggle_researcher.facts import collect
from kaggle_researcher.facts.models import (
    CompetitionMetadata,
    DiscussionFacts,
    FileManifest,
)

NOTEBOOK_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "facts" / "notebook_groupkfold.ipynb"
)


def test_wave_facts_writes_offline_checkpoint_and_cluster_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    _patch_offline_stages(monkeypatch)
    writeup_limits: list[int] = []

    def winner_writeups(slugs: list[str], limit: int) -> list[DiscussionFacts]:
        writeup_limits.append(limit)
        return [_discussion(slug, "winner_writeup") for slug in slugs]

    monkeypatch.setattr(collect, "fetch_winner_writeups", winner_writeups)

    wave.main(
        [
            "facts",
            "offline-comp",
            "--max-notebooks",
            "2",
            "--max-discussions",
            "3",
            "--writeups-per-competition",
            "4",
            "--similar",
            "past-comp",
            "--out",
            str(tmp_path),
        ]
    )

    facts_path = tmp_path / "facts.json"
    payload = json.loads(facts_path.read_text(encoding="utf-8"))
    output = capsys.readouterr().out
    assert payload["competition_id"] == "offline-comp"
    assert len(payload["notebooks"]) == 2
    assert len({item["lineage_cluster_id"] for item in payload["notebooks"]}) == 1
    assert payload["similar_competitions"][0]["status"] == "not_computable"
    assert payload["similar_competitions"][0]["not_computable_reason"] == (
        "Meta Kaggle dumps not configured."
    )
    assert writeup_limits == [4]
    assert payload["collection_errors"] == []
    assert payload["discussion_collection_status"] == "collected"
    assert payload["cv_lb_diagnostics"] == {
        "notebooks_total": 2,
        "notebooks_with_public_score": 2,
        "notebooks_with_declared_cv": 2,
        "notebooks_with_both": 2,
        "comparable_pairs": 2,
        "rejected_non_comparable_pairs": 0,
        "zero_pairs_reason": None,
    }
    assert "deepseek" not in facts_path.read_text(encoding="utf-8").lower()
    assert "metric: roc_auc" in output
    assert "notebooks: 2" in output
    assert "lineage clusters: 1" in output
    assert "'StratifiedGroupKFold': 1" in output
    assert "discussions: 2" in output
    assert "topics: 1" in output
    assert "messages: 0" in output
    assert "writeup candidates: 0" in output
    assert "external links: 0" in output
    assert "failed topics: 0" in output
    assert "public notebook scores: 2" in output
    assert "cv observations: 2" in output
    assert "comparable cv/lb pairs: 2" in output
    assert "discussion status: collected" in output


def test_wave_facts_writes_checkpoint_after_stage_three_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    _patch_offline_stages(monkeypatch)
    monkeypatch.setattr(
        collect,
        "list_competition_notebooks",
        lambda slug, limit: _raise(RuntimeError("forced stage 3 failure")),
    )

    wave.main(["facts", "offline-comp", "--out", str(tmp_path)])

    payload = json.loads((tmp_path / "facts.json").read_text(encoding="utf-8"))
    assert payload["notebooks"] == []
    assert "notebook listing failed (RuntimeError)" in payload["collection_errors"]


def _patch_offline_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        collect,
        "fetch_competition_metadata",
        lambda slug: CompetitionMetadata(
            competition_id=slug,
            metric_name="roc_auc",
            is_code_competition=True,
            unavailable_fields=[],
        ),
    )
    monkeypatch.setattr(
        collect,
        "fetch_file_manifest",
        lambda slug, limit: FileManifest(
            files=[],
            train_test_size_ratio=1.5,
            sample_submission_columns=["id", "target"],
            sample_submission_source="api",
            limitations=[],
        ),
    )
    monkeypatch.setattr(
        collect,
        "list_competition_notebooks",
        lambda slug, limit: [
            {
                "ref": f"author/fork-{index}",
                "title": f"Fork {index}",
                "author": "author",
                "votes": 5 - index,
                "public_score": 0.78,
                "last_run": None,
            }
            for index in range(limit)
        ],
    )
    monkeypatch.setattr(
        collect,
        "pull_notebook",
        lambda ref, destination: NOTEBOOK_FIXTURE,
    )
    monkeypatch.setattr(
        collect,
        "fetch_competition_discussions",
        lambda slug, limit: [_discussion(slug, "discussion")],
    )
    monkeypatch.setattr(
        collect,
        "fetch_winner_writeups",
        lambda slugs, limit: [_discussion(slug, "winner_writeup") for slug in slugs],
    )


def _discussion(slug: str, source_type: str) -> DiscussionFacts:
    return DiscussionFacts(
        topic_id=f"topic-{slug}-{source_type}",
        title=slug,
        author="author",
        author_is_host=False,
        votes=3,
        source_type=source_type,
        competition_id=slug,
        text="Offline fixture discussion.",
    )


def _raise(exc: BaseException):
    raise exc
