from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kaggle_researcher.facts import collect
from kaggle_researcher.facts.models import (
    CodeObservation,
    CompetitionMetadata,
    DiscussionFacts,
    FileManifest,
    UserConstraints,
)


def test_collect_facts_runs_stages_in_order_and_clusters_before_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    records = [_record("author/one"), _record("author/two")]

    monkeypatch.setattr(
        collect,
        "fetch_competition_metadata",
        lambda slug: calls.append("metadata") or _metadata(slug),
    )
    monkeypatch.setattr(
        collect,
        "fetch_file_manifest",
        lambda slug, limit: calls.append("files") or _manifest(),
    )
    monkeypatch.setattr(
        collect,
        "list_competition_notebooks",
        lambda slug, limit: calls.append("list_notebooks") or records,
    )
    monkeypatch.setattr(collect, "pull_notebook", _fake_pull)
    monkeypatch.setattr(collect, "extract_observations", lambda path: _observations())
    monkeypatch.setattr(collect, "ast_fingerprint", lambda path: "a" * 64)
    real_assign = collect.assign_lineage_clusters

    def assign(notebooks):
        calls.append(f"assign:{len(notebooks)}")
        return real_assign(notebooks)

    def build_pairs(notebooks):
        calls.append(f"pairs:{len({item.lineage_cluster_id for item in notebooks})}")
        return []

    monkeypatch.setattr(collect, "assign_lineage_clusters", assign)
    monkeypatch.setattr(collect, "build_cv_lb_pairs", build_pairs)
    monkeypatch.setattr(
        collect,
        "fetch_competition_discussions",
        lambda slug, limit: calls.append("discussions") or [_discussion(slug)],
    )
    monkeypatch.setattr(
        collect,
        "fetch_winner_writeups",
        lambda slugs, limit: calls.append("writeups") or [_discussion(slugs[0], True)],
    )

    facts = collect.collect_facts(
        "example",
        max_notebooks=2,
        max_discussions=5,
        similar=["past-comp"],
        user_constraints=UserConstraints(vram_gb=12),
        max_sample_sub_bytes=1000,
    )

    assert calls == [
        "metadata",
        "files",
        "list_notebooks",
        "assign:2",
        "pairs:1",
        "discussions",
        "writeups",
    ]
    assert len(facts.notebooks) == 2
    assert len({item.lineage_cluster_id for item in facts.notebooks}) == 1
    assert [item.source_type for item in facts.discussions] == [
        "discussion",
        "winner_writeup",
    ]
    assert facts.similar_competitions[0].status == "not_computable"
    assert facts.similar_competitions[0].source == "unavailable"
    assert facts.user_constraints.vram_gb == 12
    assert facts.collection_errors == []


def test_nonfatal_stage_failures_are_recorded_and_collection_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(collect, "fetch_competition_metadata", lambda slug: _metadata(slug))
    monkeypatch.setattr(
        collect,
        "fetch_file_manifest",
        lambda slug, limit: _raise(RuntimeError("files unavailable")),
    )
    monkeypatch.setattr(
        collect,
        "list_competition_notebooks",
        lambda slug, limit: _raise(RuntimeError("notebooks unavailable")),
    )
    monkeypatch.setattr(
        collect,
        "fetch_competition_discussions",
        lambda slug, limit: _raise(RuntimeError("forum unavailable")),
    )
    monkeypatch.setattr(
        collect,
        "fetch_winner_writeups",
        lambda slugs, limit: _raise(RuntimeError("writeups unavailable")),
    )

    facts = collect.collect_facts(
        "example",
        3,
        5,
        ["past-comp"],
        UserConstraints(),
        1000,
    )

    assert facts.files.sample_submission_source == "unavailable"
    assert facts.notebooks == []
    assert facts.discussions == []
    assert facts.similar_competitions[0].competition_id == "past-comp"
    assert facts.collection_errors == [
        "file manifest failed (RuntimeError)",
        "notebook listing failed (RuntimeError)",
        "competition discussions failed (RuntimeError)",
        "winner writeups failed (RuntimeError)",
    ]


def test_single_notebook_pull_failure_does_not_drop_successful_notebooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_successful_non_notebook_stages(monkeypatch)
    monkeypatch.setattr(
        collect,
        "list_competition_notebooks",
        lambda slug, limit: [_record("author/broken"), _record("author/good")],
    )

    def pull(ref: str, destination: Path) -> Path | None:
        return None if ref.endswith("broken") else _fake_pull(ref, destination)

    monkeypatch.setattr(collect, "pull_notebook", pull)
    monkeypatch.setattr(collect, "extract_observations", lambda path: _observations())
    monkeypatch.setattr(collect, "ast_fingerprint", lambda path: "b" * 64)

    facts = collect.collect_facts(
        "example", 2, 0, [], UserConstraints(), max_sample_sub_bytes=1000
    )

    assert [notebook.ref for notebook in facts.notebooks] == ["author/good"]
    assert facts.collection_errors == ["notebook author/broken pull failed"]


def test_metadata_failure_is_the_only_fatal_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        collect,
        "fetch_competition_metadata",
        lambda slug: _raise(PermissionError("denied")),
    )

    with pytest.raises(RuntimeError, match="competition metadata collection failed"):
        collect.collect_facts("example", 1, 1, [], UserConstraints(), 1000)


def test_notebook_work_is_concurrent_and_bounded_to_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_successful_non_notebook_stages(monkeypatch)
    records = [_record(f"author/notebook-{index}") for index in range(8)]
    monkeypatch.setattr(
        collect,
        "list_competition_notebooks",
        lambda slug, limit: records,
    )
    lock = threading.Lock()
    active = 0
    peak = 0

    def pull(ref: str, destination: Path) -> Path:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        path = _fake_pull(ref, destination)
        with lock:
            active -= 1
        return path

    monkeypatch.setattr(collect, "pull_notebook", pull)
    monkeypatch.setattr(collect, "extract_observations", lambda path: _observations())
    monkeypatch.setattr(collect, "ast_fingerprint", lambda path: "c" * 64)

    facts = collect.collect_facts("example", 8, 0, [], UserConstraints(), 1000)

    assert len(facts.notebooks) == 8
    assert 1 < peak <= collect.NOTEBOOK_CONCURRENCY == 4


def _patch_successful_non_notebook_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect, "fetch_competition_metadata", lambda slug: _metadata(slug))
    monkeypatch.setattr(collect, "fetch_file_manifest", lambda slug, limit: _manifest())
    monkeypatch.setattr(collect, "fetch_competition_discussions", lambda slug, limit: [])
    monkeypatch.setattr(collect, "fetch_winner_writeups", lambda slugs, limit: [])


def _metadata(slug: str) -> CompetitionMetadata:
    return CompetitionMetadata(
        competition_id=slug,
        metric_name="roc_auc",
        is_code_competition=True,
        unavailable_fields=[],
    )


def _manifest() -> FileManifest:
    return FileManifest(
        files=[],
        train_test_size_ratio=2.0,
        sample_submission_columns=["id", "target"],
        sample_submission_source="api",
        limitations=[],
    )


def _record(ref: str) -> dict[str, Any]:
    return {
        "ref": ref,
        "title": ref,
        "author": ref.split("/", 1)[0],
        "votes": 10,
        "public_score": 0.7,
        "last_run": None,
    }


def _fake_pull(ref: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    return destination / "notebook.ipynb"


def _observations() -> dict[str, Any]:
    return {
        "splitters": [
            CodeObservation(name="GroupKFold", kwargs={"n_splits": "5"}, locator="cell_0")
        ],
        "models": [],
        "metrics": [],
        "feature_ops": [],
        "declared_cv": ["0.71"],
        "parse_status": "ok",
    }


def _discussion(slug: str, writeup: bool = False) -> DiscussionFacts:
    return DiscussionFacts(
        topic_id=f"topic-{slug}",
        title=slug,
        author="author",
        author_is_host=False,
        votes=1,
        source_type="winner_writeup" if writeup else "discussion",
        competition_id=slug,
        text="Useful thread.",
    )


def _raise(exc: BaseException) -> Any:
    raise exc
