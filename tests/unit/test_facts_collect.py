from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kaggle_researcher.facts import collect
from kaggle_researcher.facts.cv_lb import CvLbPairList
from kaggle_researcher.facts.discussions import DiscussionCollection
from kaggle_researcher.facts.models import (
    CodeObservation,
    CompetitionMetadata,
    CvLbPair,
    DiscussionFacts,
    FileManifest,
    PublicLeaderboard,
    UserConstraints,
)


@pytest.fixture(autouse=True)
def public_leaderboard_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        collect,
        "fetch_public_leaderboard",
        lambda slug: PublicLeaderboard(
            status="collected",
            entries=[],
            entry_count=0,
            unavailable_reason=None,
        ),
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
    monkeypatch.setattr(
        collect,
        "extract_observations",
        lambda path, **kwargs: {
            **_observations(),
            "style_markup_stripped_cells": 1,
            "style_markup_stripped_markdown_cells": 1,
            "style_markup_stripped_code_strings": 0,
        },
    )
    monkeypatch.setattr(collect, "ast_fingerprint", lambda path: "a" * 64)
    real_assign = collect.assign_lineage_clusters

    def assign(notebooks):
        calls.append(f"assign:{len(notebooks)}")
        return real_assign(notebooks)

    def build_pairs(notebooks, competition_metric_name=None):
        assert competition_metric_name == "roc_auc"
        calls.append(f"pairs:{len({item.lineage_cluster_id for item in notebooks})}")
        rejected = CvLbPair(
            notebook_ref="author/one",
            declared_cv=0.7,
            public_score=0.8,
            lineage_cluster_id="lc_rejected",
            metric_canonical="roc_auc",
            gap=-0.1,
            absolute_gap=0.1,
            comparability_status="implausible_gap",
        )
        return CvLbPairList([], implausible_gap_pairs=[rejected])

    monkeypatch.setattr(collect, "assign_lineage_clusters", assign)
    monkeypatch.setattr(
        collect,
        "fetch_public_leaderboard",
        lambda slug: calls.append("leaderboard")
        or PublicLeaderboard(
            status="collected",
            entries=[],
            entry_count=0,
            unavailable_reason=None,
        ),
    )
    monkeypatch.setattr(collect, "build_cv_lb_pairs", build_pairs)
    monkeypatch.setattr(
        collect,
        "fetch_competition_discussions",
        lambda slug, limit: calls.append("discussions") or [_discussion(slug)],
    )
    monkeypatch.setattr(
        collect,
        "fetch_winner_writeups",
        lambda slugs, limit: calls.append(f"writeups:{limit}") or [_discussion(slugs[0], True)],
    )

    facts = collect.collect_facts(
        "example",
        max_notebooks=2,
        max_discussions=5,
        similar=["past-comp"],
        user_constraints=UserConstraints(vram_gb=12),
        max_sample_sub_bytes=1000,
        writeups_per_competition=10,
    )

    assert calls == [
        "metadata",
        "files",
        "list_notebooks",
        "assign:2",
        "leaderboard",
        "pairs:1",
        "discussions",
        "metadata",
        "writeups:10",
    ]
    assert len(facts.notebooks) == 2
    assert len({item.lineage_cluster_id for item in facts.notebooks}) == 1
    assert [item.source_type for item in facts.discussions] == [
        "discussion",
        "winner_writeup",
    ]
    assert facts.similar_competitions[0].status == "not_computable"
    assert facts.similar_competitions[0].source == "unavailable"
    assert (
        facts.similar_competitions[0].not_computable_reason == "Meta Kaggle dumps not configured."
    )
    assert "task" not in facts.similar_competitions[0].not_computable_reason.lower()
    assert facts.user_constraints.vram_gb == 12
    assert facts.collection_errors == []
    assert facts.discussion_collection_status == "collected"
    assert facts.cv_lb_diagnostics.notebooks_total == 2
    assert facts.cv_lb_diagnostics.notebooks_with_both == 2
    assert facts.score_diagnostics.style_markup_stripped_cells == 2
    assert facts.score_diagnostics.style_markup_stripped_markdown_cells == 2
    assert facts.score_diagnostics.style_markup_stripped_code_strings == 0
    assert [pair.notebook_ref for pair in facts.implausible_gap_pairs] == [
        "author/one"
    ]


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


def test_unavailable_public_leaderboard_is_recorded_and_collection_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_successful_non_notebook_stages(monkeypatch)
    monkeypatch.setattr(
        collect,
        "fetch_public_leaderboard",
        lambda slug: PublicLeaderboard(
            status="unavailable",
            entries=[],
            entry_count=0,
            unavailable_reason="HTTPError: forbidden",
        ),
    )

    facts = collect.collect_facts(
        "example",
        0,
        0,
        [],
        UserConstraints(),
        max_sample_sub_bytes=1000,
    )

    assert facts.public_leaderboard.status == "unavailable"
    assert facts.cv_lb_pairs == []
    assert facts.collection_errors == [
        "public leaderboard unavailable (HTTPError: forbidden)"
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
    monkeypatch.setattr(collect, "extract_observations", lambda path, **kwargs: _observations())
    monkeypatch.setattr(collect, "ast_fingerprint", lambda path: "b" * 64)

    facts = collect.collect_facts("example", 2, 0, [], UserConstraints(), max_sample_sub_bytes=1000)

    assert [notebook.ref for notebook in facts.notebooks] == ["author/good"]
    assert facts.collection_errors == ["notebook author/broken pull failed"]


def test_notebook_pull_failures_are_counted_by_http_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_successful_non_notebook_stages(monkeypatch)
    records = [
        _record("author/rate-limited"),
        _record("author/private"),
        _record("author/ssl"),
        _record("author/good"),
    ]
    monkeypatch.setattr(
        collect,
        "list_competition_notebooks",
        lambda slug, limit: records,
    )

    def pull(ref: str, destination: Path) -> collect.NotebookPullResult:
        if ref.endswith("rate-limited"):
            return collect.NotebookPullResult(
                path=None,
                http_status=429,
                attempt=6,
                max_attempts=6,
                error_type="HTTPError",
            )
        if ref.endswith("private"):
            return collect.NotebookPullResult(
                path=None,
                http_status=404,
                attempt=1,
                max_attempts=6,
                error_type="HTTPError",
            )
        if ref.endswith("ssl"):
            return collect.NotebookPullResult(
                path=None,
                attempt=6,
                max_attempts=6,
                error_type="SSLError",
            )
        return collect.NotebookPullResult(path=_fake_pull(ref, destination))

    monkeypatch.setattr(collect, "pull_notebook", pull)
    monkeypatch.setattr(collect, "extract_observations", lambda path, **kwargs: _observations())
    monkeypatch.setattr(collect, "ast_fingerprint", lambda path: "b" * 64)

    facts = collect.collect_facts("example", 4, 0, [], UserConstraints(), max_sample_sub_bytes=1000)

    assert [notebook.ref for notebook in facts.notebooks] == ["author/good"]
    assert facts.score_diagnostics.notebooks_failed_by_status == {404: 1, 429: 1}
    assert facts.score_diagnostics.notebooks_failed_by_exception == {
        "HTTPError": 2,
        "SSLError": 1,
    }
    assert facts.collection_errors == [
        "notebook author/rate-limited pull failed (HTTP 429, attempt 6/6)",
        "notebook author/private pull failed (HTTP 404, attempt 1/6)",
        "notebook author/ssl pull failed (HTTP unknown, attempt 6/6)",
    ]


def test_title_and_ref_scores_are_collected_once_with_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_successful_non_notebook_stages(monkeypatch)
    record = _record("rommelsharma/0-95-jaguar-re-id-frozen-dinov2-giant")
    record["title"] = "0.95 Jaguar Re-ID Frozen DINOv2 Giant"
    monkeypatch.setattr(
        collect,
        "list_competition_notebooks",
        lambda slug, limit: [record],
    )
    monkeypatch.setattr(collect, "pull_notebook", _fake_pull)
    monkeypatch.setattr(collect, "extract_observations", lambda path, **kwargs: _observations())
    monkeypatch.setattr(collect, "ast_fingerprint", lambda path: "f" * 64)

    facts = collect.collect_facts("example", 1, 0, [], UserConstraints(), max_sample_sub_bytes=1000)

    observations = facts.notebooks[0].score_observations
    assert len(observations) == 1
    assert observations[0].value == pytest.approx(0.95)
    assert observations[0].source == "title"
    assert facts.score_diagnostics.observations_total == 1
    assert facts.score_diagnostics.title_or_ref_observations == 1
    assert facts.score_diagnostics.candidates_seen == 2


def test_competition_metric_is_passed_to_notebook_score_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_successful_non_notebook_stages(monkeypatch)
    monkeypatch.setattr(
        collect,
        "list_competition_notebooks",
        lambda slug, limit: [_record("author/one")],
    )
    monkeypatch.setattr(collect, "pull_notebook", _fake_pull)
    received_hints: list[tuple[str, ...]] = []

    def extract(path: Path, *, metric_hints: tuple[str, ...]) -> dict[str, Any]:
        received_hints.append(metric_hints)
        return _observations()

    monkeypatch.setattr(collect, "extract_observations", extract)
    monkeypatch.setattr(collect, "ast_fingerprint", lambda path: "f" * 64)

    collect.collect_facts("example", 1, 0, [], UserConstraints(), max_sample_sub_bytes=1000)

    assert received_hints == [("roc_auc",)]


def test_discussion_forbidden_is_recorded_without_failing_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_successful_non_notebook_stages(monkeypatch)
    monkeypatch.setattr(collect, "list_competition_notebooks", lambda slug, limit: [])
    monkeypatch.setattr(
        collect,
        "fetch_competition_discussions",
        lambda slug, limit: DiscussionCollection(
            status="forbidden",
            error="Kaggle Discussion API returned HTTP 403.",
            limitation="Kaggle Discussion API returned HTTP 403.",
        ),
    )

    facts = collect.collect_facts("example", 0, 5, [], UserConstraints(), max_sample_sub_bytes=1000)

    assert facts.discussions == []
    assert facts.discussion_collection_status == "forbidden"
    assert facts.discussion_collection_error == ("Kaggle Discussion API returned HTTP 403.")
    assert facts.limitations == ["Kaggle Discussion API returned HTTP 403."]
    assert facts.collection_errors == []


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


def test_notebook_work_is_concurrent_and_bounded_to_two(
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
    monkeypatch.setattr(collect, "extract_observations", lambda path, **kwargs: _observations())
    monkeypatch.setattr(collect, "ast_fingerprint", lambda path: "c" * 64)

    facts = collect.collect_facts(
        "example",
        8,
        0,
        [],
        UserConstraints(),
        1000,
        notebook_concurrency=2,
    )

    assert len(facts.notebooks) == 8
    assert peak == 2


def test_concurrency_one_collects_multiple_notebooks_without_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_successful_non_notebook_stages(monkeypatch)
    records = [_record(f"author/notebook-{index}") for index in range(3)]
    monkeypatch.setattr(
        collect,
        "list_competition_notebooks",
        lambda slug, limit: records,
    )
    monkeypatch.setattr(collect, "pull_notebook", _fake_pull)
    monkeypatch.setattr(collect, "extract_observations", lambda path, **kwargs: _observations())
    monkeypatch.setattr(collect, "ast_fingerprint", lambda path: "d" * 64)
    progress_updates: list[int] = []
    progress_postfixes: list[dict[str, int]] = []

    class RecordingProgress:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["total"] == 3
            assert kwargs["desc"] == "Pulling notebooks"
            assert kwargs["unit"] == "notebook"

        def update(self, amount: int) -> None:
            progress_updates.append(amount)

        def set_postfix(self, **kwargs: int) -> None:
            progress_postfixes.append(kwargs)

        def close(self) -> None:
            pass

    monkeypatch.setattr(collect, "tqdm", RecordingProgress)

    facts = collect.collect_facts(
        "example",
        3,
        0,
        [],
        UserConstraints(),
        1000,
        notebook_concurrency=1,
    )

    assert [notebook.ref for notebook in facts.notebooks] == [
        "author/notebook-0",
        "author/notebook-1",
        "author/notebook-2",
    ]
    assert progress_updates == [1, 1, 1]
    assert progress_postfixes[-1] == {"ok": 3, "failed": 0}


def test_download_semaphore_is_not_reacquired_for_ast_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class CountingSemaphore:
        entries = 0

        async def __aenter__(self):
            self.entries += 1
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    semaphore = CountingSemaphore()

    def pull(ref: str, destination: Path) -> Path:
        calls.append("pull")
        return _fake_pull(ref, destination)

    def extract(path: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append("extract")
        return _observations()

    def fingerprint(path: Path) -> str:
        calls.append("fingerprint")
        return "e" * 64

    monkeypatch.setattr(collect, "pull_notebook", pull)
    monkeypatch.setattr(collect, "extract_observations", extract)
    monkeypatch.setattr(collect, "ast_fingerprint", fingerprint)

    result = asyncio.run(
        collect._collect_one_notebook(
            record=_record("author/one"),
            destination=tmp_path,
            semaphore=semaphore,  # type: ignore[arg-type]
        )
    )

    assert semaphore.entries == 1
    assert calls == ["pull", "extract", "fingerprint"]
    assert not isinstance(result, str)
    assert result.ast_fingerprint == "e" * 64
    assert result.splitters == _observations()["splitters"]


def test_discussion_and_writeup_limits_are_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_successful_non_notebook_stages(monkeypatch)
    monkeypatch.setattr(collect, "list_competition_notebooks", lambda slug, limit: [])
    calls: list[tuple[str, object, int]] = []

    def discussions(slug: str, limit: int) -> list[DiscussionFacts]:
        calls.append(("discussions", slug, limit))
        discussion = _discussion(slug)
        discussion.text = "competitions/auto-detected"
        return [discussion]

    def writeups(slugs: list[str], limit: int) -> list[DiscussionFacts]:
        calls.append(("writeups", tuple(slugs), limit))
        return [_discussion(slug, True) for slug in slugs]

    monkeypatch.setattr(collect, "fetch_competition_discussions", discussions)
    monkeypatch.setattr(collect, "fetch_winner_writeups", writeups)

    facts = collect.collect_facts(
        "example",
        max_notebooks=0,
        max_discussions=200,
        similar=["past-a", "past-b", "past-c"],
        user_constraints=UserConstraints(),
        max_sample_sub_bytes=1000,
        writeups_per_competition=10,
    )

    assert calls == [
        ("discussions", "example", 200),
        ("writeups", ("past-a", "past-b", "past-c"), 10),
    ]
    writeups = [item for item in facts.discussions if item.source_type == "winner_writeup"]
    assert len(writeups) == 3
    assert all(item.status == "not_computable" for item in facts.similar_competitions)
    assert any(item.slug == "auto-detected" for item in facts.similar_candidates)


@pytest.mark.parametrize("limit", [0, -1])
def test_collect_rejects_nonpositive_writeup_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="writeups_per_competition"):
        collect.collect_facts(
            "example",
            0,
            0,
            [],
            UserConstraints(),
            1000,
            writeups_per_competition=limit,
        )


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
