from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_researcher import journal, wave


@pytest.fixture(autouse=True)
def isolated_journal_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        journal,
        "JOURNAL_PATH",
        tmp_path / "journal" / "participation.jsonl",
    )
    monkeypatch.setattr(journal, "RUNS_DIR", tmp_path / "runs")


def test_empty_journal_loads_and_summarizes_as_zeros() -> None:
    assert journal.load_entries() == []
    assert journal.summarize() == {
        "entries": 0,
        "briefs_useful": 0,
        "validation_matches": 0,
        "median_percentile": 0,
    }


def test_missing_optional_fields_are_written_as_null() -> None:
    result = journal.append_entry(
        competition_id="example-comp",
        recorded_at="2026-08-03T12:00:00+00:00",
    )

    assert result is None
    assert journal.load_entries() == [
        {
            "competition_id": "example-comp",
            "recorded_at": "2026-08-03T12:00:00+00:00",
            "brief_run_id": None,
            "used_validation": None,
            "recommended_validation": None,
            "validation_matched": None,
            "final_rank": None,
            "num_teams": None,
            "percentile": None,
            "brief_was_useful": None,
            "notes": None,
        }
    ]


def test_appending_never_rewrites_existing_lines() -> None:
    journal.append_entry(
        competition_id="first-comp",
        recorded_at="2026-08-03T10:00:00+00:00",
    )
    original_bytes = journal.JOURNAL_PATH.read_bytes()

    journal.append_entry(
        competition_id="second-comp",
        recorded_at="2026-08-03T11:00:00+00:00",
    )

    updated_bytes = journal.JOURNAL_PATH.read_bytes()
    assert updated_bytes.startswith(original_bytes)
    assert len(journal.load_entries()) == 2


def test_percentile_is_top_rank_percentage() -> None:
    journal.append_entry(
        competition_id="example-comp",
        final_rank=47,
        num_teams=1200,
    )

    assert journal.load_entries()[0]["percentile"] == pytest.approx(
        47 / 1200 * 100
    )


def test_recommended_validation_is_loaded_from_run_and_compared() -> None:
    _write_brief(
        "run-001",
        {"recommended_validation": "GroupKFold(customer_id)"},
    )

    journal.append_entry(
        competition_id="example-comp",
        brief_run_id="run-001",
        used_validation="groupkfold(customer_id)",
    )

    entry = journal.load_entries()[0]
    assert entry["recommended_validation"] == "GroupKFold(customer_id)"
    assert entry["validation_matched"] is True


def test_current_brief_schema_uses_first_validation_claim_as_recommendation() -> None:
    _write_brief(
        "run-002",
        {
            "validation": [
                {
                    "claim_id": "claim_validation",
                    "text": "Use GroupKFold by customer_id.",
                    "source_ids": ["facts"],
                    "kind": "claim",
                }
            ]
        },
    )

    journal.append_entry(
        competition_id="example-comp",
        brief_run_id="run-002",
        used_validation="Random holdout",
    )

    entry = journal.load_entries()[0]
    assert entry["recommended_validation"] == "Use GroupKFold by customer_id."
    assert entry["validation_matched"] is False


def test_missing_used_validation_keeps_match_unknown() -> None:
    _write_brief("run-003", {"recommended_validation": "TimeSeriesSplit"})

    journal.append_entry(
        competition_id="example-comp",
        brief_run_id="run-003",
    )

    assert journal.load_entries()[0]["validation_matched"] is None


@pytest.mark.parametrize(
    ("final_rank", "num_teams", "message"),
    [(0, 100, "positive"), (101, 100, "cannot exceed"), (1, 0, "positive")],
)
def test_invalid_rank_inputs_are_rejected(
    final_rank: int,
    num_teams: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        journal.append_entry(
            competition_id="example-comp",
            final_rank=final_rank,
            num_teams=num_teams,
        )


def test_invalid_optional_field_types_are_rejected() -> None:
    with pytest.raises(ValueError, match="final_rank"):
        journal.append_entry(competition_id="example-comp", final_rank=-1)
    with pytest.raises(ValueError, match="brief_was_useful"):
        journal.append_entry(
            competition_id="example-comp",
            brief_was_useful="yes",
        )


def test_summary_returns_counts_and_median_without_advice() -> None:
    _write_brief("run-004", {"recommended_validation": "GroupKFold"})
    journal.append_entry(
        competition_id="comp-a",
        brief_run_id="run-004",
        used_validation="GroupKFold",
        final_rank=1,
        num_teams=10,
        brief_was_useful=True,
    )
    journal.append_entry(
        competition_id="comp-b",
        final_rank=3,
        num_teams=10,
        brief_was_useful=False,
    )
    journal.append_entry(competition_id="comp-c")

    summary = journal.summarize()

    assert summary == {
        "entries": 3,
        "briefs_useful": 1,
        "validation_matches": 1,
        "median_percentile": 20.0,
    }
    assert set(summary) == {
        "entries",
        "briefs_useful",
        "validation_matches",
        "median_percentile",
    }


def test_wave_journal_writes_entry_and_prints_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_brief("run-cli", {"recommended_validation": "GroupKFold"})

    wave.main(
        [
            "journal",
            "example-comp",
            "--brief-run-id",
            "run-cli",
            "--used-validation",
            "GroupKFold",
            "--final-rank",
            "5",
            "--num-teams",
            "100",
            "--brief-was-useful",
            "yes",
            "--notes",
            "Validation held up.",
        ]
    )

    entry = journal.load_entries()[0]
    assert entry["brief_was_useful"] is True
    assert entry["notes"] == "Validation held up."
    assert entry["percentile"] == 5.0
    assert json.loads(capsys.readouterr().out) == {
        "entries": 1,
        "briefs_useful": 1,
        "validation_matches": 1,
        "median_percentile": 5.0,
    }


def _write_brief(run_id: str, payload: dict) -> None:
    run_dir = journal.RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "brief.json").write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
