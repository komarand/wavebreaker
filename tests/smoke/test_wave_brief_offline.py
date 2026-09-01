from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from kaggle_researcher import brief as brief_module
from kaggle_researcher import wave
from kaggle_researcher.brief import BriefGenerationError
from kaggle_researcher.config import ConfigError, Settings
from kaggle_researcher.facts.models import (
    CompetitionFacts,
    CompetitionMetadata,
    FileManifest,
    UserConstraints,
)


class OfflineBriefClient:
    def __init__(self, payload: dict[str, Any], facts_path: Path) -> None:
        self.payload = payload
        self.facts_path = facts_path
        self.calls: list[dict[str, Any]] = []

    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        assert self.facts_path.exists()
        self.calls.append(kwargs)
        return self.payload


def test_wave_brief_runs_offline_pipeline_and_writes_all_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    collection_calls: list[dict[str, Any]] = []

    def collect_stub(**kwargs: Any) -> CompetitionFacts:
        collection_calls.append(kwargs)
        return _facts(user_constraints=kwargs["user_constraints"])

    monkeypatch.setattr(wave, "collect_facts", collect_stub)
    monkeypatch.setattr(wave, "load_config", lambda: _settings())
    client = OfflineBriefClient(_brief_payload(), tmp_path / "facts.json")
    monkeypatch.setattr(
        brief_module,
        "DeepSeekClient",
        lambda **kwargs: client,
    )
    docx_calls: list[dict[str, Any]] = []

    def docx_stub(**kwargs: Any) -> Path:
        docx_calls.append(kwargs)
        output_path = Path(kwargs["output_path"])
        output_path.write_bytes(b"offline docx placeholder")
        return output_path

    monkeypatch.setattr(wave, "generate_report", docx_stub)

    wave.main(
        [
            "brief",
            "current-comp",
            "--vram",
            "12",
            "--hours",
            "8",
            "--objective",
            "top_percent",
            "--max-notebooks",
            "6",
            "--max-discussions",
            "40",
            "--writeups-per-competition",
            "3",
            "--similar",
            "past-a,past-b",
            "--docx",
            "--out",
            str(tmp_path),
        ]
    )

    facts_payload = json.loads((tmp_path / "facts.json").read_text(encoding="utf-8"))
    brief_payload = json.loads((tmp_path / "brief.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "brief.md").read_text(encoding="utf-8")
    assert len(collection_calls) == 1
    assert collection_calls[0]["max_notebooks"] == 6
    assert collection_calls[0]["max_discussions"] == 40
    assert collection_calls[0]["writeups_per_competition"] == 3
    assert collection_calls[0]["similar"] == ["past-a", "past-b"]
    assert collection_calls[0]["user_constraints"] == UserConstraints(
        vram_gb=12,
        hours_per_week=8,
        objective="top_percent",
    )
    assert facts_payload["user_constraints"]["vram_gb"] == 12
    assert brief_payload["metric_notes"] == []
    assert brief_payload["prompt_version"] == "2026-09-01.2"
    assert brief_payload["claim_stats"] == {
        "fact": 1,
        "claim": 0,
        "inference": 0,
        "total": 1,
        "grounded": 1,
        "ungrounded": 0,
        "grounding_rate": 1.0,
        "distinct_sources": 1,
        "by_evidence_strength": {
            "official": 0,
            "measured_with_protocol": 1,
            "reported_score": 0,
            "prevalence": 0,
            "inference": 0,
        },
        "hypotheses_total": 0,
        "hypotheses_dropped_unverifiable": 0,
    }
    assert "unsupported: Fabricated metric claim." in brief_payload["unknowns"]
    assert "## 1. Соревнование в цифрах" in markdown
    assert "## 10. Неизвестное" in markdown
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == "offline-pro"
    assert len(docx_calls) == 1
    assert docx_calls[0]["roadmap_text"] == markdown
    assert docx_calls[0]["sources"] == []
    assert docx_calls[0]["overwrite"] is True
    assert (tmp_path / "brief.docx").exists()
    output = capsys.readouterr().out
    assert f"brief json: {tmp_path / 'brief.json'}" in output
    assert f"brief docx: {tmp_path / 'brief.docx'}" in output
    assert "context: 0 of 0 discussions included, 0 truncated, budget 10000 tokens" in output
    assert "claims: 1 (fact 1, claim 0, inference 0)" in output
    assert "grounding rate: 1.0 across 1 sources" in output
    assert (
        "evidence: official 0, measured 1, reported 0, prevalence 0, inference 0"
        in output
    )


def test_facts_from_skips_collection_and_checkpoints_before_model_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    source_path = source_dir / "facts.json"
    facts = _facts()
    source_path.write_text(facts.model_dump_json(indent=2), encoding="utf-8")

    def unexpected_collection(**kwargs: Any) -> CompetitionFacts:
        raise AssertionError("collect_facts must not run with --facts-from")

    monkeypatch.setattr(wave, "collect_facts", unexpected_collection)
    monkeypatch.setattr(wave, "load_config", lambda: _settings())
    client = OfflineBriefClient(_brief_payload(), output_dir / "facts.json")
    monkeypatch.setattr(
        brief_module,
        "DeepSeekClient",
        lambda **kwargs: client,
    )

    wave.main(
        [
            "brief",
            "ignored-slug",
            "--facts-from",
            str(source_path),
            "--out",
            str(output_dir),
        ]
    )

    checkpoint = CompetitionFacts.model_validate_json(
        (output_dir / "facts.json").read_text(encoding="utf-8")
    )
    assert checkpoint == facts
    assert (output_dir / "brief.json").exists()
    assert (output_dir / "brief.md").exists()
    assert len(client.calls) == 1


def test_model_failure_still_writes_facts_only_markdown_and_returns_successfully(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wave, "collect_facts", lambda **kwargs: _facts())
    monkeypatch.setattr(wave, "load_config", lambda: _settings())

    async def failed_generation(facts: CompetitionFacts, settings: Settings):
        assert (tmp_path / "facts.json").exists()
        raise BriefGenerationError("forced offline failure")

    monkeypatch.setattr(wave, "generate_brief", failed_generation)

    assert wave.main(["brief", "current-comp", "--out", str(tmp_path)]) is None

    markdown = (tmp_path / "brief.md").read_text(encoding="utf-8")
    assert (tmp_path / "facts.json").exists()
    assert not (tmp_path / "brief.json").exists()
    assert markdown.startswith("## 1. Соревнование в цифрах")
    assert "Full competition brief generation was unavailable" in markdown
    assert "BriefGenerationError" in markdown
    assert "## 2. Тезис" not in markdown


def test_missing_model_configuration_also_preserves_facts_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wave, "collect_facts", lambda **kwargs: _facts())
    monkeypatch.setattr(
        wave,
        "load_config",
        lambda: _raise(ConfigError("DEEPSEEK_API_KEY is required")),
    )

    wave.main(["brief", "current-comp", "--out", str(tmp_path)])

    assert (tmp_path / "facts.json").exists()
    assert (tmp_path / "brief.md").exists()
    assert not (tmp_path / "brief.json").exists()
    assert "ConfigError" in (tmp_path / "brief.md").read_text(encoding="utf-8")


def _settings() -> Settings:
    return Settings(
        deepseek_api_key="offline-key",
        deepseek_v4_pro="offline-pro",
        max_context_tokens=10_000,
    )


def _facts(
    *,
    user_constraints: UserConstraints | None = None,
) -> CompetitionFacts:
    return CompetitionFacts(
        competition_id="current-comp",
        collected_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        metadata=CompetitionMetadata(
            competition_id="current-comp",
            title="Current Competition",
            metric_name="roc_auc",
            is_code_competition=True,
            unavailable_fields=[],
        ),
        files=FileManifest(
            files=[],
            sample_submission_columns=[],
            sample_submission_source="unavailable",
            limitations=[],
        ),
        notebooks=[],
        discussions=[],
        similar_competitions=[],
        cv_lb_pairs=[],
        user_constraints=user_constraints or UserConstraints(),
        collection_errors=[],
    )


def _brief_payload() -> dict[str, Any]:
    return {
        "competition_id": "current-comp",
        "thesis": "Use official metric facts to anchor validation.",
        "thesis_support": ["claim_validation"],
        "validation": [
            {
                "claim_id": "claim_validation",
                "text": "The competition metric is roc_auc.",
                "source_ids": ["facts"],
                "kind": "fact",
                "evidence_strength": "measured_with_protocol",
            }
        ],
        "metric_notes": [
            {
                "claim_id": "claim_metric",
                "text": "Fabricated metric claim.",
                "source_ids": ["fabricated-source"],
                "kind": "claim",
                "evidence_strength": "reported_score",
            }
        ],
        "leakage_risks": [],
        "what_works": [],
        "time_wasters": [],
        "hypotheses": [],
        "eda_tasks": [],
        "first_moves": ["Inspect the file manifest."],
        "unknowns": [],
        "limitations": [],
    }


def _raise(error: Exception):
    raise error
