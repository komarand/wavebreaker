from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from kaggle_researcher import main as main_module
from kaggle_researcher.main import build_parser, run_research, validate_full_roadmap
from kaggle_researcher.reasoning.report_composer import SECTION_HEADINGS
from kaggle_researcher.schemas import (
    ExperimentItem,
    LeaderboardAuditResult,
    LeakageRiskResult,
    MetricResult,
    PlanData,
    RetrievedDocument,
    ReviewResult,
    SourceDocument,
    ValidationResult,
)


@dataclass(slots=True)
class FakeSettings:
    deepseek_api_key: str = "secret"
    deepseek_v4_pro: str = "deepseek-v4-pro"
    deepseek_v4_flash: str = "deepseek-v4-flash"
    embed_model: str = "Qwen/Qwen3-Embedding-4B"
    embed_dim: int = 2
    max_embed_batch_size: int = 2
    pg_dsn: str = "postgresql://example"
    top_k: int = 2
    max_notebooks: int = 1
    max_papers: int = 1
    max_repos: int = 1
    pdf_cache_dir: str = "./data/pdfs"
    github_token: str | None = None


class FakeClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key


class FakeStore:
    def __init__(self, competition_id: str, dsn: str, embed_dim: int) -> None:
        self.competition_id = competition_id

    async def init(self) -> None:
        return None

    async def upsert(self, docs: list[SourceDocument], embeddings: list[list[float]]) -> None:
        return None

    async def close(self) -> None:
        return None


class FakeDomainMemory:
    async def init(self) -> None:
        return None

    async def find_similar(self, task_type: str, domain: str, top_k: int = 5) -> list[dict[str, Any]]:
        return []

    async def close(self) -> None:
        return None


def run(coro):
    return asyncio.run(coro)


def source_doc() -> SourceDocument:
    return SourceDocument(
        id="doc-1",
        competition_id="comp-1",
        source="kaggle",
        title="Notebook",
        url="https://example.com/notebook",
        content="notebook content",
        summary="notebook summary",
    )


def retrieved_doc() -> RetrievedDocument:
    return RetrievedDocument(
        id="doc-1",
        competition_id="comp-1",
        source="kaggle",
        title="Notebook",
        url="https://example.com/notebook",
        content="retrieved content",
        score=0.9,
        rrf_score=0.1,
    )


def install_pipeline_mocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    report_texts: list[str] = []

    async def fake_plan(description: str, client: Any, model: str) -> PlanData:
        return PlanData(
            task_type="classification",
            metric="gini",
            domain="credit",
            kaggle_queries=["credit kaggle"],
            arxiv_queries=[],
            github_queries=[],
        )

    async def fake_collect_sources(**kwargs: Any) -> list[SourceDocument]:
        return [source_doc()]

    async def fake_summarize_documents(**kwargs: Any) -> list[SourceDocument]:
        return [source_doc()]

    async def fake_retrieve_documents(**kwargs: Any) -> list[RetrievedDocument]:
        return [retrieved_doc()]

    def fake_generate_report(
        competition_name: str,
        roadmap_text: str,
        sources: list[RetrievedDocument],
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        report_texts.append(roadmap_text)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("docx placeholder", encoding="utf-8")
        return path

    monkeypatch.setattr(main_module, "load_config", lambda: FakeSettings())
    monkeypatch.setattr(main_module, "DeepSeekClient", FakeClient)
    monkeypatch.setattr(main_module, "PgStore", FakeStore)
    monkeypatch.setattr(main_module, "DomainMemory", lambda dsn, embed_dim: FakeDomainMemory())
    monkeypatch.setattr(main_module, "plan", fake_plan)
    monkeypatch.setattr(main_module, "_collect_sources", fake_collect_sources)
    monkeypatch.setattr(main_module, "summarize_documents", fake_summarize_documents)
    monkeypatch.setattr(main_module, "_embed_documents", lambda texts, batch_size, show_progress: [[0.1, 0.2]])
    monkeypatch.setattr(main_module, "_retrieve_documents", fake_retrieve_documents)
    monkeypatch.setattr(main_module, "generate_report", fake_generate_report)
    monkeypatch.setattr(main_module, "_create_run_dir", lambda competition_id: tmp_path / "runs" / competition_id)
    return report_texts


def test_cli_default_report_mode_is_full() -> None:
    args = build_parser().parse_args([])

    assert args.report_mode == "full"
    assert args.allow_minimal_fallback is False


def test_validate_full_roadmap_rejects_minimal_report() -> None:
    with pytest.raises(RuntimeError, match="minimal/fallback"):
        validate_full_roadmap("# Minimal Research Report\n\n## Next Steps")


def test_validate_full_roadmap_accepts_mock_15_section_roadmap() -> None:
    roadmap = "\n\n".join(f"## {heading}\n" + ("Detailed guidance. " * 25) for heading in SECTION_HEADINGS)

    validate_full_roadmap(roadmap)


def test_full_mode_reasoning_failure_is_not_silent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_texts = install_pipeline_mocks(monkeypatch, tmp_path)

    async def failing_full_report(**kwargs: Any) -> str:
        raise RuntimeError("Full report generation failed at stage 'metric_specialist': broken")

    monkeypatch.setattr(main_module, "_build_full_report_text", failing_full_report)

    with pytest.raises(RuntimeError, match="metric_specialist"):
        run(
            run_research(
                "https://www.kaggle.com/competitions/comp-1",
                "Credit risk competition",
                show_progress=False,
            )
        )

    assert report_texts == []


def test_minimal_mode_only_when_explicitly_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_texts = install_pipeline_mocks(monkeypatch, tmp_path)

    result = run(
        run_research(
            "https://www.kaggle.com/competitions/comp-1",
            "Credit risk competition",
            report_mode="minimal",
            show_progress=False,
        )
    )

    assert result.report_mode == "minimal"
    assert "Minimal Research Report" in report_texts[0]


def test_models_used_artifact_contains_configured_model_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_pipeline_mocks(monkeypatch, tmp_path)

    result = run(
        run_research(
            "https://www.kaggle.com/competitions/comp-1",
            "Credit risk competition",
            report_mode="minimal",
            show_progress=False,
        )
    )

    models_path = Path(result.run_artifacts_path) / "models_used.json"
    payload = models_path.read_text(encoding="utf-8")
    assert '"planner": "deepseek-v4-pro"' in payload
    assert '"summarizer": "deepseek-v4-flash"' in payload
    assert '"report_composer": "deepseek-v4-pro"' in payload
    assert '"embedder": "Qwen/Qwen3-Embedding-4B"' in payload


def test_full_report_passes_artifact_dir_only_to_reviewer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen_calls: dict[str, dict[str, Any]] = {}

    async def fake_design_validation(**kwargs: Any) -> ValidationResult:
        seen_calls["validation"] = kwargs
        return ValidationResult(
            confidence="medium",
            evidence_ids=[],
            recommended_cv="temporal",
            validation_risk="high",
            likely_split="time",
            reasoning="reason",
            primary_validation={"method": "temporal_cv"},
        )

    async def fake_analyze_leakage_risk(**kwargs: Any) -> LeakageRiskResult:
        return LeakageRiskResult(confidence="low", evidence_ids=[], risk_level="medium")

    async def fake_analyze_metric(**kwargs: Any) -> MetricResult:
        return MetricResult(
            confidence="medium",
            evidence_ids=[],
            metric_explanation="gini",
            needs_calibration=False,
            rank_averaging_useful=True,
            threshold_search_needed=False,
            surrogate_loss_suggestion="auc",
        )

    async def fake_plan_experiments(**kwargs: Any) -> list[ExperimentItem]:
        return [
            ExperimentItem(
                priority="P0",
                experiment="baseline",
                why="why",
                cost="low",
                expected_gain="medium",
                risk="low",
            )
        ]

    async def fake_audit_leaderboard_risk(**kwargs: Any) -> LeaderboardAuditResult:
        return LeaderboardAuditResult(
            confidence="medium",
            evidence_ids=[],
            shake_up_risk="high",
            submission_selection_rule="cv",
            public_lb_trust="low",
        )

    async def fake_review(**kwargs: Any) -> ReviewResult:
        seen_calls["review"] = kwargs
        return ReviewResult(confidence="medium", evidence_ids=[])

    async def fake_compose_report(**kwargs: Any) -> str:
        return "\n\n".join(
            f"## {heading}\n" + ("Detailed guidance. " * 25)
            for heading in SECTION_HEADINGS
        )

    monkeypatch.setattr(main_module, "design_validation", fake_design_validation)
    monkeypatch.setattr(main_module, "analyze_leakage_risk", fake_analyze_leakage_risk)
    monkeypatch.setattr(main_module, "analyze_metric", fake_analyze_metric)
    monkeypatch.setattr(main_module, "plan_experiments", fake_plan_experiments)
    monkeypatch.setattr(main_module, "audit_leaderboard_risk", fake_audit_leaderboard_risk)
    monkeypatch.setattr(main_module, "review", fake_review)
    monkeypatch.setattr(main_module, "compose_report", fake_compose_report)

    run(
        main_module._build_full_report_text(
            competition_desc="desc",
            plan_data=PlanData(task_type="classification", metric="gini", domain="credit"),
            retrieved_documents=[retrieved_doc()],
            domain_memory=FakeDomainMemory(),
            client=FakeClient(api_key="secret"),
            model="model",
            run_dir=tmp_path,
            show_progress=False,
        )
    )

    assert "artifact_dir" not in seen_calls["validation"]
    assert seen_calls["review"]["artifact_dir"] == tmp_path
