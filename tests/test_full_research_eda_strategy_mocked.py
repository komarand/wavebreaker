from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from kaggle_researcher import main as main_module
from kaggle_researcher.main import run_research
from kaggle_researcher.reasoning.final_synthesizer import (
    FinalStrategyResult,
    REQUIRED_SECTION_IDS,
)
from kaggle_researcher.research_scout.schemas import (
    EdaTaskPlanDraft,
    ResearchScoutOutput,
    ScoutEdaTask,
    ScoutHypothesis,
)
from kaggle_researcher.schemas import PlanData, RetrievedDocument, SourceDocument
from kaggle_researcher.schemas import ResearchRunResult
from kaggle_researcher.workflow import FinalSynthesisDegradedError


@dataclass(slots=True)
class FakeSettings:
    deepseek_api_key: str = "secret"
    deepseek_v4_pro: str = "deepseek-v4-pro"
    deepseek_v4_flash: str = "deepseek-v4-flash"
    embed_model: str = "fake-embedder"
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
    calls: list[dict[str, Any]] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def chat_json(self, **kwargs: Any) -> dict[str, Any]:
        FakeClient.calls.append(kwargs)
        payload = _final_strategy_payload()
        prompt = json.loads(kwargs["user_prompt"])
        payload["acknowledged_risk_ids"] = prompt.get("allowed_risk_ids", [])
        payload["selected_validation_requirement_ids"] = prompt.get(
            "allowed_validation_requirement_ids", []
        )
        payload["enforced_safety_constraint_ids"] = prompt.get(
            "allowed_safety_constraint_ids", []
        )
        return payload


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
    def __init__(self, dsn: str, embed_dim: int) -> None:
        return None

    async def init(self) -> None:
        return None

    async def find_similar(self, task_type: str, domain: str, top_k: int = 5) -> list[dict[str, Any]]:
        return [{"competition_family": "generic_tabular"}]

    async def close(self) -> None:
        return None


def run(coro):
    return asyncio.run(coro)


def test_full_mocked_research_to_eda_to_strategy_workflow(monkeypatch, tmp_path: Path) -> None:
    FakeClient.calls = []
    fixture_path = Path("tests/fixtures/eda/iid_binary_tiny")

    async def fake_plan(description: str, client: Any, model: str) -> PlanData:
        return PlanData(
            task_type="binary_classification",
            metric="roc_auc",
            domain="generic_tabular",
            kaggle_queries=["iid binary roc auc"],
            arxiv_queries=[],
            github_queries=[],
        )

    async def fake_collect_sources(**kwargs: Any) -> list[SourceDocument]:
        return [
            SourceDocument(
                id="source-1",
                competition_id=kwargs["competition_id"],
                source="kaggle",
                title="Generic notebook",
                url="https://example.com/notebook",
                content="Uses iid binary classification and ROC AUC.",
            )
        ]

    async def fake_summarize_documents(**kwargs: Any) -> list[SourceDocument]:
        return kwargs["docs"]

    async def fake_retrieve_documents(**kwargs: Any) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                id="retrieved-1",
                competition_id=kwargs["competition_id"],
                source="kaggle",
                title="Retrieved generic evidence",
                url="https://example.com/evidence",
                content="Ordinary iid binary classification should use stratified folds.",
                score=0.9,
                rrf_score=0.2,
            )
        ]

    async def fake_run_research_scout(**kwargs: Any) -> ResearchScoutOutput:
        return _scout_output(
            competition_id=kwargs["competition_id"],
            competition_url=kwargs["competition_url"],
        )

    def fake_generate_report(
        competition_name: str,
        roadmap_text: str,
        sources: list[RetrievedDocument],
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("docx placeholder", encoding="utf-8")
        return path

    monkeypatch.setattr(main_module, "load_config", lambda: FakeSettings())
    monkeypatch.setattr(main_module, "DeepSeekClient", FakeClient)
    monkeypatch.setattr(main_module, "PgStore", FakeStore)
    monkeypatch.setattr(main_module, "DomainMemory", FakeDomainMemory)
    monkeypatch.setattr(main_module, "plan", fake_plan)
    monkeypatch.setattr(main_module, "_collect_sources", fake_collect_sources)
    monkeypatch.setattr(main_module, "summarize_documents", fake_summarize_documents)
    monkeypatch.setattr(main_module, "_embed_documents", lambda texts, batch_size, show_progress: [[0.1, 0.2] for _ in texts])
    monkeypatch.setattr(main_module, "_retrieve_documents", fake_retrieve_documents)
    monkeypatch.setattr(main_module, "run_research_scout", fake_run_research_scout)
    monkeypatch.setattr(main_module, "generate_report", fake_generate_report)
    monkeypatch.setattr(main_module, "_create_run_dir", lambda competition_id: tmp_path / "runs" / competition_id)

    result = run(
        run_research(
            "https://www.kaggle.com/competitions/iid_binary_tiny",
            "Generic iid binary classification with ROC AUC.",
            competition_id="iid_binary_tiny",
            output_dir=tmp_path / "reports",
            report_mode="minimal",
            show_progress=False,
            write_eda_plan=True,
            execute_eda=True,
            local_dataset_path=fixture_path,
            eda_output_dir=tmp_path / "eda_runs",
            final_synthesis=True,
        )
    )

    run_path = Path(result.run_artifacts_path)
    research_run = json.loads((run_path / "research_run.json").read_text(encoding="utf-8"))
    evidence_pack = json.loads(Path(result.eda_evidence_pack_path).read_text(encoding="utf-8"))
    final_strategy = json.loads(Path(result.final_strategy_path).read_text(encoding="utf-8"))

    assert (run_path / "research_run.json").is_file()
    assert (run_path / "research_hypotheses.json").is_file()
    assert (run_path / "eda_task_plan.json").is_file()
    assert Path(result.eda_evidence_pack_path).is_file()
    assert Path(result.final_strategy_path).is_file()
    assert Path(result.final_strategy_summary_path).is_file()
    assert (run_path / "final_strategy_raw_payload.json").is_file()
    assert (run_path / "final_strategy_normalized_payload.json").is_file()
    assert (run_path / "final_strategy_action_support_report.json").is_file()
    assert (run_path / "final_strategy_validation_errors.json").is_file()
    assert (run_path / "final_synthesis_diagnostics.json").is_file()
    assert result.final_synthesis_diagnostics_path == str(
        run_path / "final_synthesis_diagnostics.json"
    )
    assert result.final_synthesis_status == final_strategy["synthesis_status"]
    assert result.final_synthesis_degraded is final_strategy["fallback_used"]
    assert research_run["eda_evidence_pack_path"] == result.eda_evidence_pack_path
    assert research_run["final_strategy_path"] == result.final_strategy_path
    assert (
        research_run["final_synthesis_diagnostics_path"]
        == result.final_synthesis_diagnostics_path
    )
    assert research_run["final_synthesis_status"] == final_strategy["synthesis_status"]
    assert research_run["final_synthesis_degraded"] is final_strategy["fallback_used"]
    assert result.workflow_status == "completed_with_degradation"
    assert result.degraded_stages == ["final_synthesis"]
    assert result.final_synthesis_stage_status == "degraded_fallback"
    assert research_run["workflow_status"] == "completed_with_degradation"
    assert research_run["degraded_stages"] == ["final_synthesis"]
    assert evidence_pack["validation_evidence"]["primary_validation"]["method"] == "stratified_kfold"
    assert final_strategy["recommended_validation"] == "stratified_kfold"
    validation_action = next(
        action for action in final_strategy["actions"]
        if "validation_evidence.primary_validation" in action["evidence_refs"]
        and action.get("validation_strategy") == "stratified_kfold"
    )
    assert validation_action["evidence_refs"] == ["validation_evidence.primary_validation"]
    assert FakeClient.calls
    assert "source -> hypothesis -> EDA -> strategy" in FakeClient.calls[0]["user_prompt"]


def test_parser_accepts_full_workflow_flags() -> None:
    args = main_module.build_parser().parse_args(
        [
            "https://www.kaggle.com/competitions/iid_binary_tiny",
            "Generic binary classification.",
            "--write-eda-plan",
            "--run-eda",
            "--local-dataset-path",
            "tests/fixtures/eda/iid_binary_tiny",
            "--eda-output-dir",
            "eda-out",
            "--no-download-dataset",
            "--force-download",
            "--enable-p1-modules",
            "--enable-baseline",
            "--research-hypotheses-path",
            "research_hypotheses.json",
            "--eda-task-plan-path",
            "eda_task_plan.json",
            "--eda-evidence-pack-path",
            "eda_evidence_pack.json",
            "--eda-summary-path",
            "eda_summary.md",
            "--final-synthesis",
            "--final-output-dir",
            "final-out",
            "--require-valid-final-synthesis",
        ]
    )

    assert args.write_eda_plan is True
    assert args.execute_eda is True
    assert args.local_dataset_path == Path("tests/fixtures/eda/iid_binary_tiny")
    assert args.eda_output_dir == Path("eda-out")
    assert args.download_dataset is False
    assert args.force_download is True
    assert args.enable_p1_modules is True
    assert args.enable_baseline is True
    assert args.research_hypotheses_path == Path("research_hypotheses.json")
    assert args.eda_task_plan_path == Path("eda_task_plan.json")
    assert args.eda_evidence_pack_path == Path("eda_evidence_pack.json")
    assert args.eda_summary_path == Path("eda_summary.md")
    assert args.final_synthesis is True
    assert args.final_output_dir == Path("final-out")
    assert args.require_valid_final_synthesis is True


def test_main_accepts_existing_eda_pack_for_final_synthesis(monkeypatch, tmp_path: Path) -> None:
    FakeClient.calls = []
    _patch_research_dependencies(monkeypatch, tmp_path)
    eda_pack_path, eda_summary_path = _write_existing_eda_outputs(tmp_path)

    result = run(
        run_research(
            "https://www.kaggle.com/competitions/iid_binary_tiny",
            "Generic iid binary classification with ROC AUC.",
            competition_id="iid_binary_tiny",
            output_dir=tmp_path / "reports",
            report_mode="minimal",
            show_progress=False,
            write_eda_plan=True,
            eda_evidence_pack_path=eda_pack_path,
            eda_summary_path=eda_summary_path,
            final_synthesis=True,
            final_output_dir=tmp_path / "final",
        )
    )

    assert result.eda_evidence_pack_path == str(eda_pack_path)
    assert result.eda_summary_path == str(eda_summary_path)
    assert result.final_strategy_path is not None
    assert result.final_strategy_summary_path is not None
    assert Path(result.final_strategy_path).is_file()
    assert Path(result.final_strategy_summary_path).is_file()
    assert (tmp_path / "final" / "final_strategy.md").is_file()
    assert result.final_synthesis_diagnostics_path == str(
        tmp_path / "final" / "final_synthesis_diagnostics.json"
    )
    assert Path(result.final_synthesis_diagnostics_path).is_file()
    assert json.loads(Path(result.final_strategy_path).read_text(encoding="utf-8"))[
        "recommended_validation"
    ] == "stratified_kfold"
    assert "eda_summary_markdown" in FakeClient.calls[0]["user_prompt"]
    assert result.workflow_status == "completed_with_degradation"
    assert result.workflow_status != "success"
    assert result.degraded_stages == ["final_synthesis"]
    assert result.final_synthesis_stage_status == "degraded_fallback"
    assert any("degraded fallback" in warning for warning in result.warnings)


def test_valid_llm_synthesis_reports_workflow_success(monkeypatch, tmp_path: Path) -> None:
    _patch_research_dependencies(monkeypatch, tmp_path)
    _patch_final_synthesis_status(monkeypatch, "llm_success")
    eda_pack_path, eda_summary_path = _write_existing_eda_outputs(tmp_path)

    result = run(run_research(
        "https://www.kaggle.com/competitions/iid_binary_tiny",
        "Generic iid binary classification with ROC AUC.",
        competition_id="iid_binary_tiny",
        output_dir=tmp_path / "reports",
        report_mode="minimal",
        show_progress=False,
        write_eda_plan=True,
        eda_evidence_pack_path=eda_pack_path,
        eda_summary_path=eda_summary_path,
        final_synthesis=True,
        final_output_dir=tmp_path / "final",
    ))

    assert result.workflow_status == "success"
    assert result.degraded_stages == []
    assert result.final_synthesis_stage_status == "success"


def test_repaired_synthesis_is_success_with_warning(monkeypatch, tmp_path: Path) -> None:
    _patch_research_dependencies(monkeypatch, tmp_path)
    _patch_final_synthesis_status(monkeypatch, "repaired_success")
    eda_pack_path, _ = _write_existing_eda_outputs(tmp_path)

    result = run(run_research(
        "https://www.kaggle.com/competitions/iid_binary_tiny",
        "Generic iid binary classification with ROC AUC.",
        competition_id="iid_binary_tiny",
        output_dir=tmp_path / "reports",
        report_mode="minimal",
        show_progress=False,
        write_eda_plan=True,
        eda_evidence_pack_path=eda_pack_path,
        final_synthesis=True,
        final_output_dir=tmp_path / "final",
    ))

    assert result.workflow_status == "success"
    assert result.degraded_stages == []
    assert result.final_synthesis_stage_status == "repaired_success"
    assert any("deterministic contract repair" in warning for warning in result.warnings)


def test_strict_degraded_synthesis_writes_artifacts_then_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_research_dependencies(monkeypatch, tmp_path)
    _patch_final_synthesis_status(monkeypatch, "degraded_fallback")
    eda_pack_path, _ = _write_existing_eda_outputs(tmp_path)
    final_dir = tmp_path / "final"

    with pytest.raises(FinalSynthesisDegradedError) as caught:
        run(run_research(
            "https://www.kaggle.com/competitions/iid_binary_tiny",
            "Generic iid binary classification with ROC AUC.",
            competition_id="iid_binary_tiny",
            output_dir=tmp_path / "reports",
            report_mode="minimal",
            show_progress=False,
            write_eda_plan=True,
            eda_evidence_pack_path=eda_pack_path,
            final_synthesis=True,
            final_output_dir=final_dir,
            require_valid_final_synthesis=True,
        ))

    error = caught.value
    assert error.result.workflow_status == "failed"
    assert error.result.final_synthesis_stage_status == "failed"
    assert error.result.degraded_stages == ["final_synthesis"]
    assert str(final_dir / "final_synthesis_diagnostics.json") in str(error)
    assert (final_dir / "final_strategy.json").is_file()
    assert (final_dir / "final_strategy.md").is_file()
    assert (final_dir / "final_synthesis_diagnostics.json").is_file()
    run_json = json.loads(
        (Path(error.result.run_artifacts_path) / "research_run.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_json["workflow_status"] == "failed"
    assert run_json["degraded_stages"] == ["final_synthesis"]


def test_run_result_rejects_success_with_degraded_final_synthesis() -> None:
    with pytest.raises(ValueError, match="cannot have workflow_status='success'"):
        ResearchRunResult(
            competition_id="demo",
            num_documents=0,
            duration_sec=0,
            workflow_status="success",
            degraded_stages=["final_synthesis"],
            final_synthesis_status="degraded_fallback",
            final_synthesis_degraded=True,
            final_synthesis_stage_status="degraded_fallback",
        )


def test_final_synthesis_uses_eda_validation(monkeypatch, tmp_path: Path) -> None:
    _patch_research_dependencies(monkeypatch, tmp_path)
    eda_pack_path, _ = _write_existing_eda_outputs(tmp_path)

    result = run(
        run_research(
            "https://www.kaggle.com/competitions/iid_binary_tiny",
            "Generic iid binary classification with ROC AUC.",
            competition_id="iid_binary_tiny",
            output_dir=tmp_path / "reports",
            report_mode="minimal",
            show_progress=False,
            write_eda_plan=True,
            eda_evidence_pack_path=eda_pack_path,
            final_synthesis=True,
        )
    )

    strategy_text = Path(result.final_strategy_summary_path).read_text(encoding="utf-8")
    assert "stratified_kfold" in strategy_text.lower()


def test_final_synthesis_uses_eda_leakage_warnings(monkeypatch, tmp_path: Path) -> None:
    _patch_research_dependencies(monkeypatch, tmp_path)
    eda_pack_path, _ = _write_existing_eda_outputs(tmp_path)

    result = run(
        run_research(
            "https://www.kaggle.com/competitions/iid_binary_tiny",
            "Generic iid binary classification with ROC AUC.",
            competition_id="iid_binary_tiny",
            output_dir=tmp_path / "reports",
            report_mode="minimal",
            show_progress=False,
            write_eda_plan=True,
            eda_evidence_pack_path=eda_pack_path,
            final_synthesis=True,
        )
    )

    strategy_text = Path(result.final_strategy_summary_path).read_text(encoding="utf-8").lower()
    assert "naive target encoding" in strategy_text
    assert "fold-fitted" in strategy_text


def test_research_only_mode_unchanged(monkeypatch, tmp_path: Path) -> None:
    _patch_research_dependencies(monkeypatch, tmp_path)

    result = run(
        run_research(
            "https://www.kaggle.com/competitions/iid_binary_tiny",
            "Generic iid binary classification with ROC AUC.",
            competition_id="iid_binary_tiny",
            output_dir=tmp_path / "reports",
            report_mode="minimal",
            show_progress=False,
        )
    )

    assert result.eda_evidence_pack_path is None
    assert result.eda_summary_path is None
    assert result.final_strategy_path is None
    assert result.final_strategy_summary_path is None
    assert result.workflow_status == "success"
    assert result.degraded_stages == []


def test_eda_only_mode_remains_success(monkeypatch, tmp_path: Path) -> None:
    _patch_research_dependencies(monkeypatch, tmp_path)

    result = run(run_research(
        "https://www.kaggle.com/competitions/iid_binary_tiny",
        "Generic iid binary classification with ROC AUC.",
        competition_id="iid_binary_tiny",
        output_dir=tmp_path / "reports",
        report_mode="minimal",
        show_progress=False,
        write_eda_plan=True,
        execute_eda=True,
        local_dataset_path=Path("tests/fixtures/eda/iid_binary_tiny"),
        eda_output_dir=tmp_path / "eda_runs",
        final_synthesis=False,
    ))

    assert result.eda_evidence_pack_path is not None
    assert result.final_strategy_path is None
    assert result.workflow_status == "success"
    assert result.degraded_stages == []


def test_result_json_includes_final_paths(monkeypatch, tmp_path: Path) -> None:
    _patch_research_dependencies(monkeypatch, tmp_path)
    eda_pack_path, _ = _write_existing_eda_outputs(tmp_path)

    result = run(
        run_research(
            "https://www.kaggle.com/competitions/iid_binary_tiny",
            "Generic iid binary classification with ROC AUC.",
            competition_id="iid_binary_tiny",
            output_dir=tmp_path / "reports",
            report_mode="minimal",
            show_progress=False,
            write_eda_plan=True,
            eda_evidence_pack_path=eda_pack_path,
            final_synthesis=True,
        )
    )

    payload = json.loads((Path(result.run_artifacts_path) / "research_run.json").read_text(encoding="utf-8"))
    assert payload["final_strategy_path"] == result.final_strategy_path
    assert payload["final_strategy_summary_path"] == result.final_strategy_summary_path
    assert payload["final_strategy_path"] is not None


def test_final_synthesis_without_eda_requires_evidence_pack() -> None:
    try:
        run(
            run_research(
                "https://www.kaggle.com/competitions/iid_binary_tiny",
                "Generic iid binary classification with ROC AUC.",
                competition_id="iid_binary_tiny",
                show_progress=False,
                final_synthesis=True,
            )
        )
    except ValueError as exc:
        assert "--final-synthesis requires --run-eda or --eda-evidence-pack-path" in str(exc)
    else:
        raise AssertionError("Expected final synthesis without EDA evidence to fail clearly.")


def _patch_research_dependencies(monkeypatch, tmp_path: Path) -> None:
    FakeClient.calls = []

    async def fake_plan(description: str, client: Any, model: str) -> PlanData:
        return PlanData(
            task_type="binary_classification",
            metric="roc_auc",
            domain="generic_tabular",
            kaggle_queries=["iid binary roc auc"],
            arxiv_queries=[],
            github_queries=[],
        )

    async def fake_collect_sources(**kwargs: Any) -> list[SourceDocument]:
        return [
            SourceDocument(
                id="source-1",
                competition_id=kwargs["competition_id"],
                source="kaggle",
                title="Generic notebook",
                url="https://example.com/notebook",
                content="Uses iid binary classification and ROC AUC.",
            )
        ]

    async def fake_summarize_documents(**kwargs: Any) -> list[SourceDocument]:
        return kwargs["docs"]

    async def fake_retrieve_documents(**kwargs: Any) -> list[RetrievedDocument]:
        return [
            RetrievedDocument(
                id="retrieved-1",
                competition_id=kwargs["competition_id"],
                source="kaggle",
                title="Retrieved generic evidence",
                url="https://example.com/evidence",
                content="Ordinary iid binary classification should use stratified folds.",
                score=0.9,
                rrf_score=0.2,
            )
        ]

    async def fake_run_research_scout(**kwargs: Any) -> ResearchScoutOutput:
        return _scout_output(
            competition_id=kwargs["competition_id"],
            competition_url=kwargs["competition_url"],
        )

    def fake_generate_report(
        competition_name: str,
        roadmap_text: str,
        sources: list[RetrievedDocument],
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("docx placeholder", encoding="utf-8")
        return path

    monkeypatch.setattr(main_module, "load_config", lambda: FakeSettings())
    monkeypatch.setattr(main_module, "DeepSeekClient", FakeClient)
    monkeypatch.setattr(main_module, "PgStore", FakeStore)
    monkeypatch.setattr(main_module, "DomainMemory", FakeDomainMemory)
    monkeypatch.setattr(main_module, "plan", fake_plan)
    monkeypatch.setattr(main_module, "_collect_sources", fake_collect_sources)
    monkeypatch.setattr(main_module, "summarize_documents", fake_summarize_documents)
    monkeypatch.setattr(main_module, "_embed_documents", lambda texts, batch_size, show_progress: [[0.1, 0.2] for _ in texts])
    monkeypatch.setattr(main_module, "_retrieve_documents", fake_retrieve_documents)
    monkeypatch.setattr(main_module, "run_research_scout", fake_run_research_scout)
    monkeypatch.setattr(main_module, "generate_report", fake_generate_report)
    monkeypatch.setattr(main_module, "_create_run_dir", lambda competition_id: tmp_path / "runs" / competition_id)


def _patch_final_synthesis_status(monkeypatch, status: str) -> None:
    async def fake_synthesize_final_strategy(**kwargs: Any) -> FinalStrategyResult:
        payload = _final_strategy_payload()
        state = {
            "llm_success": (True, False, False, False),
            "repaired_success": (False, True, True, False),
            "degraded_fallback": (False, True, False, True),
        }[status]
        payload.update({
            "synthesis_status": status,
            "llm_output_valid": state[0],
            "repair_attempted": state[1],
            "repair_succeeded": state[2],
            "fallback_used": state[3],
            "synthesis_diagnostics_path": str(
                Path(kwargs["diagnostics_dir"]) / "final_synthesis_diagnostics.json"
            ),
        })
        if status == "degraded_fallback":
            payload["limitations"] = [
                "The LLM output was invalid, so a deterministic fallback was used."
            ]
            payload["sections"][0]["summary"] = (
                "Degraded fallback assembled deterministically from validated evidence."
            )
            payload["sections"][-1]["time_blocks"] = [
                {
                    "time_window": window,
                    "summary": "Continue the surviving evidence-backed validation action.",
                    "action_ids": ["action_validation"],
                }
                for window in (
                    "0-4_hours", "4-12_hours", "12-24_hours", "24-48_hours",
                )
            ]
        diagnostics_dir = Path(kwargs["diagnostics_dir"])
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        (diagnostics_dir / "final_synthesis_diagnostics.json").write_text(
            json.dumps({"schema_version": "1.0", "competition_id": "iid_binary_tiny"}),
            encoding="utf-8",
        )
        return FinalStrategyResult.model_validate(payload)

    monkeypatch.setattr(
        main_module,
        "synthesize_final_strategy",
        fake_synthesize_final_strategy,
    )


def _write_existing_eda_outputs(tmp_path: Path) -> tuple[Path, Path]:
    eda_dir = tmp_path / "existing_eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    pack_path = eda_dir / "eda_evidence_pack.json"
    summary_path = eda_dir / "eda_summary.md"
    pack_path.write_text(
        json.dumps(_existing_eda_pack(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(
        "# EDA Summary\n\nPrimary validation: `stratified_kfold`.\n",
        encoding="utf-8",
    )
    return pack_path, summary_path


def _existing_eda_pack() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "competition_id": "iid_binary_tiny",
        "created_at": "2026-07-08T12:00:00+03:00",
        "run_id": "iid_binary_tiny_20260708_120000",
        "file_inventory": {
            "reconciled_table_roles": {
                "train_base": "train.csv",
                "test_base": "test.csv",
                "sample_submission": "sample_submission.csv",
            }
        },
        "inferred_schema": {
            "target_column": "target",
            "primary_id_column": "row_id",
            "prediction_column": "target",
            "train_base_table": "train.csv",
            "test_base_table": "test.csv",
            "sample_submission_table": "sample_submission.csv",
            "global_roles": {
                "target_column": "target",
                "primary_id_column": "row_id",
                "prediction_column": "target",
            },
        },
        "metric_evidence": {
            "metric_name": "roc_auc",
            "normalized_metric_name": "roc_auc",
            "requires_probabilities": True,
        },
        "validation_evidence": {
            "primary_validation": {
                "method": "stratified_kfold",
                "reason": "IID binary target requires class-balanced folds.",
            },
            "diagnostic_validations": [
                {"method": "temporal_holdout", "reason": "No required temporal split."}
            ],
            "evidence_refs": ["validation_evidence.primary_validation"],
        },
        "leakage_evidence": [
            {
                "check_id": "target_in_test",
                "status": "passed",
                "severity": "low",
                "finding": "Target is absent from test base table.",
            }
        ],
        "drift_evidence": {
            "status": "completed",
            "feature_drift_severity": "high",
            "severity": "high",
        },
        "feature_probe_evidence": [
            {
                "feature_family": "target_encoding_or_woe",
                "status": "unsafe",
                "leakage_risk": "high",
                "finding": "Target encoding requires fold-fitted implementation.",
            }
        ],
        "eda_strategy_hints": {
            "validation": [
                {
                    "priority": "P0",
                    "action": "Use StratifiedKFold.",
                    "evidence_refs": ["validation_evidence.primary_validation"],
                }
            ],
            "do_not_do": [
                {
                    "priority": "P0",
                    "action": "Do not use row_id as a predictive feature.",
                    "evidence_refs": ["inferred_schema.primary_id_column"],
                }
            ],
        },
        "hypothesis_results": [
            {
                "hypothesis_id": "val_001",
                "category": "validation",
                "status": "confirmed",
                "confidence_after_eda": "high",
                "finding": "EDA selected StratifiedKFold.",
                "evidence_refs": ["validation_evidence.primary_validation"],
                "impact_on_strategy": "Use StratifiedKFold.",
            }
        ],
        "recommended_next_actions": [
            {
                "priority": "P0",
                "action": "Use StratifiedKFold for model validation.",
                "why": "Validation evidence selected stratified CV.",
                "evidence_refs": ["validation_evidence.primary_validation"],
            }
        ],
    }


def _scout_output(*, competition_id: str, competition_url: str) -> ResearchScoutOutput:
    hypotheses = [
        ScoutHypothesis(
            hypothesis_id="schema_001",
            category="schema",
            claim="Infer target and id columns before modeling.",
            rationale="Schema roles are required for generic EDA.",
            expected_eda_checks=["schema_inferer.roles"],
            priority="P0",
            confidence_before_eda="medium",
            source_refs=["retrieved-1"],
        ),
        ScoutHypothesis(
            hypothesis_id="metric_001",
            category="metric",
            claim="ROC AUC requires continuous positive-class scores.",
            rationale="Metric semantics determine prediction output.",
            expected_eda_checks=["metric_analyzer.resolve_metric"],
            priority="P0",
            confidence_before_eda="high",
            source_refs=["retrieved-1"],
        ),
        ScoutHypothesis(
            hypothesis_id="val_001",
            category="validation",
            claim="Use StratifiedKFold for ordinary iid binary classification.",
            rationale="Class balance should be preserved across folds.",
            expected_eda_checks=["validation_analyzer.select_strategy"],
            priority="P0",
            confidence_before_eda="medium",
            source_refs=["retrieved-1"],
        ),
        ScoutHypothesis(
            hypothesis_id="leak_001",
            category="leakage",
            claim="Check train/test schema and target proxy leakage.",
            rationale="Leakage must be measured on actual files.",
            expected_eda_checks=["leakage_checker.target_proxy_scan"],
            priority="P0",
            confidence_before_eda="medium",
            source_refs=["retrieved-1"],
        ),
        ScoutHypothesis(
            hypothesis_id="drift_001",
            category="drift",
            claim="Measure train/test feature drift before trusting validation.",
            rationale="Distribution shift can affect leaderboard reliability.",
            expected_eda_checks=["drift_analyzer.feature_shift"],
            priority="P1",
            confidence_before_eda="medium",
            source_refs=["retrieved-1"],
        ),
    ]
    tasks = [
        ScoutEdaTask(
            task_id="schema_001",
            module="schema_inferer",
            priority="P0",
            blocking=True,
            related_hypothesis_ids=["schema_001"],
        ),
        ScoutEdaTask(
            task_id="metric_001",
            module="metric_analyzer",
            priority="P0",
            related_hypothesis_ids=["metric_001"],
        ),
        ScoutEdaTask(
            task_id="validation_001",
            module="validation_analyzer",
            priority="P0",
            blocking=True,
            related_hypothesis_ids=["val_001"],
        ),
        ScoutEdaTask(
            task_id="leakage_001",
            module="leakage_checker",
            priority="P0",
            blocking=True,
            related_hypothesis_ids=["leak_001"],
        ),
    ]
    return ResearchScoutOutput(
        competition_id=competition_id,
        competition_url=competition_url,
        task_type="binary_classification",
        metric={"name": "roc_auc", "requires_probabilities": True},
        dataset={},
        hypotheses=hypotheses,
        eda_task_plan=EdaTaskPlanDraft(
            competition_id=competition_id,
            task_type="binary_classification",
            metric={"name": "roc_auc", "requires_probabilities": True},
            eda_tasks=tasks,
            hypothesis_index={
                "schema_001": ["schema_001"],
                "metric_001": ["metric_001"],
                "val_001": ["validation_001"],
                "leak_001": ["leakage_001"],
            },
            recommended_module_sequence=[
                "schema_inferer",
                "metric_analyzer",
                "validation_analyzer",
                "leakage_checker",
            ],
            blocking_tasks=["schema_inferer", "validation_analyzer", "leakage_checker"],
        ),
        models_used={"research_scout": "deepseek-v4-pro"},
    )


def _final_strategy_payload() -> dict[str, Any]:
    action = {
        "action_id": "action_validation",
        "priority": "P0",
        "action": "Use StratifiedKFold for model comparison.",
        "reason": "EDA selected stratified folds for the iid binary target.",
        "evidence_refs": ["validation_evidence.primary_validation"],
        "related_hypothesis_ids": ["val_001"],
        "source_claim": "Retrieved source describes iid binary classification.",
        "source_refs": ["retrieved-1"],
        "eda_result_refs": ["validation_evidence.primary_validation"],
        "validation_strategy": "stratified_kfold",
        "confidence": "high",
    }
    sections = []
    for section_id in REQUIRED_SECTION_IDS:
        section = {
            "section_id": section_id,
            "title": section_id.replace("_", " ").title(),
            "summary": "Strategy guidance tied to EDA evidence.",
            "evidence_refs": ["validation_evidence.primary_validation"],
            "related_hypothesis_ids": ["val_001"],
        }
        if section_id == "metric_and_validation":
            section["actions"] = [action]
        sections.append(section)
    return {
        "competition_id": "iid_binary_tiny",
        "task_type": "binary_classification",
        "metric": {"name": "roc_auc"},
        "recommended_validation": "stratified_kfold",
        "sections": sections,
        "actions": [action],
        "source_to_hypothesis_links": [
            {
                "source_ref": "retrieved-1",
                "source_claim": "Retrieved source describes iid binary classification.",
                "hypothesis_id": "val_001",
            }
        ],
        "hypothesis_to_eda_links": [
            {
                "hypothesis_id": "val_001",
                "eda_result_ref": "validation_evidence.primary_validation",
            }
        ],
    }
