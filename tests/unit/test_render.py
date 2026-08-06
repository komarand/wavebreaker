from __future__ import annotations

import inspect
from datetime import datetime, timezone

from kaggle_researcher import render
from kaggle_researcher.brief_schemas import Claim, CompetitionBrief
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
from kaggle_researcher.research_scout_schemas import EdaTask, ResearchHypothesis


def test_render_facts_section_works_without_a_brief() -> None:
    markdown = render.render_facts_section(_facts())

    assert markdown.startswith("## 1. Соревнование в цифрах")
    assert "| Показатель | Значение |" in markdown
    assert "| Metric | roc_auc |" in markdown
    assert "| Code competition | yes |" in markdown
    assert "| Submissions per day | 5 |" in markdown
    assert "| Maximum team size | 4 |" in markdown
    assert "2026-12-01T12:00:00+00:00" in markdown
    assert "| Train/test size ratio | 2.5 |" in markdown
    assert "id, target" in markdown


def test_facts_section_reports_cluster_counts_and_cv_lb_summary() -> None:
    markdown = render.render_facts_section(_facts())

    assert "| Notebook count | 3 |" in markdown
    assert "| Lineage cluster count | 2 |" in markdown
    assert "GroupKFold: 1; KFold: 1" in markdown
    assert "| CV/LB observations | 3 |" in markdown
    assert "| CV/LB distinct lineage clusters | 2 |" in markdown
    assert "| CV/LB mean gap | 0.02 |" in markdown
    assert "| CV/LB median gap | 0.02 |" in markdown
    assert "| CV/LB Spearman | 1 |" in markdown
    assert "| CV/LB reliability | insufficient |" in markdown
    assert "fewer than 5 pairs; gap is not evidence of a systematic pattern" in markdown


def test_splitter_distribution_counts_lineages_not_forks() -> None:
    markdown = render.render_facts_section(_facts())

    assert "GroupKFold: 1" in markdown
    assert "GroupKFold: 2" not in markdown


def test_facts_section_renders_computed_and_unavailable_shake_up_separately() -> None:
    markdown = render.render_facts_section(_facts())

    assert "past-a: computed; Spearman=0.8" in markdown
    assert "top-10 retention=0.7" in markdown
    assert "match fraction=0.8" in markdown
    assert "limitations=Synthetic stability limitation." in markdown
    assert "limitations=No public snapshot." in markdown
    assert (
        "past-b: not computable (Meta Kaggle dumps not configured.); "
        "matched teams=0, match fraction=unavailable"
    ) in markdown
    assert "No data for similar competition" not in markdown


def test_render_brief_has_sections_one_to_ten_and_sources_in_order() -> None:
    markdown = render.render_brief(_brief(), _facts())
    headings = [
        "## 1. Соревнование в цифрах",
        "## 2. Тезис",
        "## 3. Валидация",
        "## 4. Метрика",
        "## 5. Риски утечки",
        "## 6. Что работает у других",
        "## 7. Чего избегать",
        "## 8. Первые шаги",
        "## 9. Что проверить на данных",
        "## 10. Неизвестное",
        "## Приложение: источники",
    ]

    positions = [markdown.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert markdown.endswith("\n")


def test_every_claim_renders_source_ids_as_a_bracketed_list() -> None:
    markdown = render.render_brief(_brief(), _facts())

    assert (
        "**claim_validation** (fact): Grouped validation appears in notebook evidence. "
        "[facts, author/fork-1]"
    ) in markdown
    assert "**claim_metric** (inference): Verify metric behavior. [facts]" in markdown
    assert "**claim_leakage** (claim): Audit entity overlap. [topic-101]" in markdown
    assert "**claim_works** (claim): A common approach is discussed. [topic-101]" in markdown
    assert "**claim_waste** (inference): Avoid unsupported tuning. [facts]" in markdown


def test_empty_sections_have_an_explicit_no_supported_findings_line() -> None:
    brief = _brief(
        thesis="",
        thesis_support=[],
        validation=[],
        metric_notes=[],
        leakage_risks=[],
        what_works=[],
        time_wasters=[],
        first_moves=[],
        hypotheses=[],
        eda_tasks=[],
        unknowns=[],
        limitations=[],
    )

    markdown = render.render_brief(brief, _minimal_facts())

    assert markdown.count(render.NO_SUPPORTED_FINDINGS) >= 10
    assert "## 3. Валидация\n\nNo supported findings." in markdown
    assert "## 9. Что проверить на данных\n\nNo supported findings." in markdown


def test_hypotheses_and_eda_tasks_render_in_data_checks_section() -> None:
    markdown = render.render_brief(_brief(), _facts())
    section = markdown.split("## 9. Что проверить на данных", 1)[1].split("## 10. Неизвестное", 1)[
        0
    ]

    assert "### Гипотезы" in section
    assert "**val_001** (P0, validation, confidence=medium)" in section
    assert "[facts]" in section
    assert "### EDA-задачи" in section
    assert "**eda_val_001** (P0, validation_analyzer)" in section
    assert "[val_001]" in section


def test_sources_appendix_lists_facts_notebooks_and_discussions() -> None:
    markdown = render.render_brief(_brief(), _facts())
    appendix = markdown.split("## Приложение: источники", 1)[1]

    assert "**facts**: CompetitionFacts for `current-comp`" in appendix
    assert "**author/fork-1**: notebook `Fork 1`" in appendix
    assert "lineage cluster `lineage_group`" in appendix
    assert "**topic-101**: discussion `Host guidance`" in appendix
    assert "Discussion source body" not in appendix


def test_renderer_is_byte_identical_for_identical_input() -> None:
    facts = _facts()
    brief = _brief()

    first = render.render_brief(brief, facts)
    second = render.render_brief(brief, facts)

    assert first.encode("utf-8") == second.encode("utf-8")


def test_table_cells_are_escaped_without_rewriting_claim_prose() -> None:
    facts = _facts()
    facts.metadata.title = "Title | with separator"
    brief = _brief()
    brief.validation[0].text = "Keep | claim punctuation?!"

    markdown = render.render_brief(brief, facts)

    assert "Title \\| with separator" in markdown
    assert "Keep | claim punctuation?!" in markdown


def test_renderer_has_no_llm_or_regular_expression_dependency() -> None:
    source = inspect.getsource(render)

    for forbidden in ("import re", "DeepSeekClient", "chat_json", "chat_text"):
        assert forbidden not in source


def _brief(**overrides) -> CompetitionBrief:
    payload = {
        "competition_id": "current-comp",
        "thesis": "Validation design should drive the initial strategy.",
        "thesis_support": ["claim_validation", "claim_leakage"],
        "validation": [
            _claim(
                "claim_validation",
                "Grouped validation appears in notebook evidence.",
                ["facts", "author/fork-1"],
                "fact",
            )
        ],
        "metric_notes": [_claim("claim_metric", "Verify metric behavior.", ["facts"], "inference")],
        "leakage_risks": [_claim("claim_leakage", "Audit entity overlap.", ["topic-101"], "claim")],
        "what_works": [
            _claim(
                "claim_works",
                "A common approach is discussed.",
                ["topic-101"],
                "claim",
            )
        ],
        "time_wasters": [
            _claim("claim_waste", "Avoid unsupported tuning.", ["facts"], "inference")
        ],
        "hypotheses": [_hypothesis()],
        "eda_tasks": [_eda_task()],
        "first_moves": ["Inspect schema roles.", "Build a validation baseline."],
        "unknowns": ["The temporal boundary is unknown."],
        "limitations": ["Sample submission metadata is incomplete."],
    }
    payload.update(overrides)
    return CompetitionBrief.model_validate(payload)


def _claim(
    claim_id: str,
    text: str,
    source_ids: list[str],
    kind: str,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        text=text,
        source_ids=source_ids,
        kind=kind,
    )


def _facts() -> CompetitionFacts:
    notebooks = [
        _notebook("author/fork-1", "Fork 1", "lineage_group", "GroupKFold"),
        _notebook("author/fork-2", "Fork 2", "lineage_group", "GroupKFold"),
        _notebook("author/other", "Other", "lineage_other", "KFold"),
    ]
    return CompetitionFacts(
        competition_id="current-comp",
        collected_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        metadata=CompetitionMetadata(
            competition_id="current-comp",
            title="Current Competition",
            metric_name="roc_auc",
            is_code_competition=True,
            submissions_per_day=5,
            max_team_size=4,
            deadline=datetime(2026, 12, 1, 12, tzinfo=timezone.utc),
            unavailable_fields=[],
        ),
        files=FileManifest(
            files=[FileInfo(name="train.csv", size_bytes=100, role_hint="train")],
            train_test_size_ratio=2.5,
            sample_submission_columns=["id", "target"],
            sample_submission_source="api",
            limitations=[],
        ),
        notebooks=notebooks,
        discussions=[
            DiscussionFacts(
                topic_id="topic-101",
                title="Host guidance",
                author="host",
                author_is_host=True,
                votes=10,
                source_type="discussion",
                competition_id="current-comp",
                text="Discussion source body.",
            )
        ],
        similar_competitions=[
            LeaderboardStability(
                competition_id="past-b",
                status="not_computable",
                matched_teams=0,
                source="unavailable",
                not_computable_reason="Meta Kaggle dumps not configured.",
                limitations=["No public snapshot."],
            ),
            LeaderboardStability(
                competition_id="past-a",
                status="computed",
                public_private_spearman=0.8,
                top10_retention=0.7,
                median_rank_change=4.0,
                matched_teams=100,
                match_fraction=0.8,
                source="meta_kaggle",
                limitations=["Synthetic stability limitation."],
            ),
        ],
        cv_lb_pairs=[
            CvLbPair(
                notebook_ref="author/fork-1",
                declared_cv=0.82,
                public_score=0.8,
                lineage_cluster_id="lineage_group",
            ),
            CvLbPair(
                notebook_ref="author/fork-2",
                declared_cv=0.83,
                public_score=0.81,
                lineage_cluster_id="lineage_group",
            ),
            CvLbPair(
                notebook_ref="author/other",
                declared_cv=0.84,
                public_score=0.82,
                lineage_cluster_id="lineage_other",
            ),
        ],
        user_constraints=UserConstraints(vram_gb=12),
        collection_errors=[],
    )


def _minimal_facts() -> CompetitionFacts:
    return CompetitionFacts(
        competition_id="current-comp",
        collected_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        metadata=CompetitionMetadata(
            competition_id="current-comp",
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
        user_constraints=UserConstraints(),
        collection_errors=[],
    )


def _notebook(
    ref: str,
    title: str,
    cluster_id: str,
    splitter: str,
) -> NotebookFacts:
    return NotebookFacts(
        ref=ref,
        title=title,
        author="author",
        ast_fingerprint="a" * 64,
        lineage_cluster_id=cluster_id,
        splitters=[CodeObservation(name=splitter, kwargs={}, locator="cell_1")],
        models=[],
        metrics=[],
        feature_ops=[],
        declared_cv=[],
        parse_status="ok",
    )


def _hypothesis() -> ResearchHypothesis:
    return ResearchHypothesis(
        id="val_001",
        category="validation",
        priority="P0",
        claim="Grouped validation may be required.",
        why_it_matters="Entity overlap may inflate validation.",
        how_to_verify=["Measure entity overlap between folds."],
        provenance=["heuristic", "not_verified_on_data"],
        supporting_source_ids=["facts"],
        confidence="medium",
    )


def _eda_task() -> EdaTask:
    return EdaTask(
        id="eda_val_001",
        priority="P0",
        module="validation_analyzer",
        question="Which column defines independent validation groups?",
        rationale="Independent groups are needed for validation.",
        required_inputs=["train"],
        expected_outputs=["validation_evidence.group_column"],
        related_hypothesis_ids=["val_001"],
        blocking=True,
    )
