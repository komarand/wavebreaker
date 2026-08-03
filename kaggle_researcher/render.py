from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from kaggle_researcher.brief_schemas import Claim, CompetitionBrief
from kaggle_researcher.facts.cv_lb import summarize_cv_lb
from kaggle_researcher.facts.models import CompetitionFacts, LeaderboardStability
from kaggle_researcher.research_scout_schemas import EdaTask, ResearchHypothesis

NO_SUPPORTED_FINDINGS = "No supported findings."


def render_facts_section(facts: CompetitionFacts) -> str:
    metadata = facts.metadata
    cv_lb = summarize_cv_lb(facts.cv_lb_pairs)
    rows = [
        ("Competition ID", facts.competition_id),
        ("Title", metadata.title),
        ("Metric", metadata.metric_name),
        ("Code competition", _format_boolean(metadata.is_code_competition)),
        ("Submissions per day", metadata.submissions_per_day),
        ("Maximum team size", metadata.max_team_size),
        ("Deadline", _format_datetime(metadata.deadline)),
        ("Train/test size ratio", facts.files.train_test_size_ratio),
        (
            "Sample submission columns",
            _format_sequence(facts.files.sample_submission_columns),
        ),
        ("Sample submission source", facts.files.sample_submission_source),
        ("Notebook count", len(facts.notebooks)),
        ("Lineage cluster count", _lineage_cluster_count(facts)),
        ("Splitter distribution by cluster", _splitter_distribution(facts)),
        ("CV/LB observations", cv_lb["count"]),
        ("CV/LB distinct lineage clusters", cv_lb["distinct_lineage_clusters"]),
        ("CV/LB mean gap", cv_lb["mean_gap"]),
        ("CV/LB median gap", cv_lb["median_gap"]),
        ("CV/LB Spearman", cv_lb["spearman"]),
        ("Similar-competition shake-up", _shake_up_summary(facts)),
    ]
    lines = [
        "## 1. Соревнование в цифрах",
        "",
        "| Показатель | Значение |",
        "|---|---|",
    ]
    lines.extend(
        f"| {_escape_table_cell(label)} | {_escape_table_cell(_format_value(value))} |"
        for label, value in rows
    )
    return "\n".join(lines)


def render_brief(
    brief: CompetitionBrief,
    facts: CompetitionFacts,
) -> str:
    title = facts.metadata.title or facts.competition_id
    sections = [
        f"# Competition Brief: {title}",
        "",
        render_facts_section(facts),
        "",
        _render_thesis(brief),
        "",
        _render_claim_section("## 3. Валидация", brief.validation),
        "",
        _render_claim_section("## 4. Метрика", brief.metric_notes),
        "",
        _render_claim_section("## 5. Риски утечки", brief.leakage_risks),
        "",
        _render_claim_section("## 6. Что работает у других", brief.what_works),
        "",
        _render_claim_section("## 7. Чего избегать", brief.time_wasters),
        "",
        _render_string_section("## 8. Первые шаги", brief.first_moves),
        "",
        _render_data_checks(brief.hypotheses, brief.eda_tasks),
        "",
        _render_unknowns(brief),
        "",
        _render_sources_appendix(facts),
    ]
    return "\n".join(sections) + "\n"


def _render_thesis(brief: CompetitionBrief) -> str:
    lines = ["## 2. Тезис", ""]
    if not brief.thesis:
        lines.append(NO_SUPPORTED_FINDINGS)
        return "\n".join(lines)
    lines.append(brief.thesis)
    lines.extend(
        [
            "",
            f"Поддерживающие claims: {_bracketed_ids(brief.thesis_support)}",
        ]
    )
    return "\n".join(lines)


def _render_claim_section(heading: str, claims: list[Claim]) -> str:
    lines = [heading, ""]
    if not claims:
        lines.append(NO_SUPPORTED_FINDINGS)
        return "\n".join(lines)
    for claim in claims:
        lines.append(
            f"- **{claim.claim_id}** ({claim.kind}): {claim.text} "
            f"{_bracketed_ids(claim.source_ids)}"
        )
    return "\n".join(lines)


def _render_string_section(heading: str, items: list[str]) -> str:
    lines = [heading, ""]
    if not items:
        lines.append(NO_SUPPORTED_FINDINGS)
        return "\n".join(lines)
    lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)


def _render_data_checks(
    hypotheses: list[ResearchHypothesis],
    eda_tasks: list[EdaTask],
) -> str:
    lines = ["## 9. Что проверить на данных", ""]
    if not hypotheses and not eda_tasks:
        lines.append(NO_SUPPORTED_FINDINGS)
        return "\n".join(lines)

    lines.extend(["### Гипотезы", ""])
    if hypotheses:
        for hypothesis in hypotheses:
            lines.extend(_render_hypothesis(hypothesis))
    else:
        lines.append(NO_SUPPORTED_FINDINGS)

    lines.extend(["", "### EDA-задачи", ""])
    if eda_tasks:
        for task in eda_tasks:
            lines.extend(_render_eda_task(task))
    else:
        lines.append(NO_SUPPORTED_FINDINGS)
    return "\n".join(lines)


def _render_hypothesis(hypothesis: ResearchHypothesis) -> list[str]:
    lines = [
        f"- **{hypothesis.id}** ({hypothesis.priority}, {hypothesis.category}, "
        f"confidence={hypothesis.confidence}): {hypothesis.claim} "
        f"{_bracketed_ids(hypothesis.supporting_source_ids)}",
        f"  - Почему важно: {hypothesis.why_it_matters}",
    ]
    if hypothesis.how_to_verify:
        lines.append(f"  - Как проверить: {_format_sequence(hypothesis.how_to_verify)}")
    return lines


def _render_eda_task(task: EdaTask) -> list[str]:
    lines = [
        f"- **{task.id}** ({task.priority}, {task.module}): {task.question}",
        f"  - Обоснование: {task.rationale}",
        f"  - Ожидаемые результаты: {_bracketed_ids(task.expected_outputs)}",
        f"  - Связанные гипотезы: {_bracketed_ids(task.related_hypothesis_ids)}",
        f"  - Блокирующая: {_format_boolean(task.blocking)}",
    ]
    if task.required_inputs:
        lines.insert(2, f"  - Входы: {_bracketed_ids(task.required_inputs)}")
    return lines


def _render_unknowns(brief: CompetitionBrief) -> str:
    lines = ["## 10. Неизвестное", ""]
    if brief.unknowns:
        lines.extend(f"- {item}" for item in brief.unknowns)
    else:
        lines.append(NO_SUPPORTED_FINDINGS)

    lines.extend(["", "### Ограничения", ""])
    if brief.limitations:
        lines.extend(f"- {item}" for item in brief.limitations)
    else:
        lines.append(NO_SUPPORTED_FINDINGS)
    return "\n".join(lines)


def _render_sources_appendix(facts: CompetitionFacts) -> str:
    lines = ["## Приложение: источники", ""]
    lines.append(f"- **facts**: CompetitionFacts for `{facts.competition_id}`.")
    for notebook in sorted(facts.notebooks, key=lambda item: item.ref):
        author = notebook.author or "unknown author"
        lines.append(
            f"- **{notebook.ref}**: notebook `{notebook.title}` by {author}; "
            f"lineage cluster `{notebook.lineage_cluster_id}`."
        )
    for discussion in sorted(facts.discussions, key=lambda item: item.topic_id):
        author = discussion.author or "unknown author"
        lines.append(
            f"- **{discussion.topic_id}**: {discussion.source_type} "
            f"`{discussion.title}` by {author}; competition "
            f"`{discussion.competition_id}`."
        )
    return "\n".join(lines)


def _lineage_cluster_count(facts: CompetitionFacts) -> int:
    return len({notebook.lineage_cluster_id for notebook in facts.notebooks})


def _splitter_distribution(facts: CompetitionFacts) -> str:
    splitters_by_cluster: dict[str, set[str]] = {}
    for notebook in facts.notebooks:
        names = splitters_by_cluster.setdefault(notebook.lineage_cluster_id, set())
        names.update(observation.name for observation in notebook.splitters)
    counts = Counter(
        name
        for cluster_names in splitters_by_cluster.values()
        for name in cluster_names
    )
    if not counts:
        return "unavailable"
    return "; ".join(f"{name}: {counts[name]}" for name in sorted(counts))


def _shake_up_summary(facts: CompetitionFacts) -> str:
    if not facts.similar_competitions:
        return "unavailable"
    return "; ".join(
        _format_stability(item)
        for item in sorted(
            facts.similar_competitions,
            key=lambda item: item.competition_id,
        )
    )


def _format_stability(item: LeaderboardStability) -> str:
    diagnostics = (
        f"matched teams={item.matched_teams}, match fraction="
        f"{_format_value(item.match_fraction)}"
    )
    if item.status == "not_computable":
        reason = item.not_computable_reason or "reason unavailable"
        rendered = f"{item.competition_id}: not computable ({reason}); {diagnostics}"
    else:
        rendered = (
            f"{item.competition_id}: computed; Spearman="
            f"{_format_value(item.public_private_spearman)}, top-10 retention="
            f"{_format_value(item.top10_retention)}, median rank change="
            f"{_format_value(item.median_rank_change)}, {diagnostics}"
        )
    if item.limitations:
        rendered += f"; limitations={' | '.join(item.limitations)}"
    return rendered


def _format_boolean(value: bool | None) -> str:
    if value is None:
        return "unavailable"
    return "yes" if value else "no"


def _format_datetime(value: datetime | None) -> str:
    return value.isoformat() if value is not None else "unavailable"


def _format_sequence(values: Iterable[Any]) -> str:
    rendered = [str(value) for value in values]
    return ", ".join(rendered) if rendered else "unavailable"


def _bracketed_ids(values: Iterable[str]) -> str:
    return f"[{', '.join(values)}]"


def _format_value(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return format(value, ".8g")
    return str(value)


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")
