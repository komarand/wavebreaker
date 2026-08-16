from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from kaggle_researcher import journal
from kaggle_researcher.brief import generate_brief
from kaggle_researcher.brief_schemas import CompetitionBrief
from kaggle_researcher.brief_validate import validate_brief
from kaggle_researcher.config import (
    DEFAULT_MAX_SAMPLE_SUB_BYTES,
    get_writeups_per_competition,
    load_config,
)
from kaggle_researcher.facts.collect import collect_facts
from kaggle_researcher.facts.cv_lb import summarize_cv_lb
from kaggle_researcher.facts.models import CompetitionFacts, UserConstraints
from kaggle_researcher.render import render_brief, render_facts_section
from kaggle_researcher.report.docx_generator import generate_report

OBJECTIVES = ("medal", "top_percent", "learn", "fast_baseline")
DEFAULT_MAX_NOTEBOOKS = 20
DEFAULT_MAX_DISCUSSIONS = 200


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wave",
        description="Wavebreaker B5 competition research pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    facts_parser = subparsers.add_parser(
        "facts",
        help="Collect deterministic competition facts.",
    )
    _add_competition_arguments(facts_parser)

    brief_parser = subparsers.add_parser(
        "brief",
        help="Generate a grounded competition brief.",
    )
    _add_competition_arguments(brief_parser)
    brief_parser.add_argument("--vram", type=float, help="Available VRAM in GB.")
    brief_parser.add_argument("--hours", type=float, help="Available hours per week.")
    brief_parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        default="medal",
        help="Primary competition objective.",
    )
    brief_parser.add_argument(
        "--facts-from",
        help="Read previously collected facts from this path.",
    )
    brief_parser.add_argument(
        "--docx",
        action="store_true",
        help="Also write brief.docx using the rendered markdown.",
    )

    journal_parser = subparsers.add_parser(
        "journal",
        help="Record the outcome of a competition.",
    )
    journal_parser.add_argument("slug", help="Kaggle competition slug.")
    journal_parser.add_argument("--used-validation", help="Validation scheme used.")
    journal_parser.add_argument("--final-rank", type=int, help="Final leaderboard rank.")
    journal_parser.add_argument("--num-teams", type=int, help="Number of competing teams.")
    journal_parser.add_argument(
        "--brief-run-id",
        help="Run directory name under runs/ containing brief.json.",
    )
    journal_parser.add_argument(
        "--brief-was-useful",
        choices=("yes", "no"),
        help="Whether the generated brief was useful.",
    )
    journal_parser.add_argument("--notes", help="Optional participation notes.")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "facts":
        _run_facts(args)
        return
    if args.command == "brief":
        _run_brief(args)
        return
    if args.command == "journal":
        _run_journal(args)
        return
    raise AssertionError(f"unhandled wave command: {args.command}")


def _run_facts(args: argparse.Namespace) -> None:
    facts = collect_facts(
        slug=args.slug,
        max_notebooks=(
            args.max_notebooks if args.max_notebooks is not None else DEFAULT_MAX_NOTEBOOKS
        ),
        max_discussions=(
            args.max_discussions if args.max_discussions is not None else DEFAULT_MAX_DISCUSSIONS
        ),
        writeups_per_competition=args.writeups_per_competition,
        similar=_parse_similar(args.similar),
        user_constraints=UserConstraints(),
        max_sample_sub_bytes=DEFAULT_MAX_SAMPLE_SUB_BYTES,
    )
    output_dir = _create_output_dir(facts.competition_id, args.out)
    facts_path = output_dir / "facts.json"
    facts_path.write_text(
        json.dumps(facts.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _print_facts_summary(facts, facts_path)


def _run_brief(args: argparse.Namespace) -> None:
    facts = _facts_for_brief(args)
    output_dir = _create_output_dir(facts.competition_id, args.out)
    facts_path = output_dir / "facts.json"
    _write_model_json(facts_path, facts)

    try:
        settings = load_config()
        generated_brief = asyncio.run(generate_brief(facts, settings))
    except Exception as exc:
        markdown = _facts_only_brief(facts, exc)
        markdown_path = output_dir / "brief.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        docx_path = _write_docx(facts, markdown, output_dir) if args.docx else None
        _print_brief_paths(
            facts_path=facts_path,
            brief_path=None,
            markdown_path=markdown_path,
            docx_path=docx_path,
        )
        return

    brief = validate_brief(generated_brief, facts)
    brief_path = output_dir / "brief.json"
    markdown_path = output_dir / "brief.md"
    markdown = render_brief(brief, facts)
    _write_model_json(brief_path, brief)
    markdown_path.write_text(markdown, encoding="utf-8")
    docx_path = _write_docx(facts, markdown, output_dir) if args.docx else None
    _print_brief_paths(
        facts_path=facts_path,
        brief_path=brief_path,
        markdown_path=markdown_path,
        docx_path=docx_path,
    )
    _print_claim_stats(brief)


def _facts_for_brief(args: argparse.Namespace) -> CompetitionFacts:
    if args.facts_from:
        return CompetitionFacts.model_validate_json(
            Path(args.facts_from).read_text(encoding="utf-8")
        )
    return collect_facts(
        slug=args.slug,
        max_notebooks=(
            args.max_notebooks if args.max_notebooks is not None else DEFAULT_MAX_NOTEBOOKS
        ),
        max_discussions=(
            args.max_discussions if args.max_discussions is not None else DEFAULT_MAX_DISCUSSIONS
        ),
        writeups_per_competition=args.writeups_per_competition,
        similar=_parse_similar(args.similar),
        user_constraints=UserConstraints(
            vram_gb=args.vram,
            hours_per_week=args.hours,
            objective=args.objective,
        ),
        max_sample_sub_bytes=DEFAULT_MAX_SAMPLE_SUB_BYTES,
    )


def _write_model_json(path: Path, model: BaseModel) -> None:
    path.write_text(
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _facts_only_brief(facts: CompetitionFacts, error: Exception) -> str:
    return (
        f"{render_facts_section(facts)}\n\n"
        "> Full competition brief generation was unavailable "
        f"({type(error).__name__}). The facts checkpoint remains usable.\n"
    )


def _write_docx(
    facts: CompetitionFacts,
    markdown: str,
    output_dir: Path,
) -> Path:
    return generate_report(
        competition_name=facts.metadata.title or facts.competition_id,
        roadmap_text=markdown,
        sources=[],
        output_path=output_dir / "brief.docx",
        overwrite=True,
    )


def _print_brief_paths(
    *,
    facts_path: Path,
    brief_path: Path | None,
    markdown_path: Path,
    docx_path: Path | None,
) -> None:
    print(f"facts: {facts_path}")
    if brief_path is not None:
        print(f"brief json: {brief_path}")
    print(f"brief markdown: {markdown_path}")
    if docx_path is not None:
        print(f"brief docx: {docx_path}")


def _print_claim_stats(brief: CompetitionBrief) -> None:
    stats = brief.claim_stats
    if stats is None:
        return
    print(
        f"claims: {stats.total} (fact {stats.fact}, claim {stats.claim}, "
        f"inference {stats.inference})"
    )
    print(
        f"grounding rate: {stats.grounding_rate} across "
        f"{stats.distinct_sources} sources"
    )


def _run_journal(args: argparse.Namespace) -> None:
    useful = args.brief_was_useful == "yes" if args.brief_was_useful is not None else None
    journal.append_entry(
        competition_id=args.slug,
        brief_run_id=args.brief_run_id,
        used_validation=args.used_validation,
        final_rank=args.final_rank,
        num_teams=args.num_teams,
        brief_was_useful=useful,
        notes=args.notes,
    )
    print(json.dumps(journal.summarize(), ensure_ascii=False, sort_keys=True))


def _create_output_dir(competition_id: str, requested: str | None) -> Path:
    if requested:
        output_dir = Path(requested)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("runs") / f"{competition_id}_{timestamp}"
    counter = 2
    while output_dir.exists():
        output_dir = Path("runs") / f"{competition_id}_{timestamp}_{counter:03d}"
        counter += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _print_facts_summary(facts: CompetitionFacts, facts_path: Path) -> None:
    metadata = facts.metadata
    code_competition = (
        "yes"
        if metadata.is_code_competition is True
        else "no"
        if metadata.is_code_competition is False
        else "unavailable"
    )
    ratio = facts.files.train_test_size_ratio
    clusters = {notebook.lineage_cluster_id for notebook in facts.notebooks}
    print(f"facts: {facts_path}")
    print(f"metric: {metadata.metric_name or 'unavailable'}")
    print(f"code competition: {code_competition}")
    print(f"train/test ratio: {ratio if ratio is not None else 'unavailable'}")
    sample_columns = ", ".join(facts.files.sample_submission_columns) or "unavailable"
    print(f"sample submission columns: {sample_columns}")
    print(f"sample submission status: {facts.files.sample_submission_status}")
    print(f"notebooks: {len(facts.notebooks)}")
    print(
        f"public leaderboard: {facts.public_leaderboard.status} "
        f"({facts.public_leaderboard.entry_count} entries)"
    )
    leaderboard_shape = facts.public_leaderboard.shape
    if leaderboard_shape is not None:
        shape_parts = [
            f"{leaderboard_shape.entry_count} entries",
            "top "
            + (
                f"{leaderboard_shape.top_score:.5f}"
                if leaderboard_shape.top_score is not None
                else "unavailable"
            ),
        ]
        if 50 in leaderboard_shape.score_at_rank:
            shape_parts.append(f"rank50 {leaderboard_shape.score_at_rank[50]:.5f}")
        if leaderboard_shape.median_adjacent_delta is not None:
            shape_parts.append(
                f"nonzero delta {leaderboard_shape.median_adjacent_delta:.5f}"
            )
        if leaderboard_shape.tied_ratio is not None:
            shape_parts.append(f"tied ratio {leaderboard_shape.tied_ratio:.4g}")
        if leaderboard_shape.plateau_ratio is not None:
            shape_parts.append(f"plateau ratio {leaderboard_shape.plateau_ratio:.4g}")
        print(f"leaderboard shape: {', '.join(shape_parts)}")
    match_confidences = Counter(
        match.match_confidence for match in facts.leaderboard_matches
    )
    print(
        "leaderboard matches: "
        f"{match_confidences['exact']} exact, {match_confidences['partial']} partial"
    )
    print(f"lineage clusters: {len(clusters)}")
    if facts.code_aggregates is not None:
        model_counts = ", ".join(
            f"{item.name} {item.cluster_count}"
            for item in facts.code_aggregates.models
        )
        print(f"models by lineage cluster: {model_counts or 'none'}")
        combinations = ", ".join(
            f"{'+'.join(item.names)} {item.cluster_count} clusters"
            for item in facts.code_aggregates.model_combinations
        )
        print(f"model combinations: {combinations or 'none'}")
    if facts.dataset_references:
        top_references = ", ".join(
            f"{item.slug} {item.cluster_count} clusters"
            for item in facts.dataset_references[:2]
        )
        print(
            f"dataset references: {len(facts.dataset_references)} "
            f"(top: {top_references})"
        )
    else:
        print("dataset references: 0")
    if facts.similar_diagnostics is not None:
        similar_diagnostics = facts.similar_diagnostics
        print(
            f"similar candidates: {similar_diagnostics.candidates_seen} "
            f"({similar_diagnostics.verified} verified, "
            f"{similar_diagnostics.rejected} rejected, "
            f"{similar_diagnostics.not_found} not found)"
        )
        for candidate in facts.similar_candidates:
            detail = (
                candidate.rejection_reason
                if candidate.rejection_reason is not None
                else f"{candidate.mention_topic_count} topics"
            )
            print(
                f"  {candidate.slug:<22} {candidate.verification:<10} "
                f"{candidate.discovered_by:<19} {detail}"
            )
    print(f"splitters by lineage cluster: {dict(_splitter_distribution(facts))}")
    print(f"cv/lb: {summarize_cv_lb(facts.cv_lb_pairs)}")
    diagnostics = facts.cv_lb_diagnostics
    print(f"public notebook scores: {diagnostics.notebooks_with_public_score}")
    print(
        "notebooks with declared CV text: "
        f"{diagnostics.notebooks_with_declared_cv}"
    )
    print(f"notebooks with both: {diagnostics.notebooks_with_both}")
    print(f"comparable cv/lb pairs: {diagnostics.comparable_pairs}")
    print(
        "notebooks with CV-side score observations: "
        f"{diagnostics.notebooks_with_cv_scores}"
    )
    print(f"notebooks with LB: {diagnostics.notebooks_with_lb_scores}")
    print(f"notebooks with both sides: {diagnostics.notebooks_with_both_sides}")
    print(f"cv/lb pairs: {diagnostics.pairs_created}")
    print(f"  API LB: {diagnostics.pairs_created_from_api_lb}")
    print(f"  observation LB: {diagnostics.pairs_created_from_observation_lb}")
    print(
        "  leaderboard match LB: "
        f"{diagnostics.pairs_created_from_leaderboard_match}"
    )
    if diagnostics.zero_pairs_reason:
        print(f"cv/lb diagnostic: {diagnostics.zero_pairs_reason}")
    score_diagnostics = facts.score_diagnostics
    print(f"score observations: {score_diagnostics.observations_total}")
    print(f"  cv: {score_diagnostics.split_cv}")
    print(f"  lb: {score_diagnostics.split_lb}")
    print(f"  unknown: {score_diagnostics.split_unknown}")
    print(
        "notebooks with score observations: "
        f"{score_diagnostics.notebooks_with_score_observations}"
    )
    print(
        "canonical score metrics: "
        f"{score_diagnostics.observations_with_canonical_metric} "
        f"(alias {score_diagnostics.canonical_by_alias}, competition hint "
        f"{score_diagnostics.canonical_by_competition_hint})"
    )
    print(
        "uncanonicalized score metrics: "
        f"{score_diagnostics.observations_without_canonical_metric}"
    )
    print("title/ref score observations: " f"{score_diagnostics.title_or_ref_observations}")
    print(f"excluded score candidates: {score_diagnostics.candidates_excluded}")
    if score_diagnostics.implausible_observations:
        implausible_details = ", ".join(
            f"{reason} {count}"
            for reason, count in sorted(score_diagnostics.implausible_observations.items())
        )
        print(
            "implausible observations: "
            f"{sum(score_diagnostics.implausible_observations.values())} "
            f"({implausible_details})"
        )
    if score_diagnostics.implausible_top_labels:
        top_labels = ", ".join(
            f"{label} {count}"
            for label, count in list(
                score_diagnostics.implausible_top_labels.items()
            )[:8]
        )
        print(f"top excluded labels: {top_labels}")
    if score_diagnostics.notebooks_failed_by_status:
        print(
            "notebook pull failures by HTTP status: "
            f"{score_diagnostics.notebooks_failed_by_status}"
        )
    if score_diagnostics.notebooks_failed_by_exception:
        print(
            "notebook pull failures by exception: "
            f"{score_diagnostics.notebooks_failed_by_exception}"
        )
    rejection_total = sum(
        (
            diagnostics.pairs_rejected_missing_cv,
            diagnostics.pairs_rejected_missing_lb,
            diagnostics.pairs_rejected_metric_mismatch,
            diagnostics.pairs_rejected_ambiguous_metric,
            diagnostics.pairs_rejected_scale_mismatch,
            diagnostics.rejected_implausible_gap,
            diagnostics.pairs_rejected_ambiguous_split,
        )
    )
    if not facts.cv_lb_pairs or rejection_total:
        print(
            "cv/lb rejections: "
            f"missing_cv={diagnostics.pairs_rejected_missing_cv}, "
            f"missing_lb={diagnostics.pairs_rejected_missing_lb}, "
            f"metric_mismatch={diagnostics.pairs_rejected_metric_mismatch}, "
            f"ambiguous_metric={diagnostics.pairs_rejected_ambiguous_metric}, "
            f"scale_mismatch={diagnostics.pairs_rejected_scale_mismatch}, "
            f"implausible_gap={diagnostics.rejected_implausible_gap}, "
            f"ambiguous_split={diagnostics.pairs_rejected_ambiguous_split}"
        )
    competition_discussions = [
        discussion
        for discussion in facts.discussions
        if discussion.source_type == "discussion"
        and discussion.competition_id == facts.competition_id
    ]
    message_count = sum(len(discussion.messages) for discussion in competition_discussions)
    writeup_candidates = sum(
        discussion.is_writeup_candidate for discussion in competition_discussions
    )
    external_links = sum(
        link.kind == "external"
        for discussion in competition_discussions
        for message in discussion.messages
        for link in message.links
    )
    failed_topics = sum(
        discussion.collection_status not in {"collected", "empty"}
        for discussion in competition_discussions
    )
    print(f"discussions: {len(facts.discussions)}")
    print(f"discussion status: {facts.discussion_collection_status}")
    print(f"topics: {len(competition_discussions)}")
    print(f"messages: {message_count}")
    print(f"writeup candidates: {writeup_candidates}")
    print(f"external links: {external_links}")
    print(f"failed topics: {failed_topics}")
    print(f"discussion auth: {facts.discussion_auth_mode}")
    if facts.discussion_collection_error:
        print(f"discussion error: {facts.discussion_collection_error}")
    for limitation in facts.limitations:
        print(f"limitation: {limitation}")


def _splitter_distribution(facts: CompetitionFacts) -> Counter[str]:
    splitters_by_cluster: dict[str, set[str]] = {}
    for notebook in facts.notebooks:
        cluster_splitters = splitters_by_cluster.setdefault(
            notebook.lineage_cluster_id,
            set(),
        )
        cluster_splitters.update(splitter.name for splitter in notebook.splitters)
    return Counter(
        splitter
        for cluster_splitters in splitters_by_cluster.values()
        for splitter in cluster_splitters
    )


def _parse_similar(value: str | None) -> list[str]:
    if not value:
        return []
    return list(dict.fromkeys(slug.strip() for slug in value.split(",") if slug.strip()))


def _add_competition_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("slug", help="Kaggle competition slug.")
    parser.add_argument("--max-notebooks", type=int, help="Maximum notebooks to collect.")
    parser.add_argument("--max-discussions", type=int, help="Maximum discussions to collect.")
    parser.add_argument(
        "--writeups-per-competition",
        type=_positive_int,
        default=get_writeups_per_competition(),
        help="Maximum winner writeups per similar competition.",
    )
    parser.add_argument("--similar", help="Comma-separated similar competition slugs.")
    parser.add_argument("--out", help="Output directory.")


def _positive_int(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


if __name__ == "__main__":
    main()
