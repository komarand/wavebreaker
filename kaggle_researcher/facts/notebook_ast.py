from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import re
import tokenize
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from kaggle_researcher.facts.models import (
    CodeObservation,
    DeclaredCvObservation,
    NotebookFacts,
    ScoreDiagnostics,
    ScoreObservation,
    ScoreSplit,
)

SPLITTER_NAMES = {
    "KFold",
    "StratifiedKFold",
    "GroupKFold",
    "StratifiedGroupKFold",
    "TimeSeriesSplit",
    "train_test_split",
    "PurgedGroupTimeSeriesSplit",
    "RepeatedKFold",
    "ShuffleSplit",
    "GroupShuffleSplit",
}
MODEL_NAMES = {
    "LGBMClassifier",
    "LGBMRegressor",
    "LGBMRanker",
    "XGBClassifier",
    "XGBRegressor",
    "XGBRanker",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "LogisticRegression",
    "AutoModel",
    "AutoModelForSequenceClassification",
    "AutoModelForCausalLM",
}
METRIC_NAMES = {
    "roc_auc_score",
    "log_loss",
    "accuracy_score",
    "f1_score",
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "r2_score",
    "average_precision_score",
    "cohen_kappa_score",
    "ndcg_score",
    "mean_absolute_percentage_error",
}
FEATURE_OPERATION_NAMES = {
    "groupby",
    "agg",
    "merge",
    "rolling",
    "shift",
    "fillna",
    "cumsum",
    "rank",
    "TargetEncoder",
    "LabelEncoder",
    "OneHotEncoder",
    "StandardScaler",
}
SPLITTER_KWARGS = {
    "n_splits",
    "shuffle",
    "random_state",
    "groups",
    "test_size",
    "stratify",
}
MODEL_KWARGS = {
    "n_estimators",
    "learning_rate",
    "max_depth",
    "num_leaves",
}
_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?%?"
SCORE_EXPLICIT_PATTERN = re.compile(
    rf"(?P<label>[A-Za-z][A-Za-z0-9_./ -]{{0,48}}?)\s*(?::|=)\s*" rf"(?P<value>{_NUMBER_PATTERN})"
)
SCORE_ADJACENT_PATTERN = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z0-9_.-]*"
    r"(?:[ /_-]+[A-Za-z][A-Za-z0-9_.-]*){0,4})\s+"
    rf"(?P<value>{_NUMBER_PATTERN})"
)
ENCODED_TITLE_SCORE_PATTERN = re.compile(
    r"(?<!\d)(?:(?P<decimal>0\.\d+)|0[-_](?P<fraction>\d{2,4}))(?:\b|[-_ ])"
)
VALIDATION_CONTEXT_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:cv|oof|local|fold|validation|valid|val|holdout|offline|"
    r"cross[-_ ]validation)(?![a-z0-9])",
    re.IGNORECASE,
)
LEADERBOARD_CONTEXT_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:lb|leaderboard|public|private|submission|online|"
    r"public[-_ ]score|private[-_ ]score|public[-_ ]lb|private[-_ ]lb)"
    r"(?![a-z0-9])",
    re.IGNORECASE,
)
_CV_SPLIT_SIGNALS = (
    ("cv", re.compile(r"\bcv\b")),
    ("oof", re.compile(r"\boof\b")),
    ("local", re.compile(r"\blocal\b")),
    ("fold", re.compile(r"\bfold\b")),
    ("validation", re.compile(r"\bvalidation\b")),
    ("valid", re.compile(r"\bvalid\b")),
    ("val", re.compile(r"\bval\b")),
    ("holdout", re.compile(r"\bholdout\b")),
    ("offline", re.compile(r"\boffline\b")),
    ("cross-validation", re.compile(r"\bcross validation\b")),
)
_LB_SPLIT_SIGNALS = (
    ("lb", re.compile(r"\blb\b")),
    ("leaderboard", re.compile(r"\bleaderboard\b")),
    ("public", re.compile(r"\bpublic\b")),
    ("private", re.compile(r"\bprivate\b")),
    ("submission", re.compile(r"\bsubmission\b")),
    ("online", re.compile(r"\bonline\b")),
)
EXCLUDED_SCORE_LABELS = frozenset(
    {
        "abs_tol",
        "atol",
        "batch_size",
        "default",
        "dimension",
        "dropout",
        "delta",
        "embed_dim",
        "embedding_dimension",
        "epoch",
        "epochs",
        "early_stopping_min_delta",
        "eps",
        "epsilon",
        "fold",
        "factor",
        "frac",
        "frac_per_id",
        "fraction",
        "ftol",
        "gtol",
        "image_size",
        "img_size",
        "input_size",
        "k",
        "k1",
        "k2",
        "lambda",
        "learning_rate",
        "lr",
        "margin",
        "max_epochs",
        "min_delta",
        "n_splits",
        "num_folds",
        "num_workers",
        "patience",
        "random_state",
        "rel_tol",
        "rtol",
        "resolution",
        "scale",
        "seed",
        "split",
        "steps",
        "test_split",
        "threshold",
        "tol",
        "top_k",
        "total_steps",
        "train_split",
        "val_split",
        "version",
        "warmup_steps",
        "weight_decay",
        "workers",
        "xtol",
    }
)
CANONICAL_METRIC_ALIASES: dict[str, str] = {
    "accuracy": "accuracy",
    "acc": "accuracy",
    "accuracyscore": "accuracy",
    "auc": "roc_auc",
    "f1": "f1",
    "f1score": "f1",
    "identitybalancedmap": "mAP",
    "logloss": "log_loss",
    "mae": "mae",
    "map": "mAP",
    "meanabsoluteerror": "mae",
    "meanaverageprecision": "mAP",
    "mse": "mse",
    "rank1": "rank-1",
    "rmse": "rmse",
    "rocauc": "roc_auc",
    "top1": "rank-1",
}
GENERIC_METRIC_LABELS = frozenset(
    {
        "bestscore",
        "cv",
        "lb",
        "leaderboard",
        "oof",
        "privatelb",
        "privatescore",
        "publiclb",
        "publicscore",
        "score",
        "scored",
        "scores",
        "validation",
    }
)
MAX_EXPRESSION_LENGTH = 80
MAX_METRIC_LABEL_CHARS = 24
MAX_METRIC_LABEL_TOKENS = 3
METRIC_VALUE_RANGE: dict[str, tuple[float, float]] = {
    "auc": (0.0, 1.0),
    "average_precision": (0.0, 1.0),
    "accuracy": (0.0, 1.0),
    "f1": (0.0, 1.0),
    "precision": (0.0, 1.0),
    "recall": (0.0, 1.0),
    "map": (0.0, 1.0),
    "ap50": (0.0, 1.0),
    "ap75": (0.0, 1.0),
    "iou": (0.0, 1.0),
    "dice": (0.0, 1.0),
    "top1": (0.0, 1.0),
    "top5": (0.0, 1.0),
    "kappa": (-1.0, 1.0),
    "mcc": (-1.0, 1.0),
    "r2": (float("-inf"), 1.0),
    "gini": (-1.0, 1.0),
}
_METRIC_RANGE_ALIASES = {
    "mAP": "map",
    "rank-1": "top1",
    "roc_auc": "auc",
}


def extract_observations(
    notebook_path: Path,
    *,
    metric_hints: Iterable[str] = (),
) -> dict[str, Any]:
    metric_hints = _normalized_metric_hints(metric_hints)
    try:
        notebook = _read_notebook(notebook_path)
        cells = _notebook_cells(notebook)
    except Exception:
        return _empty_result()

    observations: dict[str, list[CodeObservation]] = {
        "splitters": [],
        "models": [],
        "metrics": [],
        "feature_ops": [],
    }
    declared_cv: list[str] = []
    declared_cv_observations: list[DeclaredCvObservation] = []
    score_observations: list[ScoreObservation] = []
    score_candidates_seen = 0
    score_candidates_excluded = 0
    seen_cv: set[str] = set()
    parsed_code_cells = 0
    code_cell_index = 0
    has_syntax_error = False

    for cell_index, cell in enumerate(cells):
        cell_type = _cell_value(cell, "cell_type")
        source = _cell_source(cell)
        if cell_type == "markdown":
            added, candidates_seen, candidates_excluded = extract_score_observations(
                source,
                locator=f"cell_{cell_index}",
                source="markdown",
                metric_hints=metric_hints,
            )
            score_observations.extend(added)
            score_candidates_seen += candidates_seen
            score_candidates_excluded += candidates_excluded
            _append_declared_cv_from_scores(
                added,
                declared_cv,
                seen_cv,
                declared_cv_observations,
            )
            continue
        if cell_type != "code":
            continue

        locator = f"cell_{code_cell_index}"
        code_cell_index += 1
        python_source = _strip_magics(source)
        try:
            tree = _parse_source(python_source)
        except SyntaxError:
            has_syntax_error = True
            continue

        parsed_code_cells += 1
        visitor = _ObservationVisitor(
            source=python_source,
            locator=locator,
            observations=observations,
        )
        visitor.visit(tree)
        added, candidates_seen, candidates_excluded = _code_score_observations(
            tree,
            python_source,
            locator,
            metric_hints=metric_hints,
        )
        score_observations.extend(added)
        score_candidates_seen += candidates_seen
        score_candidates_excluded += candidates_excluded
        for string_value in _string_literals_in_source_order(tree):
            added, candidates_seen, candidates_excluded = extract_score_observations(
                string_value,
                locator=locator,
                source="code_string",
                metric_hints=metric_hints,
            )
            score_observations.extend(added)
            score_candidates_seen += candidates_seen
            score_candidates_excluded += candidates_excluded
            _append_declared_cv_from_scores(
                added,
                declared_cv,
                seen_cv,
                declared_cv_observations,
            )

    if parsed_code_cells == 0:
        return _empty_result()

    return {
        **observations,
        "declared_cv": declared_cv,
        "declared_cv_observations": declared_cv_observations,
        "score_observations": score_observations,
        "score_candidates_seen": score_candidates_seen,
        "score_candidates_excluded": score_candidates_excluded,
        "parse_status": "partial" if has_syntax_error else "ok",
    }


def ast_fingerprint(notebook_path: Path) -> str:
    try:
        notebook = _read_notebook(notebook_path)
        cells = _notebook_cells(notebook)
    except Exception:
        return _fingerprint_parts(["<unparsed>"])

    tree_parts: list[str] = []
    for cell in cells:
        if _cell_value(cell, "cell_type") != "code":
            continue
        source = _strip_magics(_cell_source(cell))
        try:
            tree = _parse_source(source)
        except (SyntaxError, ValueError, TypeError):
            tree_parts.append("<unparsed>")
            continue

        normalized_tree = _FingerprintNormalizer().visit(tree)
        tree_parts.append(ast.dump(normalized_tree, annotate_fields=False))

    return _fingerprint_parts(tree_parts)


def assign_lineage_clusters(facts: list[NotebookFacts]) -> list[NotebookFacts]:
    # TODO: Task 74 collect_facts must call assign_lineage_clusters after all notebook fingerprints are collected.
    return [
        fact.model_copy(
            update={
                "lineage_cluster_id": f"lc_{fact.ast_fingerprint[:12]}",
            }
        )
        for fact in facts
    ]


class _ObservationVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        source: str,
        locator: str,
        observations: dict[str, list[CodeObservation]],
    ) -> None:
        self.source = source
        self.locator = locator
        self.observations = observations

    def visit_Call(self, node: ast.Call) -> None:
        name = _callee_name(node.func)
        category, allowed_kwargs = _observation_category(name)
        if category is not None:
            kwargs = _call_kwargs(
                node,
                source=self.source,
                allowed_names=allowed_kwargs,
            )
            self.observations[category].append(
                CodeObservation(
                    name=name,
                    kwargs=kwargs,
                    locator=self.locator,
                )
            )
        self.generic_visit(node)


class _FingerprintNormalizer(ast.NodeTransformer):
    def visit_Module(self, node: ast.Module) -> ast.AST:
        node.body = _without_leading_docstring(node.body)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.body = _without_leading_docstring(node.body)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.body = _without_leading_docstring(node.body)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.body = _without_leading_docstring(node.body)
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        return ast.copy_location(ast.Constant(value="?"), node)


def _read_notebook(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _notebook_cells(notebook: Any) -> list[Any]:
    cells = _cell_value(notebook, "cells")
    if not isinstance(cells, list):
        raise ValueError("Notebook cells must be a list")
    return cells


def _cell_value(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _cell_source(cell: Any) -> str:
    source = _cell_value(cell, "source")
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    if source is None:
        return ""
    return str(source)


def _strip_magics(source: str) -> str:
    string_lines: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type == tokenize.STRING:
                string_lines.update(range(token.start[0], token.end[0] + 1))
    except (IndentationError, tokenize.TokenError):
        pass

    stripped_lines: list[str] = []
    for line_number, line in enumerate(source.splitlines(keepends=True), start=1):
        if line_number in string_lines or not line.lstrip().startswith(("!", "%", "%%")):
            stripped_lines.append(line)
            continue
        if line.endswith("\r\n"):
            stripped_lines.append("\r\n")
        elif line.endswith(("\n", "\r")):
            stripped_lines.append(line[-1])
        else:
            stripped_lines.append("\n")
    return "".join(stripped_lines)


def _callee_name(function: ast.expr) -> str:
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _observation_category(name: str) -> tuple[str | None, set[str] | None]:
    if name in SPLITTER_NAMES:
        return "splitters", SPLITTER_KWARGS
    if name in MODEL_NAMES:
        return "models", MODEL_KWARGS
    if name in METRIC_NAMES:
        return "metrics", None
    if name in FEATURE_OPERATION_NAMES:
        return "feature_ops", None
    return None, None


def _call_kwargs(
    call: ast.Call,
    *,
    source: str,
    allowed_names: set[str] | None,
) -> dict[str, str]:
    kwargs: dict[str, str] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        if allowed_names is not None and keyword.arg not in allowed_names:
            continue
        kwargs[keyword.arg] = _keyword_value(keyword.value, source)
    return kwargs


def _keyword_value(value: ast.expr, source: str) -> str:
    if isinstance(value, ast.Constant):
        return str(value.value)
    if isinstance(value, ast.Name):
        return value.id

    segment = ast.get_source_segment(source, value)
    if segment is None:
        return "<expr>"
    compact_segment = " ".join(segment.strip().split())
    if compact_segment and len(compact_segment) <= MAX_EXPRESSION_LENGTH:
        return compact_segment
    return "<expr>"


def _string_literals_in_source_order(tree: ast.AST) -> list[str]:
    string_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    string_nodes.sort(
        key=lambda node: (
            getattr(node, "lineno", 0),
            getattr(node, "col_offset", 0),
        )
    )
    return [node.value for node in string_nodes]


def extract_score_observations(
    text: str,
    *,
    locator: str,
    source: Literal["markdown", "code", "code_string", "title", "ref"],
    metric_hints: Iterable[str] = (),
) -> tuple[list[ScoreObservation], int, int]:
    metric_hints = _normalized_metric_hints(metric_hints)
    observations: list[ScoreObservation] = []
    candidates_seen = 0
    candidates_excluded = 0
    occupied_ranges: list[tuple[int, int]] = []

    for pattern, implicit_position in (
        (SCORE_EXPLICIT_PATTERN, False),
        (SCORE_ADJACENT_PATTERN, True),
    ):
        for match in pattern.finditer(text):
            if any(match.start() < end and match.end() > start for start, end in occupied_ranges):
                continue
            occupied_ranges.append(match.span())
            candidates_seen += 1
            metric_raw = " ".join(match.group("label").strip().split()) or None
            value_raw = match.group("value")
            excluded_label = _is_excluded_score_label(metric_raw)
            if excluded_label:
                candidates_excluded += 1
            if not excluded_label and not _is_text_score_position(
                metric_raw, value_raw, implicit=implicit_position, metric_hints=metric_hints
            ):
                candidates_excluded += 1
                continue
            value = _score_value(value_raw)
            if value is None:
                candidates_excluded += 1
                continue
            raw_text = " ".join(match.group(0).strip().split())
            metric_canonical = canonicalize_metric_label(
                metric_raw,
                metric_hints=metric_hints,
            )
            plausible, implausible_reason, metric_canonical, value = _score_plausibility(
                metric_raw=metric_raw,
                metric_canonical=metric_canonical,
                value=value,
                raw_text=raw_text,
            )
            observations.append(
                ScoreObservation(
                    value=value,
                    value_raw=value_raw,
                    metric_raw=metric_raw,
                    metric_canonical=metric_canonical,
                    locator=locator,
                    raw_text=raw_text,
                    source=source,
                    source_kind=source,
                    context_text=_score_context(text, match.start(), match.end()),
                    plausible=plausible,
                    implausible_reason=implausible_reason,
                )
            )

    if source in {"title", "ref"}:
        title_match = ENCODED_TITLE_SCORE_PATTERN.search(text)
        if title_match is not None and not occupied_ranges:
            candidates_seen += 1
            decimal = title_match.group("decimal")
            value_raw = decimal or f"0.{title_match.group('fraction')}"
            observations.append(
                ScoreObservation(
                    value=float(value_raw),
                    value_raw=value_raw,
                    metric_raw=None,
                    metric_canonical=None,
                    locator=locator,
                    raw_text=title_match.group(0).rstrip("-_"),
                    source=source,
                    source_kind=source,
                    context_text=" ".join(text.strip().split())[:320],
                )
            )

    return observations, candidates_seen, candidates_excluded


def _score_context(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    context = text[line_start:line_end]
    return " ".join(context.strip().split())[:320]


def _code_score_observations(
    tree: ast.AST,
    source_text: str,
    locator: str,
    *,
    metric_hints: Iterable[str] = (),
) -> tuple[list[ScoreObservation], int, int]:
    metric_hints = _normalized_metric_hints(metric_hints)
    observations: list[ScoreObservation] = []
    candidates_seen = 0
    candidates_excluded = 0
    candidates: list[tuple[str, ast.expr, ast.AST]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                label = _assignment_label(target, source_text)
                if label is not None:
                    candidates.append((label, node.value, node))
        elif isinstance(node, ast.AnnAssign):
            label = _assignment_label(node.target, source_text)
            if label is not None and node.value is not None:
                candidates.append((label, node.value, node))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and key.value.strip()
                ):
                    candidates.append((key.value.strip(), value, node))

    for metric_raw, value_node, context_node in candidates:
        numeric = _numeric_literal(value_node)
        if numeric is None:
            continue
        candidates_seen += 1
        excluded_label = _is_excluded_score_label(metric_raw)
        if excluded_label:
            candidates_excluded += 1
        if not excluded_label and not _is_code_score_label(
            metric_raw,
            metric_hints=metric_hints,
        ):
            candidates_excluded += 1
            continue
        value, value_raw = numeric
        raw_text = ast.get_source_segment(source_text, context_node)
        compact_text = " ".join((raw_text or f"{metric_raw}={value_raw}").split())
        metric_canonical = canonicalize_metric_label(
            metric_raw,
            metric_hints=metric_hints,
        )
        (
            plausible,
            implausible_reason,
            metric_canonical,
            value,
        ) = _score_plausibility(
            metric_raw=metric_raw,
            metric_canonical=metric_canonical,
            value=value,
            raw_text=compact_text,
        )
        observations.append(
            ScoreObservation(
                value=value,
                value_raw=value_raw,
                metric_raw=metric_raw,
                metric_canonical=metric_canonical,
                locator=locator,
                raw_text=compact_text[:160],
                source="code",
                source_kind="code",
                context_text=compact_text[:320],
                plausible=plausible,
                implausible_reason=implausible_reason,
            )
        )
    return observations, candidates_seen, candidates_excluded


def _assignment_label(target: ast.expr, source_text: str) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Subscript):
        slice_value = target.slice
        if isinstance(slice_value, ast.Constant) and isinstance(slice_value.value, str):
            return slice_value.value.strip() or None
    segment = ast.get_source_segment(source_text, target)
    return segment.strip() if segment and len(segment.strip()) <= 48 else None


def _numeric_literal(value: ast.expr) -> tuple[float, str] | None:
    sign = 1
    constant = value
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
        sign = -1
        constant = value.operand
    if (
        not isinstance(constant, ast.Constant)
        or isinstance(constant.value, bool)
        or not isinstance(constant.value, int | float)
    ):
        return None
    number = float(constant.value) * sign
    if not math.isfinite(number):
        return None
    value_raw = repr(number) if isinstance(constant.value, float) else str(int(number))
    return number, value_raw


def canonicalize_metric_label(
    metric_raw: str | None,
    *,
    metric_hints: Iterable[str] = (),
) -> str | None:
    if not metric_raw:
        return None
    alias = _canonical_metric_alias(metric_raw)
    if alias is not None:
        return alias
    match_key = _metric_match_key(metric_raw)
    for metric_hint in _normalized_metric_hints(metric_hints):
        if match_key and match_key == _metric_match_key(metric_hint):
            return _canonical_metric_alias(metric_hint) or metric_hint
    return None


def _canonical_metric_alias(metric_raw: str) -> str | None:
    normalized = "".join(character for character in metric_raw.lower() if character.isalnum())
    for prefix in ("validation", "local", "oof", "fold", "val", "cv"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    direct = CANONICAL_METRIC_ALIASES.get(normalized)
    if direct is not None:
        return direct
    if "identitybalancedmap" in normalized or "meanaverageprecision" in normalized:
        return "mAP"
    token_candidates = {
        canonical
        for token in re.findall(r"[a-z]+\d*", metric_raw.lower())
        if (canonical := CANONICAL_METRIC_ALIASES.get(token)) is not None
    }
    return next(iter(token_candidates)) if len(token_candidates) == 1 else None


def _metric_match_key(metric_raw: str) -> str:
    tokens = re.findall(r"[a-z]+\d*|\d+", metric_raw.lower())
    context_tokens = {
        "cv",
        "epoch",
        "fold",
        "local",
        "oof",
        "val",
        "validation",
    }
    while tokens and (tokens[0] in context_tokens or tokens[0].isdigit()):
        tokens.pop(0)
    return "".join(tokens)


def _normalized_metric_hints(metric_hints: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            metric_hint.strip()
            for metric_hint in metric_hints
            if metric_hint and metric_hint.strip()
        )
    )


def classify_score_split(
    *,
    source_kind: str,
    context_text: str,
    context_signals: Sequence[str],
    metric_raw: str | None,
    locator: str,
) -> tuple[ScoreSplit, list[str]]:
    if _is_excluded_score_label(metric_raw):
        return "unknown", []

    primary_text = " ".join(
        part
        for part in (
            metric_raw or "",
            " ".join(context_signals),
        )
        if part
    )
    primary_cv, primary_lb = _score_split_signals(primary_text)
    if primary_cv and not primary_lb:
        return "cv", list(dict.fromkeys(primary_cv))
    if primary_lb and not primary_cv:
        return "lb", list(dict.fromkeys(primary_lb))
    if primary_cv and primary_lb:
        context_cv, context_lb = _score_split_signals(context_text)
        return _split_from_signals(
            list(dict.fromkeys([*context_cv, *primary_cv])),
            list(dict.fromkeys([*context_lb, *primary_lb])),
        )

    context_has_multiple_values = len(list(SCORE_EXPLICIT_PATTERN.finditer(context_text))) > 1
    if context_has_multiple_values:
        context_cv, context_lb = _score_split_signals(context_text)
        if context_lb and not context_cv and canonicalize_metric_label(metric_raw) is not None:
            return "cv", ["paired-with-lb"]
        context_cv, context_lb = [], []
    else:
        context_cv, context_lb = _score_split_signals(" ".join((context_text, locator)))
    if context_cv or context_lb:
        return _split_from_signals(context_cv, context_lb)
    if source_kind in {"title", "ref", "notebook_title", "notebook_ref"}:
        return "lb", [f"source:{source_kind}"]
    return "unknown", []


def _score_split_signals(searchable: str) -> tuple[list[str], list[str]]:
    searchable = searchable.lower()
    searchable = re.sub(r"[-_/]+", " ", searchable)
    searchable = re.sub(r"\s+", " ", searchable)
    cv_signals = [name for name, pattern in _CV_SPLIT_SIGNALS if pattern.search(searchable)]
    lb_signals = [name for name, pattern in _LB_SPLIT_SIGNALS if pattern.search(searchable)]
    return cv_signals, lb_signals


def _split_from_signals(
    cv_signals: list[str],
    lb_signals: list[str],
) -> tuple[ScoreSplit, list[str]]:
    signals = list(dict.fromkeys([*cv_signals, *lb_signals]))
    if cv_signals and not lb_signals:
        return "cv", signals
    if lb_signals and not cv_signals:
        return "lb", signals
    if cv_signals and lb_signals:
        return "unknown", signals
    return "unknown", []


def metric_optimization_direction(
    metric_name: str | None,
) -> Literal["maximize", "minimize"] | None:
    canonical = canonicalize_metric_label(metric_name) or (
        metric_name.strip().lower() if metric_name else None
    )
    if canonical in {"log_loss", "logloss", "mae", "mse", "rmse", "mape"}:
        return "minimize"
    if canonical in {
        "accuracy",
        "f1",
        "mAP",
        "map",
        "rank-1",
        "roc_auc",
        "r2",
        "ndcg",
    }:
        return "maximize"
    return None


def _score_observation_id(notebook_ref: str, observation: ScoreObservation) -> str:
    identity = "\x1f".join(
        (
            notebook_ref,
            observation.source_kind or observation.source,
            observation.locator,
            observation.value_raw,
            observation.metric_raw or "",
            observation.raw_text,
        )
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return f"score-observation:{digest}"


def recanonicalize_score_observations(
    notebooks: list[NotebookFacts],
    *,
    competition_metric_name: str | None,
) -> list[NotebookFacts]:
    competition_metric_canonical = canonicalize_metric_label(competition_metric_name)
    metric_hints = _normalized_metric_hints(
        metric_hint
        for metric_hint in (
            competition_metric_name,
            *(metric.name for notebook in notebooks for metric in notebook.metrics),
        )
        if metric_hint is not None
    )
    canonicalized_notebooks: list[NotebookFacts] = []
    for notebook in notebooks:
        score_observations: list[ScoreObservation] = []
        for observation in notebook.score_observations:
            source_kind = observation.source_kind or observation.source
            metric_canonical = (
                canonicalize_metric_label(
                    observation.metric_raw,
                    metric_hints=metric_hints,
                )
                or observation.metric_canonical
            )
            if (
                metric_canonical is None
                and competition_metric_canonical is not None
                and _is_generic_metric_label(observation.metric_raw, source_kind)
            ):
                metric_canonical = competition_metric_canonical
            (
                plausible,
                implausible_reason,
                metric_canonical,
                normalized_value,
            ) = _score_plausibility(
                metric_raw=observation.metric_raw,
                metric_canonical=metric_canonical,
                value=observation.value,
                raw_text=observation.raw_text,
            )
            split, split_signals = classify_score_split(
                source_kind=source_kind,
                context_text=observation.context_text or observation.raw_text,
                context_signals=[],
                metric_raw=observation.metric_raw,
                locator=observation.locator,
            )
            updated = observation.model_copy(
                update={
                    "metric_canonical": metric_canonical,
                    "value": normalized_value,
                    "split": split,
                    "split_signals": split_signals,
                    "source_kind": source_kind,
                    "optimization_direction": metric_optimization_direction(
                        metric_canonical or observation.metric_raw
                    ),
                    "plausible": plausible,
                    "implausible_reason": implausible_reason,
                }
            )
            score_observations.append(
                updated.model_copy(
                    update={
                        "observation_id": observation.observation_id
                        or _score_observation_id(notebook.ref, updated)
                    }
                )
            )
        canonical_by_evidence = {
            (observation.locator, observation.raw_text, observation.value): (
                observation.metric_canonical
            )
            for observation in score_observations
        }
        cv_score_observations = [
            observation for observation in score_observations if observation.split == "cv"
        ]
        if cv_score_observations:
            declared_cv: list[str] = []
            declared_cv_observations: list[DeclaredCvObservation] = []
            seen_cv: set[str] = set()
            _append_declared_cv_from_scores(
                cv_score_observations,
                declared_cv,
                seen_cv,
                declared_cv_observations,
            )
        else:
            structured_evidence = set(canonical_by_evidence)
            legacy_is_fully_structured = bool(notebook.declared_cv_observations) and all(
                (
                    observation.locator,
                    observation.raw_text,
                    observation.value,
                )
                in structured_evidence
                for observation in notebook.declared_cv_observations
            )
            if legacy_is_fully_structured:
                declared_cv = []
                declared_cv_observations = []
            else:
                declared_cv = notebook.declared_cv
                declared_cv_observations = [
                    observation.model_copy(
                        update={
                            "metric_name": canonical_by_evidence.get(
                                (
                                    observation.locator,
                                    observation.raw_text,
                                    observation.value,
                                ),
                                observation.metric_name,
                            )
                        }
                    )
                    for observation in notebook.declared_cv_observations
                ]
        canonicalized_notebooks.append(
            notebook.model_copy(
                update={
                    "score_observations": score_observations,
                    "declared_cv": declared_cv,
                    "declared_cv_observations": declared_cv_observations,
                }
            )
        )
    return canonicalized_notebooks


def _is_generic_metric_label(metric_raw: str | None, source_kind: str) -> bool:
    if metric_raw is None:
        return source_kind in {"title", "ref", "notebook_title", "notebook_ref"}
    normalized = "".join(
        character for character in metric_raw.casefold() if character.isalnum()
    )
    return normalized in GENERIC_METRIC_LABELS


def diagnose_scores(notebooks: list[NotebookFacts]) -> ScoreDiagnostics:
    observations = [
        observation for notebook in notebooks for observation in notebook.score_observations
    ]
    plausible_observations = [observation for observation in observations if observation.plausible]
    implausible_observations = Counter(
        observation.implausible_reason or "unspecified"
        for observation in observations
        if not observation.plausible
    )
    return ScoreDiagnostics(
        notebooks_total=len(notebooks),
        notebooks_with_score_observations=sum(
            any(observation.plausible for observation in notebook.score_observations)
            for notebook in notebooks
        ),
        observations_total=len(plausible_observations),
        observations_with_raw_metric=sum(
            observation.metric_raw is not None for observation in plausible_observations
        ),
        observations_with_canonical_metric=sum(
            observation.metric_canonical is not None for observation in plausible_observations
        ),
        observations_without_canonical_metric=sum(
            observation.metric_canonical is None for observation in plausible_observations
        ),
        title_or_ref_observations=sum(
            observation.source in {"title", "ref"}
            for observation in plausible_observations
        ),
        candidates_seen=sum(notebook.score_candidates_seen for notebook in notebooks),
        candidates_excluded=sum(notebook.score_candidates_excluded for notebook in notebooks),
        split_cv=sum(observation.split == "cv" for observation in plausible_observations),
        split_lb=sum(observation.split == "lb" for observation in plausible_observations),
        split_unknown=sum(
            observation.split == "unknown" for observation in plausible_observations
        ),
        notebooks_with_cv_scores=sum(
            any(
                observation.plausible and observation.split == "cv"
                for observation in notebook.score_observations
            )
            for notebook in notebooks
        ),
        notebooks_with_lb_scores=sum(
            any(
                observation.plausible and observation.split == "lb"
                for observation in notebook.score_observations
            )
            for notebook in notebooks
        ),
        notebooks_with_both_sides=sum(
            any(
                observation.plausible and observation.split == "cv"
                for observation in notebook.score_observations
            )
            and any(
                observation.plausible and observation.split == "lb"
                for observation in notebook.score_observations
            )
            for notebook in notebooks
        ),
        implausible_observations=dict(sorted(implausible_observations.items())),
    )


def _append_declared_cv_from_scores(
    score_observations: list[ScoreObservation],
    values: list[str],
    seen: set[str],
    observations: list[DeclaredCvObservation],
) -> None:
    for score_observation in score_observations:
        if not score_observation.plausible:
            continue
        has_legacy_cv_signal = VALIDATION_CONTEXT_PATTERN.search(
            score_observation.context_text or score_observation.raw_text
        )
        if score_observation.split != "cv" and not (
            score_observation.split == "unknown" and has_legacy_cv_signal
        ):
            continue
        score = (
            f"{score_observation.value:.12g}"
            if score_observation.value_raw.endswith("%")
            else score_observation.value_raw
        )
        if score in seen:
            continue
        seen.add(score)
        values.append(score)
        observations.append(
            DeclaredCvObservation(
                value=score_observation.value,
                metric_name=score_observation.metric_canonical,
                locator=score_observation.locator,
                raw_text=score_observation.raw_text,
            )
        )


def _score_value(raw_value: str) -> float | None:
    try:
        value = float(raw_value.rstrip("%"))
    except ValueError:
        return None
    if raw_value.endswith("%"):
        value /= 100
    return value if math.isfinite(value) else None


def _score_plausibility(
    *,
    metric_raw: str | None,
    metric_canonical: str | None,
    value: float,
    raw_text: str,
) -> tuple[bool, str | None, str | None, float]:
    if metric_raw and (
        len(metric_raw) > MAX_METRIC_LABEL_CHARS
        or len(metric_raw.split()) > MAX_METRIC_LABEL_TOKENS
    ):
        return False, "label_too_long", metric_canonical, value
    if _is_excluded_score_label(metric_raw):
        return False, "excluded_label", metric_canonical, value

    range_key = _METRIC_RANGE_ALIASES.get(metric_canonical or "", metric_canonical)
    value_range = METRIC_VALUE_RANGE.get(range_key or "")
    if value_range is None:
        return True, None, metric_canonical, value

    lower, upper = value_range
    if value > upper and 0 < value <= 100 and "%" in raw_text:
        value /= 100
    if value < lower or value > upper:
        return False, "value_out_of_range", None, value
    if lower == 0.0 and value < 1e-3:
        return False, "value_implausibly_small", metric_canonical, value
    return True, None, metric_canonical, value


def _is_excluded_score_label(metric_raw: str | None) -> bool:
    if not metric_raw:
        return False
    normalized = re.sub(r"[^a-z0-9]+", "_", metric_raw.lower()).strip("_")
    return any(
        normalized == excluded or normalized.endswith(f"_{excluded}")
        for excluded in EXCLUDED_SCORE_LABELS
    )


def _is_code_score_label(
    metric_raw: str,
    *,
    metric_hints: Iterable[str] = (),
) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", metric_raw.lower()).strip("_")
    tokens = set(normalized.split("_"))
    return (
        bool(
            tokens
            & {
                "cv",
                "lb",
                "local",
                "metric",
                "oof",
                "private",
                "public",
                "score",
                "val",
                "validation",
            }
        )
        or canonicalize_metric_label(metric_raw, metric_hints=metric_hints) is not None
    )


def _is_text_score_position(
    metric_raw: str | None,
    value_raw: str,
    *,
    implicit: bool,
    metric_hints: Iterable[str] = (),
) -> bool:
    if not metric_raw:
        return False
    has_context = _has_score_position_marker(metric_raw) or (
        canonicalize_metric_label(metric_raw, metric_hints=metric_hints) is not None
    )
    if implicit:
        return has_context or _looks_like_metric_identifier(metric_raw)
    normalized_value = value_raw.lower()
    return has_context or any(marker in normalized_value for marker in (".", "%", "e"))


def _looks_like_metric_identifier(metric_raw: str) -> bool:
    return re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+", metric_raw) is not None


def _has_score_position_marker(metric_raw: str) -> bool:
    tokens = re.findall(r"[a-z]+\d*", metric_raw.lower())
    position_tokens = tokens[-2:]
    return any(
        token
        in {
            "cv",
            "lb",
            "local",
            "metric",
            "oof",
            "private",
            "public",
            "score",
            "scores",
            "val",
            "validation",
        }
        or token.startswith("score")
        for token in position_tokens
    )


def _parse_source(source: str) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source)


def _without_leading_docstring(statements: list[ast.stmt]) -> list[ast.stmt]:
    if not statements:
        return statements
    first_statement = statements[0]
    if (
        isinstance(first_statement, ast.Expr)
        and isinstance(first_statement.value, ast.Constant)
        and isinstance(first_statement.value.value, str)
    ):
        return statements[1:]
    return statements


def _fingerprint_parts(parts: list[str]) -> str:
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def _empty_result() -> dict[str, Any]:
    return {
        "splitters": [],
        "models": [],
        "metrics": [],
        "feature_ops": [],
        "declared_cv": [],
        "declared_cv_observations": [],
        "score_observations": [],
        "score_candidates_seen": 0,
        "score_candidates_excluded": 0,
        "parse_status": "failed",
    }
