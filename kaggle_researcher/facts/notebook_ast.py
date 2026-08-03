from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import tokenize
from pathlib import Path
from typing import Any

from kaggle_researcher.facts.models import (
    CodeObservation,
    DeclaredCvObservation,
    NotebookFacts,
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
DECLARED_CV_PATTERN = re.compile(
    r"\b(?:cv|oof|local|fold|validation)\b[^\n]{0,40}?(0\.\d{3,})",
    re.IGNORECASE,
)
DECLARED_CV_METRICS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bmAP\b", re.IGNORECASE), "mAP"),
    (re.compile(r"\b(?:top|rank)[-_ ]?1\b", re.IGNORECASE), "rank-1"),
    (re.compile(r"\baccuracy\b", re.IGNORECASE), "accuracy"),
)
MAX_EXPRESSION_LENGTH = 80


def extract_observations(notebook_path: Path) -> dict[str, Any]:
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
    seen_cv: set[str] = set()
    parsed_code_cells = 0
    code_cell_index = 0
    has_syntax_error = False

    for cell_index, cell in enumerate(cells):
        cell_type = _cell_value(cell, "cell_type")
        source = _cell_source(cell)
        if cell_type == "markdown":
            _append_declared_cv(
                source,
                declared_cv,
                seen_cv,
                declared_cv_observations,
                locator=f"cell_{cell_index}",
            )
            continue
        if cell_type != "code":
            continue

        locator = f"cell_{code_cell_index}"
        code_cell_index += 1
        python_source = _strip_magics(source)
        try:
            tree = ast.parse(python_source)
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
        for string_value in _string_literals_in_source_order(tree):
            _append_declared_cv(
                string_value,
                declared_cv,
                seen_cv,
                declared_cv_observations,
                locator=locator,
            )

    if parsed_code_cells == 0:
        return _empty_result()

    return {
        **observations,
        "declared_cv": declared_cv,
        "declared_cv_observations": declared_cv_observations,
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
            tree = ast.parse(source)
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
        if line_number in string_lines or not line.lstrip().startswith(
            ("!", "%", "%%")
        ):
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


def _append_declared_cv(
    text: str,
    values: list[str],
    seen: set[str],
    observations: list[DeclaredCvObservation],
    *,
    locator: str,
) -> None:
    for match in DECLARED_CV_PATTERN.finditer(text):
        score = match.group(1)
        if score in seen:
            continue
        seen.add(score)
        values.append(score)
        raw_text = " ".join(match.group(0).strip().split())
        observations.append(
            DeclaredCvObservation(
                value=float(score),
                metric_name=_declared_cv_metric(raw_text),
                locator=locator,
                raw_text=raw_text,
            )
        )


def _declared_cv_metric(text: str) -> str | None:
    for pattern, metric_name in DECLARED_CV_METRICS:
        if pattern.search(text):
            return metric_name
    return None


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
        "parse_status": "failed",
    }
