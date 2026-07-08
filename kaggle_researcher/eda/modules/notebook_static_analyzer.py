from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PatternSpec = tuple[str, tuple[str, ...], str]


CV_PATTERNS: tuple[PatternSpec, ...] = (
    ("kfold", ("KFold", "k-fold", "kfold"), "KFold cross-validation pattern observed."),
    (
        "stratified_kfold",
        ("StratifiedKFold", "stratified kfold", "stratified folds"),
        "StratifiedKFold pattern observed.",
    ),
    (
        "group_kfold",
        ("GroupKFold", "group kfold", "group folds"),
        "GroupKFold pattern observed.",
    ),
    (
        "stratified_group_kfold",
        ("StratifiedGroupKFold", "stratified group kfold"),
        "StratifiedGroupKFold pattern observed.",
    ),
    (
        "time_series_split",
        ("TimeSeriesSplit", "time series split", "expanding window"),
        "TimeSeriesSplit or expanding-window pattern observed.",
    ),
)
FEATURE_PATTERNS: tuple[PatternSpec, ...] = (
    (
        "target_encoding",
        ("target encoding", "targetencoder", "category_encoders.targetencoder", "woe"),
        "Target encoding or WoE pattern observed; treat as leakage-sensitive.",
    ),
    (
        "adversarial_validation",
        ("adversarial validation", "is_test", "train vs test classifier"),
        "Adversarial validation pattern observed.",
    ),
    (
        "missingness_features",
        ("isnull", "isna", "missing indicator", "null indicator"),
        "Missingness feature pattern observed.",
    ),
    (
        "date_features",
        ("dt.", "datepart", "dayofweek", "month", "week_num"),
        "Date-derived feature pattern observed.",
    ),
    (
        "aggregation_features",
        ("groupby", "agg(", ".agg", "aggregation", "mean encoding"),
        "Groupby/aggregation feature pattern observed.",
    ),
)
MODEL_PATTERNS: tuple[PatternSpec, ...] = (
    ("lightgbm", ("LightGBM", "LGBMClassifier", "LGBMRegressor", "lgb.train"), "LightGBM model pattern observed."),
    ("catboost", ("CatBoost", "CatBoostClassifier", "CatBoostRegressor"), "CatBoost model pattern observed."),
    ("xgboost", ("XGBoost", "XGBClassifier", "XGBRegressor", "xgb.train"), "XGBoost model pattern observed."),
    ("linear_model", ("LogisticRegression", "LinearRegression", "Ridge", "ElasticNet"), "Linear model pattern observed."),
)
METRIC_PATTERNS: tuple[PatternSpec, ...] = (
    ("roc_auc", ("roc_auc_score", "roc auc", "AUC"), "AUC metric code/text observed."),
    ("logloss_calibration", ("log_loss", "logloss", "calibration", "CalibratedClassifierCV"), "Logloss/calibration pattern observed."),
    ("rmsle_target_transform", ("RMSLE", "mean_squared_log_error", "log1p", "expm1"), "RMSLE/log target transform pattern observed."),
    ("qwk_threshold_optimization", ("quadratic_weighted_kappa", "cohen_kappa_score", "QWK", "threshold optimization"), "QWK threshold optimization pattern observed."),
    ("ndcg", ("ndcg_score", "NDCG", "ndcg@"), "NDCG/ranking metric pattern observed."),
)
POSTPROCESSING_PATTERNS: tuple[PatternSpec, ...] = (
    ("rank_averaging", ("rank averaging", "rankdata", "rank mean", "rank ensemble"), "Rank averaging postprocessing observed."),
    ("clipping", ("clip(", ".clip", "np.clip", "clipping"), "Prediction clipping observed."),
    ("threshold_tuning", ("threshold tuning", "best_threshold", "optimal threshold", "thresholds"), "Threshold tuning observed."),
    ("probability_calibration", ("calibration", "isotonic", "sigmoid", "Platt"), "Probability calibration postprocessing observed."),
)
LEADERBOARD_RISK_PATTERNS: tuple[PatternSpec, ...] = (
    ("public_lb_tuning", ("public lb", "leaderboard probing", "lb probing", "blend for lb"), "Public leaderboard tuning/probing language observed."),
    ("hardcoded_blend_weights", ("0.7*", "0.8*", "manual blend", "blend weights"), "Manual blend-weight pattern observed."),
    ("submission_overfit", ("submit many", "many submissions", "shake-up", "private lb"), "Leaderboard shake-up/submission-overfit pattern observed."),
)


def analyze_notebooks_static(
    notebook_sources: list[Any],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Extract static notebook/code-text patterns without executing notebooks."""

    documents = [_normalise_source(source) for source in notebook_sources]
    warnings: list[str] = []
    if not documents:
        warnings.append("No notebook source text was provided.")

    result = {
        "status": "completed" if documents else "skipped",
        "documents_analyzed": len(documents),
        "source_ids": [document["id"] for document in documents],
        "cv_strategy": _detect_patterns(documents, CV_PATTERNS),
        "feature_families": _detect_patterns(documents, FEATURE_PATTERNS),
        "model_families": _detect_patterns(documents, MODEL_PATTERNS),
        "metric_code": _detect_patterns(documents, METRIC_PATTERNS),
        "postprocessing": _detect_patterns(documents, POSTPROCESSING_PATTERNS),
        "suspicious_leaderboard_overfit_patterns": _detect_patterns(
            documents,
            LEADERBOARD_RISK_PATTERNS,
        ),
        "notebook_scores_are_observations_not_truth": True,
        "warnings": warnings + _contextual_warnings(documents),
        "limitations": [
            "Notebook analysis is static text/code pattern extraction only.",
            "No notebook code was executed.",
            "Observed notebook scores or claims are not treated as proof.",
        ],
    }
    if output_dir is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "notebook_static_analysis.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        result["artifacts"] = {"notebook_static_analysis": str(path / "notebook_static_analysis.json")}
    return result


def _detect_patterns(
    documents: list[dict[str, Any]],
    specs: tuple[PatternSpec, ...],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for pattern_name, aliases, description in specs:
        matches = []
        for document in documents:
            snippet = _first_snippet(document["content"], aliases)
            if snippet is None:
                continue
            matches.append(
                {
                    "source_id": document["id"],
                    "title": document["title"],
                    "snippet": snippet,
                }
            )
        if matches:
            findings.append(
                {
                    "pattern": pattern_name,
                    "count": len(matches),
                    "documents": matches,
                    "description": description,
                }
            )
    return findings


def _normalise_source(source: Any) -> dict[str, Any]:
    if hasattr(source, "model_dump"):
        payload = source.model_dump(mode="json")
    elif isinstance(source, dict):
        payload = dict(source)
    else:
        payload = {
            "id": getattr(source, "id", None),
            "title": getattr(source, "title", None),
            "source": getattr(source, "source", None),
            "url": getattr(source, "url", None),
            "content": getattr(source, "content", ""),
        }

    content = str(payload.get("content") or payload.get("text") or payload.get("summary") or "")
    title = str(payload.get("title") or payload.get("name") or payload.get("id") or "untitled")
    return {
        "id": str(payload.get("id") or payload.get("doc_id") or title),
        "title": title,
        "source": str(payload.get("source") or "unknown"),
        "url": str(payload.get("url") or ""),
        "content": _extract_ipynb_text_if_needed(content),
    }


def _extract_ipynb_text_if_needed(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("{") or '"cells"' not in stripped[:500]:
        return content
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return content
    cells = []
    for cell in payload.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            cells.append("".join(str(part) for part in source))
        else:
            cells.append(str(source))
    return "\n".join(cells) if cells else content


def _first_snippet(content: str, aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        match = re.search(re.escape(alias), content, flags=re.IGNORECASE)
        if match is None:
            continue
        start = max(0, match.start() - 60)
        end = min(len(content), match.end() + 80)
        return _compact_snippet(content[start:end])
    return None


def _compact_snippet(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= 220:
        return compact
    return compact[:217].rstrip() + "..."


def _contextual_warnings(documents: list[dict[str, Any]]) -> list[str]:
    combined = "\n".join(document["content"] for document in documents).lower()
    warnings: list[str] = []
    if "target encoding" in combined or "woe" in combined:
        warnings.append("Target encoding/WoE patterns require OOF or group/time-safe fitting.")
    if "public lb" in combined or "leaderboard probing" in combined or "lb probing" in combined:
        warnings.append("Leaderboard-tuning patterns are observations and should be audited before use.")
    if "threshold" in combined and ("f1" in combined or "qwk" in combined):
        warnings.append("Threshold tuning must be validated on held-out folds only.")
    if "logloss" in combined or "log_loss" in combined:
        warnings.append("Logloss patterns often require probability calibration and clipping checks.")
    return warnings


__all__ = ["analyze_notebooks_static"]
