from __future__ import annotations

import math
import statistics
import warnings
from typing import Any

from kaggle_researcher.facts.models import CvLbPair, NotebookFacts


def build_cv_lb_pairs(notebooks: list[NotebookFacts]) -> list[CvLbPair]:
    pairs: list[CvLbPair] = []
    for notebook in notebooks:
        if not notebook.declared_cv or notebook.public_score is None:
            continue
        declared_cv = _finite_float(notebook.declared_cv[0])
        public_score = _finite_float(notebook.public_score)
        if declared_cv is None or public_score is None:
            continue
        pairs.append(
            CvLbPair(
                notebook_ref=notebook.ref,
                declared_cv=declared_cv,
                public_score=public_score,
                lineage_cluster_id=notebook.lineage_cluster_id,
            )
        )
    return pairs


def summarize_cv_lb(pairs: list[CvLbPair]) -> dict[str, int | float | None]:
    count = len(pairs)
    gaps = [pair.declared_cv - pair.public_score for pair in pairs]
    return {
        "count": count,
        "mean_gap": statistics.fmean(gaps) if gaps else None,
        "median_gap": statistics.median(gaps) if gaps else None,
        "spearman": _spearman(pairs) if count >= 3 else None,
        "distinct_lineage_clusters": len(
            {pair.lineage_cluster_id for pair in pairs}
        ),
    }


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _spearman(pairs: list[CvLbPair]) -> float | None:
    try:
        from scipy.stats import spearmanr
    except (ImportError, ModuleNotFoundError):
        return None

    declared_values = [pair.declared_cv for pair in pairs]
    public_values = [pair.public_score for pair in pairs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(declared_values, public_values)
    statistic = getattr(result, "statistic", None)
    if statistic is None:
        try:
            statistic = result[0]
        except (IndexError, KeyError, TypeError):
            return None
    return _finite_float(statistic)
