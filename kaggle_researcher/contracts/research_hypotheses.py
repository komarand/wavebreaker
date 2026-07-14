"""Deprecated compatibility import for the canonical research contract."""

from kaggle_researcher.contracts.artifacts import (
    load_research_hypotheses,
    write_research_hypotheses_atomic,
)
from kaggle_researcher.contracts.errors import UnsupportedSchemaVersionError
from kaggle_researcher.contracts.migration import (
    HypothesisMigrationResult,
    LEGACY_CATEGORY_ALIASES,
    migrate_research_hypotheses_payload,
)
from kaggle_researcher.contracts.research import (
    ALLOWED_HYPOTHESIS_CATEGORIES,
    Confidence,
    HypothesisCategory,
    HypothesisStatus,
    Priority,
    ResearchHypothesis,
    ResearchHypotheses,
)
from kaggle_researcher.contracts.versions import CURRENT_SCHEMA_VERSION as SCHEMA_VERSION

__all__ = [
    "ALLOWED_HYPOTHESIS_CATEGORIES",
    "Confidence",
    "HypothesisCategory",
    "HypothesisMigrationResult",
    "HypothesisStatus",
    "LEGACY_CATEGORY_ALIASES",
    "Priority",
    "ResearchHypothesis",
    "ResearchHypotheses",
    "SCHEMA_VERSION",
    "UnsupportedSchemaVersionError",
    "load_research_hypotheses",
    "migrate_research_hypotheses_payload",
    "write_research_hypotheses_atomic",
]
