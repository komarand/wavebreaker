from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


MODEL_REGISTRY_VERSION = "1.0"


@dataclass(frozen=True)
class ModelIdentity:
    canonical_family_id: str
    implementation_id: str
    display_name: str
    task_types: tuple[str, ...]
    aliases: tuple[str, ...]
    available: bool
    availability_reason: str | None
    capabilities: Mapping[str, Any]
    registry_version: str = MODEL_REGISTRY_VERSION

    def model_dump(self) -> dict[str, Any]:
        return {
            "canonical_family_id": self.canonical_family_id,
            "implementation_id": self.implementation_id,
            "display_name": self.display_name,
            "task_types": list(self.task_types),
            "aliases": list(self.aliases),
            "available": self.available,
            "availability_reason": self.availability_reason,
            "capabilities": dict(self.capabilities),
            "registry_version": self.registry_version,
        }


def _identity(
    family: str,
    implementation: str,
    display: str,
    task_types: Iterable[str],
    aliases: Iterable[str],
    **capabilities: Any,
) -> ModelIdentity:
    return ModelIdentity(
        canonical_family_id=family,
        implementation_id=implementation,
        display_name=display,
        task_types=tuple(task_types),
        aliases=tuple(dict.fromkeys((family, implementation, display, *aliases))),
        available=True,
        availability_reason=None,
        capabilities=capabilities,
    )


_MODELS = (
    _identity(
        "sklearn_hist_gradient_boosting",
        "sklearn.ensemble.HistGradientBoostingClassifier",
        "HistGradientBoostingClassifier",
        ("binary_classification", "multiclass_classification", "classification"),
        (
            "sklearn_hist_gradient_boosting_classifier",
            "hist_gradient_boosting_classifier",
            "histgradientboostingclassifier",
        ),
        probabilities=True,
        native_missing_values=True,
        preprocessing_contract="numeric matrix after fold-fitted preprocessing",
    ),
    _identity(
        "sklearn_logistic_regression",
        "sklearn.linear_model.LogisticRegression",
        "LogisticRegression",
        ("binary_classification", "multiclass_classification", "classification"),
        ("logistic_regression", "sklearn_logistic_regression_classifier"),
        probabilities=True,
        native_missing_values=False,
        preprocessing_contract="scaled/encoded matrix fitted inside folds",
    ),
    _identity(
        "sklearn_random_forest_classifier",
        "sklearn.ensemble.RandomForestClassifier",
        "RandomForestClassifier",
        ("binary_classification", "multiclass_classification", "classification"),
        ("random_forest", "random_forest_classifier"),
        probabilities=True,
        native_missing_values=False,
        preprocessing_contract="encoded matrix fitted inside folds",
    ),
    _identity(
        "sklearn_hist_gradient_boosting_regression",
        "sklearn.ensemble.HistGradientBoostingRegressor",
        "HistGradientBoostingRegressor",
        ("regression",),
        (
            "sklearn_hist_gradient_boosting_regressor",
            "hist_gradient_boosting_regressor",
            "histgradientboostingregressor",
        ),
        probabilities=False,
        native_missing_values=True,
        preprocessing_contract="numeric matrix after fold-fitted preprocessing",
    ),
    _identity(
        "sklearn_linear_regression",
        "sklearn.linear_model.LinearRegression",
        "LinearRegression",
        ("regression",),
        ("linear_regression", "ordinary_least_squares"),
        probabilities=False,
        native_missing_values=False,
        preprocessing_contract="imputed/encoded matrix fitted inside folds",
    ),
    _identity(
        "sklearn_random_forest_regression",
        "sklearn.ensemble.RandomForestRegressor",
        "RandomForestRegressor",
        ("regression",),
        ("random_forest_regressor",),
        probabilities=False,
        native_missing_values=False,
        preprocessing_contract="encoded matrix fitted inside folds",
    ),
)


def _key(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


_ALIASES = {
    _key(alias): identity
    for identity in _MODELS
    for alias in identity.aliases
}


def model_registry() -> tuple[ModelIdentity, ...]:
    return _MODELS


def resolve_model_identity(value: str | None) -> ModelIdentity | None:
    if not value:
        return None
    normalized = _key(value)
    direct = _ALIASES.get(normalized)
    if direct is not None:
        return direct
    # Legacy values sometimes joined two alternatives with a slash. Resolve only
    # when exactly one registered identity is named; an ambiguous bundle is not a model.
    matches = {
        identity for alias, identity in _ALIASES.items() if alias and alias in normalized
    }
    return next(iter(matches)) if len(matches) == 1 else None


def supported_models(task_type: str | None) -> tuple[ModelIdentity, ...]:
    normalized = (task_type or "").casefold()
    return tuple(
        identity
        for identity in _MODELS
        if identity.available and normalized in identity.task_types
    )


def distinct_candidate(
    baseline: ModelIdentity | None,
    task_type: str | None,
) -> ModelIdentity | None:
    return next(
        (
            identity
            for identity in supported_models(task_type)
            if baseline is None
            or identity.canonical_family_id != baseline.canonical_family_id
        ),
        None,
    )


def is_valid_model_comparison(
    baseline: ModelIdentity | None,
    candidate: ModelIdentity | None,
) -> bool:
    return bool(
        baseline
        and candidate
        and baseline.available
        and candidate.available
        and baseline.canonical_family_id != candidate.canonical_family_id
    )


__all__ = [
    "MODEL_REGISTRY_VERSION",
    "ModelIdentity",
    "distinct_candidate",
    "is_valid_model_comparison",
    "model_registry",
    "resolve_model_identity",
    "supported_models",
]
