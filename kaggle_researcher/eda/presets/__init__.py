"""Competition-specific hints for the Kaggle EDA Engine."""

from kaggle_researcher.eda.presets.base import CompetitionPreset
from kaggle_researcher.eda.presets.home_credit_crms import HOME_CREDIT_CRMS_PRESET
from kaggle_researcher.eda.presets.registry import (
    PresetRegistry,
    get_preset,
    normalize_competition_id,
)

__all__ = [
    "CompetitionPreset",
    "HOME_CREDIT_CRMS_PRESET",
    "PresetRegistry",
    "get_preset",
    "normalize_competition_id",
]
