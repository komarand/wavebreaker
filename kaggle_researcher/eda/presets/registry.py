from __future__ import annotations

from kaggle_researcher.eda.presets.base import CompetitionPreset
from kaggle_researcher.eda.presets.home_credit_crms import HOME_CREDIT_CRMS_PRESET


class PresetRegistry:
    def __init__(
        self,
        presets: tuple[CompetitionPreset, ...] | list[CompetitionPreset] | None = None,
    ) -> None:
        self._presets = tuple(presets or (HOME_CREDIT_CRMS_PRESET,))

    def get(
        self,
        *,
        competition_id: str | None = None,
        preset_id: str | None = None,
    ) -> CompetitionPreset | None:
        if preset_id is not None:
            normalized_preset_id = normalize_competition_id(preset_id)
            for preset in self._presets:
                if normalize_competition_id(preset.preset_id) == normalized_preset_id:
                    return preset
        if competition_id is not None:
            for preset in self._presets:
                if preset.matches_competition_id(competition_id):
                    return preset
        return None

    def all(self) -> tuple[CompetitionPreset, ...]:
        return self._presets


def get_preset(
    competition_id: str | None = None,
    preset_id: str | None = None,
) -> CompetitionPreset | None:
    return PresetRegistry().get(competition_id=competition_id, preset_id=preset_id)


def normalize_competition_id(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


__all__ = ["PresetRegistry", "get_preset", "normalize_competition_id"]
