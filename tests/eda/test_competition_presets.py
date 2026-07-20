from __future__ import annotations

from kaggle_researcher.eda.modules.file_inventory import build_file_inventory
from kaggle_researcher.eda.presets import (
    HOME_CREDIT_CRMS_PRESET,
    CompetitionPreset,
    PresetRegistry,
    get_preset,
)


def test_home_credit_preset_exposes_expected_hints() -> None:
    preset = HOME_CREDIT_CRMS_PRESET

    assert preset.preferred_id_columns == ("case_id",)
    assert preset.preferred_target_columns == ("target",)
    assert preset.preferred_time_columns == ("WEEK_NUM",)
    assert preset.preferred_prediction_columns == ("score",)
    assert preset.metric_aliases["stability_gini"] == "gini_stability"
    assert preset.known_validation_hints


def test_preset_registry_finds_home_credit_by_competition_or_preset_id() -> None:
    registry = PresetRegistry()

    assert registry.get(competition_id="home-credit-credit-risk-model-stability") is HOME_CREDIT_CRMS_PRESET
    assert registry.get(preset_id="home_credit_crms") is HOME_CREDIT_CRMS_PRESET
    assert get_preset("fixture_competition") is HOME_CREDIT_CRMS_PRESET
    assert registry.get(competition_id="unknown-competition") is None


def test_preset_metric_aliases_are_local_to_the_preset() -> None:
    preset = HOME_CREDIT_CRMS_PRESET
    empty_preset = CompetitionPreset(preset_id="empty")

    assert preset.canonical_metric_name("stability gini") == "gini_stability"
    assert empty_preset.canonical_metric_name("stability gini") == "stability gini"


def test_file_inventory_can_consume_table_name_patterns(tmp_path) -> None:
    (tmp_path / "train_main_0.csv").write_text("case_id,target\n1,0\n", encoding="utf-8")

    preset = CompetitionPreset(
        preset_id="custom",
        table_name_patterns={
            "train": ("train_main",),
            "depth_0": ("_0",),
        },
    )

    inventory = build_file_inventory(tmp_path, preset=preset)
    only_file = inventory.files[0]

    assert only_file.role_hint == "train"
    assert only_file.table_hint == "depth_0"
