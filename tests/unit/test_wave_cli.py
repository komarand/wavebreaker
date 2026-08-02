from __future__ import annotations

import argparse

import pytest

from kaggle_researcher.config import ConfigError, load_config
from kaggle_researcher.wave import build_parser, main


B5_CONFIG_ENV_VARS = (
    "TOP_K",
    "MAX_NOTEBOOKS",
    "MAX_PAPERS",
    "MAX_REPOS",
    "PDF_CACHE_DIR",
    "MAX_DISCUSSIONS",
    "WRITEUPS_PER_COMPETITION",
    "MAX_CONTEXT_TOKENS",
    "MAX_SAMPLE_SUB_BYTES",
    "META_KAGGLE_DIR",
    "RUN_BUDGET_TOKENS",
    "KAGGLE_API_TOKEN",
)


def test_top_level_help_works_without_model_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "facts" in help_text
    assert "brief" in help_text
    assert "journal" in help_text


@pytest.mark.parametrize("command", ["facts", "brief", "journal"])
def test_subcommand_help_works(
    command: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args([command, "--help"])

    assert exc_info.value.code == 0
    assert "slug" in capsys.readouterr().out


def test_facts_arguments_are_parsed() -> None:
    args = build_parser().parse_args(
        [
            "facts",
            "example",
            "--max-notebooks",
            "60",
            "--max-discussions",
            "200",
            "--writeups-per-competition",
            "7",
            "--similar",
            "comp-a,comp-b",
            "--out",
            "./output",
        ]
    )

    assert args == argparse.Namespace(
        command="facts",
        slug="example",
        max_notebooks=60,
        max_discussions=200,
        writeups_per_competition=7,
        similar="comp-a,comp-b",
        out="./output",
    )


def test_brief_and_journal_specific_arguments_are_parsed() -> None:
    brief_args = build_parser().parse_args(
        [
            "brief",
            "example",
            "--vram",
            "12",
            "--hours",
            "10.5",
            "--objective",
            "top_percent",
            "--facts-from",
            "./facts.json",
        ]
    )
    journal_args = build_parser().parse_args(
        [
            "journal",
            "example",
            "--used-validation",
            "GroupKFold(customer_id)",
            "--final-rank",
            "47",
            "--num-teams",
            "1200",
            "--brief-was-useful",
            "yes",
        ]
    )

    assert brief_args.vram == 12.0
    assert brief_args.hours == 10.5
    assert brief_args.objective == "top_percent"
    assert brief_args.facts_from == "./facts.json"
    assert journal_args.final_rank == 47
    assert journal_args.num_teams == 1200
    assert journal_args.brief_was_useful == "yes"


@pytest.mark.parametrize("command", ["brief", "journal"])
def test_unimplemented_subcommands_are_explicit_stubs(command: str) -> None:
    with pytest.raises(NotImplementedError, match=f"wave {command} is not implemented yet"):
        main([command, "example"])


def test_b5_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-key")
    for name in B5_CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    settings = load_config()

    assert settings.top_k == 10
    assert settings.max_notebooks == 20
    assert settings.max_papers == 15
    assert settings.max_repos == 10
    assert settings.pdf_cache_dir == "./data/pdfs"
    assert settings.max_discussions == 200
    assert settings.writeups_per_competition == 10
    assert settings.max_context_tokens == 120_000
    assert settings.max_sample_sub_bytes == 5_000_000
    assert settings.meta_kaggle_dir is None
    assert settings.run_budget_tokens is None
    assert settings.kaggle_api_token is None
    assert settings.kaggle_username is None
    assert settings.kaggle_key is None


def test_b5_config_env_overrides_and_legacy_kaggle_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-key")
    monkeypatch.setenv("TOP_K", "25")
    monkeypatch.setenv("MAX_NOTEBOOKS", "60")
    monkeypatch.setenv("MAX_PAPERS", "30")
    monkeypatch.setenv("MAX_REPOS", "18")
    monkeypatch.setenv("PDF_CACHE_DIR", "./custom-pdfs")
    monkeypatch.setenv("MAX_DISCUSSIONS", "300")
    monkeypatch.setenv("WRITEUPS_PER_COMPETITION", "12")
    monkeypatch.setenv("MAX_CONTEXT_TOKENS", "90000")
    monkeypatch.setenv("MAX_SAMPLE_SUB_BYTES", "4000000")
    monkeypatch.setenv("META_KAGGLE_DIR", "./meta-kaggle")
    monkeypatch.setenv("RUN_BUDGET_TOKENS", "150000")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "api-token")
    monkeypatch.setenv("KAGGLE_USERNAME", "legacy-user")
    monkeypatch.setenv("KAGGLE_KEY", "legacy-key")

    settings = load_config()

    assert settings.top_k == 25
    assert settings.max_notebooks == 60
    assert settings.max_papers == 30
    assert settings.max_repos == 18
    assert settings.pdf_cache_dir == "./custom-pdfs"
    assert settings.max_discussions == 300
    assert settings.writeups_per_competition == 12
    assert settings.max_context_tokens == 90_000
    assert settings.max_sample_sub_bytes == 4_000_000
    assert settings.meta_kaggle_dir == "./meta-kaggle"
    assert settings.run_budget_tokens == 150_000
    assert settings.kaggle_api_token == "api-token"
    assert settings.kaggle_username == "legacy-user"
    assert settings.kaggle_key == "legacy-key"


@pytest.mark.parametrize(
    "name",
    [
        "TOP_K",
        "MAX_DISCUSSIONS",
        "WRITEUPS_PER_COMPETITION",
        "RUN_BUDGET_TOKENS",
    ],
)
def test_new_integer_config_values_must_be_positive(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-key")
    monkeypatch.setenv(name, "0")

    with pytest.raises(ConfigError, match=f"{name} must be a positive integer"):
        load_config()


@pytest.mark.parametrize("value", ["0", "-1"])
def test_writeup_cli_limit_must_be_positive(value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["facts", "example", "--writeups-per-competition", value]
        )
