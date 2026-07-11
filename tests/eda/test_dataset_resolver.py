from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from kaggle_researcher.eda.io import dataset_resolver
from kaggle_researcher.eda.io.dataset_resolver import (
    DatasetNotFoundError,
    derive_competition_slug,
    resolve_dataset,
)


def test_derive_competition_slug_prefers_competition_url() -> None:
    assert (
        derive_competition_slug(
            competition_id="fallback-id",
            competition_url="https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability",
        )
        == "home-credit-credit-risk-model-stability"
    )
    assert (
        derive_competition_slug(
            competition_id="Fallback ID",
            competition_url="https://www.kaggle.com/c/titanic/overview",
        )
        == "titanic"
    )
    assert derive_competition_slug("Messy Competition ID!") == "messy-competition-id"


def test_local_dataset_path_returns_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_dataset = tmp_path / "local_dataset"
    local_dataset.mkdir()

    def fail_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("download helper should not be called")

    monkeypatch.setattr(dataset_resolver, "_download_with_kaggle_cli", fail_download)

    resolved = resolve_dataset(
        competition_id="fixture",
        competition_url=None,
        local_dataset_path=local_dataset,
        download=True,
        force_download=True,
        cache_dir=tmp_path / "cache",
    )

    assert resolved == local_dataset.resolve()


def test_invalid_local_dataset_path_raises_clear_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing"

    with pytest.raises(DatasetNotFoundError, match="does not exist"):
        resolve_dataset(
            competition_id="fixture",
            competition_url=None,
            local_dataset_path=missing_path,
            download=False,
            force_download=False,
            cache_dir=tmp_path / "cache",
        )

    file_path = tmp_path / "dataset.csv"
    file_path.write_text("id,target\n1,0\n", encoding="utf-8")

    with pytest.raises(DatasetNotFoundError, match="not a directory"):
        resolve_dataset(
            competition_id="fixture",
            competition_url=None,
            local_dataset_path=file_path,
            download=False,
            force_download=False,
            cache_dir=tmp_path / "cache",
        )


def test_empty_cached_dataset_path_redownloads_when_download_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached_dataset = tmp_path / "cache" / "fixture-competition"
    cached_dataset.mkdir(parents=True)
    calls: list[str] = []

    def fake_download(competition_slug: str, destination_dir: Path) -> None:
        calls.append(competition_slug)
        (destination_dir / "train.csv").write_text("id,target\n1,0\n", encoding="utf-8")

    monkeypatch.setattr(dataset_resolver, "_download_with_kaggle_cli", fake_download)

    resolved = resolve_dataset(
        competition_id="Fixture Competition",
        competition_url=None,
        local_dataset_path=None,
        download=True,
        force_download=False,
        cache_dir=tmp_path / "cache",
    )

    assert resolved == cached_dataset.resolve()
    assert calls == ["fixture-competition"]
    assert (resolved / "train.csv").is_file()


def test_cached_dataset_with_supported_file_returns_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached_dataset = tmp_path / "cache" / "fixture-competition"
    cached_dataset.mkdir(parents=True)
    (cached_dataset / "train.csv").write_text("id,target\n1,0\n", encoding="utf-8")

    def fail_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("download helper should not be called")

    monkeypatch.setattr(dataset_resolver, "_download_with_kaggle_cli", fail_download)

    resolved = resolve_dataset(
        competition_id="Fixture Competition",
        competition_url=None,
        local_dataset_path=None,
        download=True,
        force_download=False,
        cache_dir=tmp_path / "cache",
    )

    assert resolved == cached_dataset.resolve()


def test_missing_dataset_with_download_disabled_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError, match="download is disabled"):
        resolve_dataset(
            competition_id="fixture",
            competition_url=None,
            local_dataset_path=None,
            download=False,
            force_download=False,
            cache_dir=tmp_path / "cache",
        )


def test_download_path_calls_mocked_helper_and_unzips_archives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_download(competition_slug: str, destination_dir: Path) -> None:
        calls.append((competition_slug, destination_dir))
        with zipfile.ZipFile(destination_dir / "competition.zip", "w") as archive:
            archive.writestr("train.csv", "id,target\n1,0\n")

    monkeypatch.setattr(dataset_resolver, "_download_with_kaggle_cli", fake_download)

    resolved = resolve_dataset(
        competition_id="fallback",
        competition_url="https://www.kaggle.com/competitions/fixture-competition",
        local_dataset_path=None,
        download=True,
        force_download=False,
        cache_dir=tmp_path / "cache",
    )

    assert calls == [("fixture-competition", (tmp_path / "cache" / "fixture-competition"))]
    assert resolved == (tmp_path / "cache" / "fixture-competition").resolve()
    assert (resolved / "competition.zip").is_file()
    assert (resolved / "train.csv").read_text(encoding="utf-8") == "id,target\n1,0\n"


def test_force_download_ignores_existing_cache_when_download_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached_dataset = tmp_path / "cache" / "fixture"
    cached_dataset.mkdir(parents=True)
    calls: list[str] = []

    def fake_download(competition_slug: str, destination_dir: Path) -> None:
        calls.append(competition_slug)
        (destination_dir / "fresh.csv").write_text("id\n1\n", encoding="utf-8")

    monkeypatch.setattr(dataset_resolver, "_download_with_kaggle_cli", fake_download)

    resolved = resolve_dataset(
        competition_id="fixture",
        competition_url=None,
        local_dataset_path=None,
        download=True,
        force_download=True,
        cache_dir=tmp_path / "cache",
    )

    assert resolved == cached_dataset.resolve()
    assert calls == ["fixture"]
    assert (resolved / "fresh.csv").is_file()
