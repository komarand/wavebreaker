from __future__ import annotations

import re
import subprocess
import zipfile
from pathlib import Path
from urllib.parse import urlparse


class DatasetResolverError(RuntimeError):
    pass


class DatasetNotFoundError(DatasetResolverError):
    pass


SUPPORTED_DATA_EXTENSIONS = {".csv", ".parquet", ".json", ".jsonl"}


def derive_competition_slug(
    competition_id: str,
    competition_url: str | None = None,
) -> str:
    if competition_url:
        parsed_url = urlparse(competition_url)
        path_parts = [part for part in parsed_url.path.split("/") if part]
        for marker in ("competitions", "c"):
            if marker in path_parts:
                index = path_parts.index(marker)
                if index + 1 < len(path_parts):
                    return _slugify(path_parts[index + 1])
        if path_parts:
            return _slugify(path_parts[-1])

    return _slugify(competition_id)


def resolve_dataset(
    competition_id: str,
    competition_url: str | None,
    local_dataset_path: Path | None,
    download: bool,
    force_download: bool,
    cache_dir: Path,
) -> Path:
    if local_dataset_path is not None:
        return _resolve_local_dataset(local_dataset_path)

    competition_slug = derive_competition_slug(competition_id, competition_url)
    cached_dataset_path = Path(cache_dir) / competition_slug
    if cached_dataset_path.is_dir() and not force_download:
        if _has_supported_data_file(cached_dataset_path):
            return cached_dataset_path.resolve()
        _unzip_archives(cached_dataset_path)
        if _has_supported_data_file(cached_dataset_path):
            return cached_dataset_path.resolve()
        if not download:
            raise DatasetNotFoundError(
                f"Cached dataset at {cached_dataset_path} contains no supported data files "
                "(.csv, .parquet, .json, .jsonl), and dataset download is disabled."
            )

    if not download:
        raise DatasetNotFoundError(
            "Dataset was not found locally and dataset download is disabled "
            f"for competition {competition_slug!r}."
        )

    cached_dataset_path.mkdir(parents=True, exist_ok=True)
    _download_with_kaggle_cli(competition_slug, cached_dataset_path)
    _unzip_archives(cached_dataset_path)

    if not cached_dataset_path.is_dir():
        raise DatasetNotFoundError(f"Dataset download did not create {cached_dataset_path}")
    if not _has_supported_data_file(cached_dataset_path):
        archive_count = len(list(cached_dataset_path.glob("*.zip")))
        archive_hint = " after extracting archives" if archive_count else ""
        raise DatasetNotFoundError(
            f"Dataset cache at {cached_dataset_path} contains no supported data files"
            f"{archive_hint}. Expected one of: .csv, .parquet, .json, .jsonl."
        )

    return cached_dataset_path.resolve()


def _resolve_local_dataset(local_dataset_path: Path) -> Path:
    dataset_path = Path(local_dataset_path)
    if not dataset_path.exists():
        raise DatasetNotFoundError(f"Local dataset path does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise DatasetNotFoundError(f"Local dataset path is not a directory: {dataset_path}")
    return dataset_path.resolve()


def _download_with_kaggle_cli(competition_slug: str, destination_dir: Path) -> None:
    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        competition_slug,
        "-p",
        str(destination_dir),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise DatasetNotFoundError(
            "Kaggle CLI is not installed or is not available on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        message = _sanitize_error_message(exc.stderr or exc.stdout or str(exc))
        raise DatasetNotFoundError(f"Kaggle dataset download failed: {message}") from exc


def _unzip_archives(dataset_path: Path) -> None:
    for archive_path in sorted(dataset_path.glob("*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(dataset_path)


def _has_supported_data_file(dataset_path: Path) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in SUPPORTED_DATA_EXTENSIONS
        for path in dataset_path.rglob("*")
    )


def _sanitize_error_message(message: str) -> str:
    sanitized = re.sub(r"(?i)(KAGGLE_KEY=)[^\s]+", r"\1<redacted>", message)
    sanitized = re.sub(r"(?i)(key['\"]?\s*[:=]\s*)[^\s,;}]+", r"\1<redacted>", sanitized)
    return sanitized.strip()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return slug or "unknown-competition"


__all__ = [
    "DatasetNotFoundError",
    "DatasetResolverError",
    "derive_competition_slug",
    "resolve_dataset",
]
