from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".json", ".jsonl"}


class ReaderError(RuntimeError):
    pass


class DatasetReader:
    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = Path(dataset_path).resolve()
        if not self.dataset_path.exists():
            raise ReaderError(f"Dataset path does not exist: {self.dataset_path}")
        if not self.dataset_path.is_dir():
            raise ReaderError(f"Dataset path is not a directory: {self.dataset_path}")

    def resolve_path(self, relative_path: str | Path) -> Path:
        candidate = (self.dataset_path / Path(relative_path)).resolve()
        if not _is_relative_to(candidate, self.dataset_path):
            raise ReaderError(f"Path escapes dataset directory: {relative_path}")
        if not candidate.is_file():
            raise ReaderError(f"Dataset file does not exist: {relative_path}")
        if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ReaderError(f"Unsupported dataset file extension: {candidate.suffix}")
        return candidate

    def read_schema(self, relative_path: str | Path) -> list[dict[str, str]]:
        path = self.resolve_path(relative_path)
        try:
            schema = _read_schema(path)
        except Exception as exc:
            raise ReaderError(f"Could not read schema for {relative_path}: {exc}") from exc
        return [{"name": name, "dtype": str(dtype)} for name, dtype in schema.items()]

    def count_rows(self, relative_path: str | Path) -> int | None:
        path = self.resolve_path(relative_path)
        try:
            if path.suffix.lower() == ".json":
                return pl.read_json(path).height
            return int(_scan_lazy(path).select(pl.len()).collect().item())
        except Exception as exc:
            raise ReaderError(f"Could not count rows for {relative_path}: {exc}") from exc

    def sample_table(
        self,
        relative_path: str | Path,
        n_rows: int,
        seed: int = 42,
    ) -> pl.DataFrame:
        if n_rows <= 0:
            raise ReaderError("n_rows must be a positive integer")
        path = self.resolve_path(relative_path)
        try:
            frame = _read_bounded(path, n_rows=n_rows)
            if frame.height <= n_rows:
                return frame
            return frame.sample(n=n_rows, seed=seed)
        except Exception as exc:
            raise ReaderError(f"Could not sample table {relative_path}: {exc}") from exc

    def read_columns(
        self,
        relative_path: str | Path,
        columns: list[str],
        n_rows: int | None = None,
    ) -> pl.DataFrame:
        if not columns:
            raise ReaderError("columns must contain at least one column name")
        if n_rows is not None and n_rows <= 0:
            raise ReaderError("n_rows must be a positive integer")
        path = self.resolve_path(relative_path)
        try:
            if path.suffix.lower() == ".json":
                frame = pl.read_json(path).select(columns)
                return frame.head(n_rows) if n_rows is not None else frame

            lazy_frame = _scan_lazy(path).select(columns)
            if n_rows is not None:
                lazy_frame = lazy_frame.head(n_rows)
            return lazy_frame.collect()
        except Exception as exc:
            raise ReaderError(f"Could not read columns from {relative_path}: {exc}") from exc

    def file_head(self, relative_path: str | Path, n_rows: int = 5) -> pl.DataFrame:
        if n_rows <= 0:
            raise ReaderError("n_rows must be a positive integer")
        path = self.resolve_path(relative_path)
        try:
            return _read_bounded(path, n_rows=n_rows)
        except Exception as exc:
            raise ReaderError(f"Could not read file head for {relative_path}: {exc}") from exc


def _read_schema(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return dict(pl.read_json(path, n_rows=1).schema)
    lazy_frame = _scan_lazy(path)
    try:
        return dict(lazy_frame.collect_schema())
    except AttributeError:
        return dict(lazy_frame.schema)


def _read_bounded(path: Path, n_rows: int) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return pl.read_json(path).head(n_rows)
    return _scan_lazy(path).head(n_rows).collect()


def _scan_lazy(path: Path) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.scan_csv(path)
    if suffix == ".parquet":
        return pl.scan_parquet(path)
    if suffix == ".jsonl":
        return pl.scan_ndjson(path)
    raise ReaderError(f"Unsupported dataset file extension: {suffix}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


__all__ = ["DatasetReader", "ReaderError"]
