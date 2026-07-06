from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from kaggle_researcher.eda.io.dataset_reader import DatasetReader, ReaderError
from kaggle_researcher.eda.presets import CompetitionPreset
from kaggle_researcher.eda.schemas import DatasetFile, FileInventoryResult


DETECTED_EXTENSIONS = {".csv", ".parquet", ".json", ".jsonl", ".zip"}
READABLE_EXTENSIONS = {".csv", ".parquet", ".json", ".jsonl"}


def build_file_inventory(
    dataset_path: Path,
    preset: CompetitionPreset | None = None,
) -> FileInventoryResult:
    root = Path(dataset_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")

    reader = DatasetReader(root)
    files: list[DatasetFile] = []
    warnings: list[str] = []

    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative_path = path.relative_to(root).as_posix()
        extension = path.suffix.lower()
        role_hint = _detect_role_hint(path.name, preset=preset)
        table_hint = _detect_table_hint(path.name, role_hint, preset=preset)
        can_read, read_error = _check_readability(reader, relative_path, extension)
        if read_error is not None:
            warnings.append(f"{relative_path}: {read_error}")

        size_bytes = path.stat().st_size
        files.append(
            DatasetFile(
                path=relative_path,
                name=path.name,
                extension=extension,
                size_bytes=size_bytes,
                size_mb=round(size_bytes / (1024 * 1024), 6),
                role_hint=role_hint,
                table_hint=table_hint,
                can_read=can_read,
                read_error=read_error,
            )
        )

    detected_formats = dict(sorted(Counter(file.extension for file in files).items()))
    table_roles = {file.path: file.role_hint for file in files}
    train_files = [file.path for file in files if file.role_hint == "train"]
    test_files = [file.path for file in files if file.role_hint == "test"]
    sample_submission_files = [
        file.path for file in files if file.role_hint == "sample_submission"
    ]
    metadata_files = [file.path for file in files if file.role_hint == "metadata"]

    duplicate_format_pairs = _detect_duplicate_format_pairs(files)
    missing_train_test_pairs = _detect_missing_train_test_pairs(files)
    suspicious_files = [
        {"path": file.path, "reason": file.read_error}
        for file in files
        if file.read_error is not None
    ]

    return FileInventoryResult(
        dataset_path=str(root),
        files=files,
        detected_formats=detected_formats,
        table_roles=table_roles,
        train_files=train_files,
        test_files=test_files,
        sample_submission_files=sample_submission_files,
        metadata_files=metadata_files,
        missing_train_test_pairs=missing_train_test_pairs,
        duplicate_format_pairs=duplicate_format_pairs,
        suspicious_files=suspicious_files,
        warnings=warnings,
    )


def _check_readability(
    reader: DatasetReader,
    relative_path: str,
    extension: str,
) -> tuple[bool, str | None]:
    if extension == ".zip":
        return True, None
    if extension not in READABLE_EXTENSIONS:
        return False, f"Unsupported dataset file extension: {extension}"
    try:
        reader.read_schema(relative_path)
    except ReaderError as exc:
        return False, str(exc)
    return True, None


def _detect_role_hint(filename: str, *, preset: CompetitionPreset | None = None) -> str:
    stem = Path(filename).stem.lower()
    if preset is not None and _matches_patterns(stem, preset, "sample_submission"):
        return "sample_submission"
    if preset is not None and _matches_patterns(stem, preset, "train"):
        return "train"
    if preset is not None and _matches_patterns(stem, preset, "test"):
        return "test"
    if "sample_submission" in stem or stem in {"submission", "sample"}:
        return "sample_submission"
    if stem.startswith("train") or "_train" in stem or stem.endswith("_train"):
        return "train"
    if stem.startswith("test") or "_test" in stem or stem.endswith("_test"):
        return "test"
    if any(token in stem for token in ("metadata", "meta", "description", "readme")):
        return "metadata"
    return "unknown"


def _detect_table_hint(
    filename: str,
    role_hint: str,
    *,
    preset: CompetitionPreset | None = None,
) -> str:
    stem = Path(filename).stem.lower()
    if role_hint == "sample_submission":
        return "submission"
    if preset is not None:
        for table_hint in ("base", "depth_0", "depth_1", "depth_2"):
            if _matches_patterns(stem, preset, table_hint):
                return table_hint
    if "base" in stem:
        return "base"
    if stem.endswith("_0") or "depth_0" in stem:
        return "depth_0"
    if stem.endswith("_1") or "depth_1" in stem:
        return "depth_1"
    if stem.endswith("_2") or "depth_2" in stem:
        return "depth_2"
    if role_hint in {"train", "test"}:
        return "secondary"
    return "unknown"


def _detect_duplicate_format_pairs(files: list[DatasetFile]) -> list[dict[str, object]]:
    by_logical_name: dict[str, list[DatasetFile]] = defaultdict(list)
    for file in files:
        if file.extension in {".csv", ".parquet"}:
            by_logical_name[_logical_table_name(file.name)].append(file)

    duplicate_pairs: list[dict[str, object]] = []
    for logical_name, grouped_files in sorted(by_logical_name.items()):
        extensions = {file.extension for file in grouped_files}
        if {".csv", ".parquet"}.issubset(extensions):
            duplicate_pairs.append(
                {
                    "logical_table": logical_name,
                    "paths": [file.path for file in grouped_files],
                    "formats": sorted(extensions),
                }
            )
    return duplicate_pairs


def _detect_missing_train_test_pairs(files: list[DatasetFile]) -> list[dict[str, str]]:
    roles_by_logical_name: dict[str, set[str]] = defaultdict(set)
    for file in files:
        if file.role_hint in {"train", "test"}:
            roles_by_logical_name[_logical_train_test_pair_name(file.name)].add(file.role_hint)

    missing_pairs: list[dict[str, str]] = []
    for logical_name, roles in sorted(roles_by_logical_name.items()):
        if "train" not in roles:
            missing_pairs.append({"logical_table": logical_name, "missing": "train"})
        if "test" not in roles:
            missing_pairs.append({"logical_table": logical_name, "missing": "test"})
    return missing_pairs


def _logical_table_name(filename: str) -> str:
    return Path(filename).stem.lower()


def _logical_train_test_pair_name(filename: str) -> str:
    stem = Path(filename).stem.lower()
    for prefix in ("train_", "test_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    for suffix in ("_train", "_test"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _matches_patterns(stem: str, preset: CompetitionPreset, pattern_name: str) -> bool:
    return any(pattern.lower() in stem for pattern in preset.table_name_patterns.get(pattern_name, ()))


__all__ = ["build_file_inventory"]
