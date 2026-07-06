from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


ARTIFACT_SUBDIRS = (
    "artifacts",
    "artifacts/plots",
    "artifacts/profiles",
    "artifacts/baseline",
    "artifacts/drift",
    "artifacts/validation",
    "artifacts/samples",
)


class ArtifactWriterError(RuntimeError):
    pass


class ArtifactWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.run_dir: Path | None = None

    def create_run_dir(self, competition_id: str, timestamp: datetime | None = None) -> Path:
        timestamp = timestamp or datetime.now()
        run_name = f"{_slugify(competition_id)}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ARTIFACT_SUBDIRS:
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir.resolve()
        return self.run_dir

    def write_json(self, name: str, data: Any) -> Path:
        path = self._run_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_jsonable(data), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def write_markdown(self, name: str, text: str) -> Path:
        path = self._run_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def copy_input(self, path: Path, name: str) -> Path:
        source_path = Path(path)
        if not source_path.is_file():
            raise ArtifactWriterError(f"Input file does not exist: {source_path}")
        destination = self._run_path(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        return destination

    def artifact_path(self, *parts: str) -> Path:
        if not parts:
            return self._run_path("artifacts")
        return self._run_path(Path("artifacts", *parts))

    def _run_path(self, name: str | Path) -> Path:
        if self.run_dir is None:
            raise ArtifactWriterError("create_run_dir must be called before writing artifacts.")
        candidate = self.run_dir / Path(name)
        resolved = candidate.resolve()
        if not _is_relative_to(resolved, self.run_dir):
            raise ArtifactWriterError(f"Artifact path escapes run directory: {name}")
        return resolved


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return slug or "unknown-competition"
