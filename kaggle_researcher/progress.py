from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable, TypeVar

from tqdm.auto import tqdm


T = TypeVar("T")


@dataclass(frozen=True)
class ProgressConfig:
    enabled: bool = True
    leave_nested: bool = False
    min_interval: float = 0.1

    @property
    def disabled(self) -> bool:
        return not self.enabled or not sys.stdout.isatty() or bool(os.getenv("CI"))


def progress_iter(
    iterable: Iterable[T],
    *,
    desc: str,
    unit: str,
    total: int | None = None,
    level: int = 0,
    config: ProgressConfig,
) -> Iterable[T]:
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        position=level,
        leave=config.leave_nested if level else True,
        mininterval=config.min_interval,
        disable=config.disabled,
    )


def progress_write(message: str, *, config: ProgressConfig) -> None:
    if config.disabled:
        return
    tqdm.write(message)
