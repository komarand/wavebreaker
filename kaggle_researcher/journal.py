from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JOURNAL_PATH = Path("journal") / "participation.jsonl"
RUNS_DIR = Path("runs")


def append_entry(**fields: Any) -> None:
    competition_id = fields.get("competition_id")
    if not isinstance(competition_id, str) or not competition_id.strip():
        raise ValueError("competition_id is required")

    brief_run_id = _optional_text(fields.get("brief_run_id"))
    used_validation = _optional_text(fields.get("used_validation"))
    recommended_validation = (
        _load_recommended_validation(brief_run_id)
        if brief_run_id is not None
        else None
    )
    final_rank = _optional_positive_integer(fields.get("final_rank"), "final_rank")
    num_teams = _optional_positive_integer(fields.get("num_teams"), "num_teams")
    percentile = _calculate_percentile(final_rank, num_teams)
    validation_matched = (
        used_validation.strip().casefold() == recommended_validation.strip().casefold()
        if used_validation is not None and recommended_validation is not None
        else None
    )
    recorded_at = fields.get("recorded_at")
    if recorded_at is None:
        recorded_at = datetime.now(timezone.utc).isoformat()
    elif isinstance(recorded_at, datetime):
        recorded_at = recorded_at.isoformat()
    elif not isinstance(recorded_at, str):
        raise ValueError("recorded_at must be an ISO timestamp string or datetime")

    brief_was_useful = fields.get("brief_was_useful")
    if brief_was_useful is not None and not isinstance(brief_was_useful, bool):
        raise ValueError("brief_was_useful must be a boolean or None")

    record = {
        "competition_id": competition_id,
        "recorded_at": recorded_at,
        "brief_run_id": brief_run_id,
        "used_validation": used_validation,
        "recommended_validation": recommended_validation,
        "validation_matched": validation_matched,
        "final_rank": final_rank,
        "num_teams": num_teams,
        "percentile": percentile,
        "brief_was_useful": brief_was_useful,
        "notes": _optional_text(fields.get("notes")),
    }
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL_PATH.open("a", encoding="utf-8", newline="\n") as journal_file:
        journal_file.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        )


def load_entries() -> list[dict[str, Any]]:
    if not JOURNAL_PATH.exists():
        return []
    entries: list[dict[str, Any]] = []
    with JOURNAL_PATH.open("r", encoding="utf-8") as journal_file:
        for line_number, line in enumerate(journal_file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"journal line {line_number} is not a JSON object")
            entries.append(payload)
    return entries


def summarize() -> dict[str, int | float]:
    entries = load_entries()
    percentiles = [
        float(entry["percentile"])
        for entry in entries
        if entry.get("percentile") is not None
    ]
    return {
        "entries": len(entries),
        "briefs_useful": sum(
            entry.get("brief_was_useful") is True for entry in entries
        ),
        "validation_matches": sum(
            entry.get("validation_matched") is True for entry in entries
        ),
        "median_percentile": statistics.median(percentiles) if percentiles else 0,
    }


def _load_recommended_validation(brief_run_id: str) -> str | None:
    brief_path = _brief_path(brief_run_id)
    payload = json.loads(brief_path.read_text(encoding="utf-8"))
    direct_value = _optional_text(payload.get("recommended_validation"))
    if direct_value is not None:
        return direct_value

    validation_claims = payload.get("validation")
    if not isinstance(validation_claims, list):
        return None
    for claim in validation_claims:
        if isinstance(claim, dict):
            claim_text = _optional_text(claim.get("text"))
            if claim_text is not None:
                return claim_text
    return None


def _brief_path(brief_run_id: str) -> Path:
    run_path = Path(brief_run_id)
    if run_path.is_absolute():
        return run_path if run_path.suffix == ".json" else run_path / "brief.json"
    return RUNS_DIR / run_path / "brief.json"


def _calculate_percentile(
    final_rank: int | None,
    num_teams: int | None,
) -> float | None:
    if final_rank is None or num_teams is None:
        return None
    if final_rank > num_teams:
        raise ValueError("final_rank cannot exceed num_teams")
    return final_rank / num_teams * 100


def _optional_positive_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional text fields must be strings or None")
    return value if value else None
