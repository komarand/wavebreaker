from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from kaggle_researcher.contracts.errors import ContractError
from kaggle_researcher.contracts.final_strategy import FinalStrategyResult
from kaggle_researcher.contracts.final_strategy_protocol import (
    StrategyRenderingDraft,
    StrategySelectionDraft,
)
from kaggle_researcher.contracts.final_synthesis_diagnostics import (
    BridgeDiagnostic,
    FinalSynthesisDiagnostics,
    RenderingAttemptDiagnostic,
    SelectionAttemptDiagnostic,
)
from kaggle_researcher.contracts.ingest import (
    MigrationPolicy,
    RepairPolicy,
    ingest_contract,
)
from kaggle_researcher.contracts.registries import ContractRegistries
from kaggle_researcher.contracts.synthesis_context import FinalSynthesisContext
from kaggle_researcher.reasoning.final_strategy_bridge import (
    StrategyBridgeError,
    freeze_fallback_result,
    freeze_strategy_selection,
    skeleton_to_result,
    validate_skeleton_integrity,
    validate_rendering_draft,
)
from kaggle_researcher.reasoning.final_strategy_context import (
    build_final_strategy_selection_context,
)
from kaggle_researcher.reasoning.prompts.final_strategy_rendering_v2 import (
    FINAL_STRATEGY_RENDERING_PROMPT_VERSION,
    build_rendering_prompt,
    build_rendering_repair_prompt,
)
from kaggle_researcher.reasoning.prompts.final_strategy_selection_v2 import (
    FINAL_STRATEGY_SELECTION_PROMPT_VERSION,
    build_selection_prompt,
    build_selection_repair_prompt,
)


class TwoCallProtocolError(RuntimeError):
    pass


async def run_two_call_final_synthesis(
    *,
    context: FinalSynthesisContext,
    registries: ContractRegistries,
    client: Any,
    selection_model: str,
    rendering_model: str,
    diagnostics: FinalSynthesisDiagnostics,
    diagnostics_dir: Path | None,
    fallback_builder: Callable[[str], FinalStrategyResult],
) -> FinalStrategyResult:
    selection_context = build_final_strategy_selection_context(context, registries)
    selection_system, selection_user, selection_fp = build_selection_prompt(selection_context)
    diagnostics.protocol = "two_call"
    diagnostics.prompt_fingerprints["selection"] = selection_fp.model_dump(mode="json")

    selection_status = "llm_success"
    selection_raw: dict[str, Any] | None = None
    draft: StrategySelectionDraft | None = None
    skeleton = None
    selection_failure = ""
    max_repairs = _env_bounded_repair_count(
        "FINAL_SYNTHESIS_SELECTION_REPAIR_ATTEMPTS", 1
    )
    for attempt_index in range(max_repairs + 1):
        attempt_name = "initial" if attempt_index == 0 else "repair"
        attempt = SelectionAttemptDiagnostic(
            attempt=attempt_name,
            model=selection_model,
            prompt_version=FINAL_STRATEGY_SELECTION_PROMPT_VERSION,
            prompt_fingerprint=selection_fp.fingerprint,
        )
        diagnostics.selection_attempts.append(attempt)
        prompt = selection_user if attempt_index == 0 else build_selection_repair_prompt(
            invalid_draft=selection_raw or {},
            issues=diagnostics.selection_attempts[-2].issues,
            context=selection_context,
        )
        try:
            selection_raw = await client.chat_json(
                model=selection_model,
                system_prompt=selection_system,
                user_prompt=prompt,
            )
            attempt.response_hash = _hash(selection_raw)
            attempt.parse_succeeded = True
            draft = _ingest_llm_contract(
                selection_raw,
                expected_family="strategy_selection_draft",
            )
            assert isinstance(draft, StrategySelectionDraft)
            attempt.schema_succeeded = True
            selection_status = "llm_success" if attempt_index == 0 else "repaired_success"
            skeleton, bridge_payload = freeze_strategy_selection(
                draft,
                synthesis_context=context,
                selection_context=selection_context,
                selection_status=selection_status,
                selection_prompt_fingerprint=selection_fp,
            )
            attempt.reference_validation_succeeded = True
            diagnostics.bridge = BridgeDiagnostic.model_validate(bridge_payload)
            break
        except (ValidationError, StrategyBridgeError, ValueError) as exc:
            selection_failure = str(exc)
            attempt.issues = _issues(exc)
        except Exception as exc:
            selection_failure = f"provider failure: {type(exc).__name__}: {exc}"
            attempt.issues = [{"type": type(exc).__name__, "message": str(exc)[:500]}]
            diagnostics.provider_failures.append({
                "stage": "selection", "error_type": type(exc).__name__,
                "message": str(exc)[:500],
            })
            break

    if skeleton is None:
        diagnostics.selection_status = "degraded_fallback"
        diagnostics.fallback_required = True
        diagnostics.fallback_reason = selection_failure or "Call 1 remained invalid after bounded repair."
        fallback = fallback_builder(diagnostics.fallback_reason)
        skeleton = freeze_fallback_result(
            fallback,
            selection_prompt_fingerprint=selection_fp,
            warning=f"Call 1 used deterministic fallback: {diagnostics.fallback_reason}",
        )
    else:
        diagnostics.selection_status = selection_status
        diagnostics.initial_output_valid = selection_status == "llm_success"
        diagnostics.repair_attempted = selection_status == "repaired_success"
        diagnostics.repair_succeeded = selection_status == "repaired_success"

    validate_skeleton_integrity(skeleton)
    rendering_system, rendering_user, rendering_fp = build_rendering_prompt(
        skeleton,
        max_chars=_env_positive_int("FINAL_SYNTHESIS_RENDERING_MAX_CHARS", 50000),
    )
    diagnostics.prompt_fingerprints["rendering"] = rendering_fp.model_dump(mode="json")
    rendering: StrategyRenderingDraft | None = None
    rendering_status = "deterministic_render"
    rendering_warnings: list[str] = []
    max_render_repairs = _env_bounded_repair_count(
        "FINAL_SYNTHESIS_RENDERING_REPAIR_ATTEMPTS", 1
    )
    rendering_raw: dict[str, Any] | None = None
    rendering_enabled = skeleton.synthesis_selection_status in {
        "llm_success", "repaired_success"
    }
    for attempt_index in range(max_render_repairs + 1 if rendering_enabled else 0):
        attempt_name = "initial" if attempt_index == 0 else "repair"
        attempt = RenderingAttemptDiagnostic(
            attempt=attempt_name,
            model=rendering_model,
            prompt_version=FINAL_STRATEGY_RENDERING_PROMPT_VERSION,
            prompt_fingerprint=rendering_fp.fingerprint,
            skeleton_id=skeleton.skeleton_id,
            skeleton_hash=skeleton.skeleton_hash,
        )
        diagnostics.rendering_attempts.append(attempt)
        prompt = rendering_user if attempt_index == 0 else build_rendering_repair_prompt(
            invalid_draft=rendering_raw or {},
            issues=diagnostics.rendering_attempts[-2].issues,
            skeleton=skeleton,
        )
        try:
            rendering_raw = await client.chat_json(
                model=rendering_model,
                system_prompt=rendering_system,
                user_prompt=prompt,
            )
            attempt.response_hash = _hash(rendering_raw)
            attempt.parse_succeeded = True
            candidate = _ingest_llm_contract(
                rendering_raw,
                expected_family="strategy_rendering_draft",
            )
            assert isinstance(candidate, StrategyRenderingDraft)
            validate_skeleton_integrity(skeleton)
            validate_rendering_draft(candidate, skeleton)
            attempt.integrity_validation_succeeded = True
            rendering = candidate
            rendering_status = "llm_success" if attempt_index == 0 else "repaired_success"
            break
        except (ValidationError, StrategyBridgeError, ValueError) as exc:
            attempt.issues = _issues(exc)
        except Exception as exc:
            attempt.issues = [{"type": type(exc).__name__, "message": str(exc)[:500]}]
            diagnostics.provider_failures.append({
                "stage": "rendering", "error_type": type(exc).__name__,
                "message": str(exc)[:500],
            })
            break

    if rendering is None:
        if rendering_enabled:
            rendering_warnings.append(
                "Call 2 did not pass frozen-skeleton integrity validation; deterministic wording was retained."
            )
        else:
            rendering_warnings.append(
                "Call 2 was skipped because Call 1 required deterministic fallback rendering."
            )
    diagnostics.rendering_status = rendering_status
    diagnostics.fallback_required = skeleton.synthesis_selection_status == "degraded_fallback"
    result = skeleton_to_result(
        skeleton,
        rendering=rendering,
        rendering_status=rendering_status,
        rendering_prompt_fingerprint=rendering_fp,
        diagnostics_path=(
            str(diagnostics_dir / "final_synthesis_diagnostics.json")
            if diagnostics_dir is not None else None
        ),
        selection_model=selection_model,
        rendering_model=rendering_model,
        additional_warnings=rendering_warnings,
    )
    diagnostics.quality_metrics = result.quality_metrics.model_dump(mode="json")
    diagnostics.provenance_telemetry = {
        "source_links": len(result.source_to_hypothesis_links),
        "hypothesis_to_eda_links": len(result.hypothesis_to_eda_links),
        "actions_with_source_refs": result.quality_metrics.actions_with_source_refs,
    }
    return result


def _issues(exc: Exception) -> list[dict[str, Any]]:
    if isinstance(exc, ContractError):
        return [issue.as_dict() for issue in exc.issues] or [{
            "type": type(exc).__name__,
            "message": str(exc)[:500],
        }]
    if isinstance(exc, ValidationError):
        return [
            {
                "type": str(item.get("type")),
                "field_path": ".".join(map(str, item.get("loc") or ())),
                "message": str(item.get("msg"))[:500],
                "invalid_value_type": type(item.get("input")).__name__,
            }
            for item in exc.errors(include_url=False)
        ]
    return [{"type": type(exc).__name__, "message": str(exc)[:500]}]


def _hash(value: Any) -> str:
    import json
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _env_positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _env_nonnegative_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _env_bounded_repair_count(name: str, default: int) -> int:
    value = _env_nonnegative_int(name, default)
    if value > 1:
        raise ValueError(f"{name} must be 0 or 1 for the bounded repair protocol")
    return value


def _ingest_llm_contract(
    raw: Any,
    *,
    expected_family: str,
) -> StrategySelectionDraft | StrategyRenderingDraft:
    result = ingest_contract(
        raw,
        expected_family=expected_family,
        source_kind="llm_generated",
        migration_policy=MigrationPolicy.REQUIRE_CURRENT,
        repair_policy=RepairPolicy.FORBID,
    )
    contract = result.contract
    if not isinstance(contract, (StrategySelectionDraft, StrategyRenderingDraft)):
        raise TwoCallProtocolError(
            f"Ingest dispatched {type(contract).__name__}, expected {expected_family}"
        )
    return contract


__all__ = ["TwoCallProtocolError", "run_two_call_final_synthesis"]
