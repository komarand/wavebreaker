from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from kaggle_researcher.contracts.registry import CONTRACT_REGISTRY, ContractRegistry


DEFAULT_SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "contracts" / "schemas"
_OUTPUT_FAMILY_NAMES = {"final_strategy": "final_strategy_result"}


class SchemaExportError(RuntimeError):
    pass


def schema_output_path(root: Path, family: str, version: str) -> Path:
    directory = _OUTPUT_FAMILY_NAMES.get(family, family)
    return Path(root) / directory / f"{version}.json"


def build_public_schemas(
    registry: ContractRegistry = CONTRACT_REGISTRY,
) -> dict[Path, dict[str, Any]]:
    """Build stable schemas for every registered public family/version."""
    outputs: dict[Path, dict[str, Any]] = {}
    owners: dict[Path, tuple[str, str]] = {}
    for family, version in registry:
        relative = schema_output_path(Path(), family, version)
        if relative in outputs:
            prior = owners[relative]
            raise SchemaExportError(
                f"Duplicate schema output {relative.as_posix()!r} for "
                f"{prior[0]}@{prior[1]} and {family}@{version}"
            )
        schema = registry[(family, version)].model_json_schema(mode="validation")
        _pin_contract_header(schema, family=family, version=version)
        outputs[relative] = schema
        owners[relative] = (family, version)
    return dict(sorted(outputs.items(), key=lambda item: item[0].as_posix()))


def export_schemas(
    output_dir: Path = DEFAULT_SCHEMA_ROOT,
    *,
    registry: ContractRegistry = CONTRACT_REGISTRY,
) -> tuple[Path, ...]:
    output_dir = Path(output_dir)
    rendered = build_public_schemas(registry)
    written: list[Path] = []
    for relative, schema in rendered.items():
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written.append(destination)
    return tuple(written)


def validate_registered_fixture(
    family: str,
    version: str,
    fixture: Mapping[str, Any],
    *,
    registry: ContractRegistry = CONTRACT_REGISTRY,
) -> None:
    """Validate fixture data with the same public model that emitted its schema."""
    payload = dict(fixture)
    if payload.get("contract_family") != family or payload.get("schema_version") != version:
        raise SchemaExportError(f"Fixture header does not select {family}@{version}")
    registry.resolve(family, version).model_validate(payload)


def _pin_contract_header(schema: dict[str, Any], *, family: str, version: str) -> None:
    properties = schema.setdefault("properties", {})
    for name, value in (("contract_family", family), ("schema_version", version)):
        field_schema = properties.setdefault(name, {})
        field_schema.pop("enum", None)
        field_schema["const"] = value
        field_schema["default"] = value
    required = list(schema.get("required", []))
    for name in ("contract_family", "schema_version"):
        if name not in required:
            required.append(name)
    schema["required"] = sorted(required)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export deterministic JSON Schemas for registered public contracts."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SCHEMA_ROOT)
    args = parser.parse_args(argv)
    paths = export_schemas(args.output_dir)
    print(f"Exported {len(paths)} contract schemas to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
