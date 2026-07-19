import json

from kaggle_researcher.source_registry.schemas import CachePolicy, CacheRunTelemetry
from kaggle_researcher.source_registry.telemetry import write_cache_report


def test_report_contains_counts_but_no_embeddings_or_content(tmp_path) -> None:
    telemetry = CacheRunTelemetry(run_id="r", competition_id="c", summaries_reused=1, embeddings_computed=2)
    path = write_cache_report(tmp_path / "source_cache_report.json", telemetry, CachePolicy())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["processing"]["summaries_reused"] == 1
    assert "embedding" not in json.dumps(payload["per_source"])
    assert "content" not in json.dumps(payload["per_source"])
