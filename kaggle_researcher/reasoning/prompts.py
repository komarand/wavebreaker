from __future__ import annotations


SYSTEM_RULES = """Return JSON only.
Separate facts, hypotheses, and recommendations.
Include confidence.
Include evidence_ids.
Use retrieved_documents terminology.
Do not claim real train/test analysis was performed.
Do not confirm leakage based only on text sources.
Do not implement or imply data-execution features.
Do not include raw chain-of-thought."""
