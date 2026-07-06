from __future__ import annotations


def map_at_k(*args: object, **kwargs: object) -> float:
    raise NotImplementedError("MAP@K requires query-group evaluation and is not implemented yet.")


def ndcg(*args: object, **kwargs: object) -> float:
    raise NotImplementedError("NDCG requires query-group evaluation and is not implemented yet.")


def recall_at_k(*args: object, **kwargs: object) -> float:
    raise NotImplementedError(
        "Recall@K requires query-group evaluation and is not implemented yet."
    )


__all__ = ["map_at_k", "ndcg", "recall_at_k"]
