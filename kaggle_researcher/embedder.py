from __future__ import annotations

import logging
import os
from typing import Any

from kaggle_researcher.config import DEFAULT_EMBED_MODEL, DEFAULT_MAX_EMBED_BATCH_SIZE

logger = logging.getLogger(__name__)

SentenceTransformer: Any | None = None
torch: Any | None = None
_model: Any | None = None
_embedding_dim: int | None = None


class EmbedderError(RuntimeError):
    """Raised when local embedding generation fails."""


def embed_texts(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    if batch_size is None:
        batch_size = _get_batch_size()

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if not texts:
        return []

    model = _get_model()
    embeddings: list[list[float]] = []

    logger.info("Embedding %s texts in batches of %s", len(texts), batch_size)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = model.encode(
            batch,
            batch_size=len(batch),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        batch_embeddings = _to_python_matrix(encoded)
        if len(batch_embeddings) != len(batch):
            raise EmbedderError(
                "SentenceTransformer returned a different number of embeddings than inputs"
            )
        embeddings.extend(batch_embeddings)

    return embeddings


def embed_one(text: str) -> list[float]:
    embeddings = embed_texts([text], batch_size=1)
    return embeddings[0]


def get_embedding_dim() -> int:
    global _embedding_dim

    if _embedding_dim is None:
        _embedding_dim = len(embed_one("embedding dimension probe"))
        logger.info("Detected embedding dimension: %s", _embedding_dim)

    return _embedding_dim


def _get_model() -> Any:
    global _model

    if _model is None:
        model_name = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
        device = _get_device()
        model_kwargs = _get_model_kwargs(device)
        sentence_transformer = _get_sentence_transformer_class()

        logger.info("Loading embedding model %s on %s", model_name, device)
        try:
            _model = sentence_transformer(
                model_name,
                device=device,
                model_kwargs=model_kwargs,
            )
        except TypeError:
            _model = sentence_transformer(model_name, device=device)
        except Exception:
            if not model_kwargs:
                raise
            logger.warning("Could not load embedding model with bf16; retrying without bf16")
            _model = sentence_transformer(model_name, device=device)

    return _model


def _get_device() -> str:
    torch_module = _get_torch_module()
    try:
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    except AttributeError:
        return "cpu"


def _get_model_kwargs(device: str) -> dict[str, Any]:
    if device != "cuda":
        return {}

    torch_module = _get_torch_module()
    try:
        if not torch_module.cuda.is_bf16_supported():
            return {}
    except AttributeError:
        return {}

    logger.info("CUDA bf16 is supported; loading embedding model with bfloat16")
    return {"torch_dtype": torch_module.bfloat16}


def _get_sentence_transformer_class() -> Any:
    global SentenceTransformer

    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as imported_sentence_transformer
        except ImportError as exc:
            raise EmbedderError(
                "sentence-transformers is required for local embeddings. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        SentenceTransformer = imported_sentence_transformer

    return SentenceTransformer


def _get_torch_module() -> Any:
    global torch

    if torch is None:
        try:
            import torch as imported_torch
        except ImportError as exc:
            raise EmbedderError(
                "torch is required for local embeddings. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc
        torch = imported_torch

    return torch


def _get_batch_size() -> int:
    raw_value = os.getenv("MAX_EMBED_BATCH_SIZE")
    if raw_value is None:
        return DEFAULT_MAX_EMBED_BATCH_SIZE

    try:
        batch_size = int(raw_value)
    except ValueError as exc:
        raise EmbedderError("MAX_EMBED_BATCH_SIZE must be a positive integer") from exc

    if batch_size <= 0:
        raise EmbedderError("MAX_EMBED_BATCH_SIZE must be a positive integer")

    return batch_size


def _to_python_matrix(value: Any) -> list[list[float]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().float().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, list):
        raise EmbedderError("SentenceTransformer returned embeddings in an unsupported format")

    if not value:
        return []

    if _is_number(value[0]):
        return [[float(item) for item in value]]

    matrix: list[list[float]] = []
    for row in value:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, list):
            raise EmbedderError("SentenceTransformer returned a malformed embedding row")
        matrix.append([float(item) for item in row])

    return matrix


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
