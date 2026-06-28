from __future__ import annotations

from types import SimpleNamespace

import pytest

from kaggle_researcher import embedder


class FakeSentenceTransformer:
    instances: list[FakeSentenceTransformer] = []

    def __init__(self, model_name: str, device: str, model_kwargs: dict | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.encode_calls: list[dict] = []
        FakeSentenceTransformer.instances.append(self)

    def encode(self, texts: list[str], **kwargs) -> list[list[float]]:
        self.encode_calls.append({"texts": texts, "kwargs": kwargs})
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]


@pytest.fixture(autouse=True)
def reset_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSentenceTransformer.instances = []
    monkeypatch.setattr(embedder, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(
        embedder,
        "torch",
        SimpleNamespace(
            bfloat16="bf16",
            cuda=SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False),
        ),
    )
    monkeypatch.setattr(embedder, "_model", None)
    monkeypatch.setattr(embedder, "_embedding_dim", None)
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    monkeypatch.delenv("MAX_EMBED_BATCH_SIZE", raising=False)


def test_embed_texts_preserves_order_and_batches() -> None:
    result = embedder.embed_texts(["a", "bb", "ccc"], batch_size=2)

    assert result == [[0.0, 1.0], [1.0, 2.0], [0.0, 3.0]]
    model = FakeSentenceTransformer.instances[0]
    assert [call["texts"] for call in model.encode_calls] == [["a", "bb"], ["ccc"]]
    assert all(call["kwargs"]["normalize_embeddings"] is True for call in model.encode_calls)


def test_embed_one_returns_single_embedding() -> None:
    result = embedder.embed_one("query")

    assert result == [0.0, 5.0]


def test_empty_list_returns_empty_list() -> None:
    assert embedder.embed_texts([]) == []
    assert FakeSentenceTransformer.instances == []


def test_get_embedding_dim_returns_detected_dimension() -> None:
    assert embedder.get_embedding_dim() == 2


def test_model_uses_cuda_bf16_when_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        embedder,
        "torch",
        SimpleNamespace(
            bfloat16="bf16",
            cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
        ),
    )

    embedder.embed_one("query")

    model = FakeSentenceTransformer.instances[0]
    assert model.device == "cuda"
    assert model.model_kwargs == {"torch_dtype": "bf16"}
