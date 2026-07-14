from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Contract tests are filesystem-only; any socket attempt is a test failure."""

    original_connect = socket.socket.connect

    def blocked(sock: socket.socket, address: object, *_args: object, **_kwargs: object) -> None:
        if isinstance(address, tuple) and address and address[0] in {
            "127.0.0.1", "::1", "localhost"
        }:
            original_connect(sock, address)
            return
        raise AssertionError("network access is forbidden in offline contract tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    try:
        import httpx

        monkeypatch.setattr(httpx.AsyncClient, "request", blocked)
        monkeypatch.setattr(httpx.Client, "request", blocked)
    except ImportError:
        pass
    try:
        import requests

        monkeypatch.setattr(requests.Session, "request", blocked)
    except ImportError:
        pass
