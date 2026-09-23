"""Thin wrapper around the Ollama client (chat, embeddings, health check)."""
from functools import lru_cache

import ollama

from .config import settings


@lru_cache(maxsize=1)
def client() -> ollama.Client:
    return ollama.Client(host=settings.ollama_host, timeout=settings.request_timeout)


def embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        resp = client().embed(model=settings.embed_model, input=texts[i : i + batch_size])
        vectors.extend(resp["embeddings"])
    return vectors


def chat(messages, tools=None):
    return client().chat(
        model=settings.chat_model,
        messages=messages,
        tools=tools,
        options={"temperature": settings.temperature},
    )


def _has_model(name: str, installed: list[str]) -> bool:
    if ":" in name:
        return name in installed
    return any(n.split(":")[0] == name for n in installed)


def health() -> dict:
    """Is Ollama reachable, and are the configured models pulled?"""
    try:
        resp = client().list()
        installed = [getattr(m, "model", None) or m["model"] for m in resp["models"]]
    except Exception as e:  # connection refused, timeout, ...
        return {"ok": False, "error": str(e), "installed": [], "missing": []}
    missing = [m for m in (settings.chat_model, settings.embed_model) if not _has_model(m, installed)]
    return {"ok": not missing, "error": None, "installed": installed, "missing": missing}
