"""ChromaDB-backed vector store. Embeddings come from Ollama."""
from __future__ import annotations

import hashlib
import time

import chromadb
from chromadb.config import Settings as ChromaSettings

from . import llm
from .config import settings
from .ingest import chunk_text


class VectorStore:
    def __init__(self, path: str | None = None, collection: str | None = None):
        self.client = chromadb.PersistentClient(
            path=path or settings.chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.col = self.client.get_or_create_collection(
            collection or settings.collection, metadata={"hnsw:space": "cosine"}
        )

    # ---------- write ----------
    def add_document(self, source: str, chunks: list[str], doc_type: str = "txt") -> int:
        self.delete_source(source)  # re-ingesting a file replaces its old chunks
        prefix = hashlib.sha1(source.encode()).hexdigest()[:12]
        now = time.time()
        self.col.upsert(
            ids=[f"{prefix}-{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=llm.embed(chunks),
            metadatas=[
                {"source": source, "chunk": i, "type": doc_type, "ingested_at": now}
                for i in range(len(chunks))
            ],
        )
        return len(chunks)

    def add_note(self, title: str, text: str) -> int:
        return self.add_document(f"note: {title}", chunk_text(text), doc_type="note")

    def delete_source(self, source: str) -> None:
        self.col.delete(where={"source": source})

    # ---------- read ----------
    def count(self) -> int:
        return self.col.count()

    def search(self, query: str, k: int | None = None) -> list[dict]:
        total = self.count()
        if total == 0 or not query.strip():
            return []
        res = self.col.query(
            query_embeddings=llm.embed([query]),
            n_results=min(k or settings.top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for id_, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(
                {
                    "id": id_,
                    "text": doc,
                    "source": meta.get("source", "?"),
                    "chunk": meta.get("chunk", 0),
                    "score": round(1.0 - float(dist), 4),  # cosine similarity
                }
            )
        return hits

    def list_sources(self) -> list[dict]:
        data = self.col.get(include=["metadatas"])
        agg: dict[str, dict] = {}
        for m in data.get("metadatas") or []:
            src = m.get("source", "?")
            row = agg.setdefault(
                src,
                {"source": src, "chunks": 0, "type": m.get("type", ""), "ingested_at": m.get("ingested_at", 0)},
            )
            row["chunks"] += 1
        return sorted(agg.values(), key=lambda r: r["ingested_at"], reverse=True)
