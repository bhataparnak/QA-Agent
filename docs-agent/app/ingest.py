"""Text extraction and chunking for uploaded documents."""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

from .config import settings

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".markdown", ".csv", ".json", ".html", ".htm"}


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if ext == ".docx":
        import docx

        doc = docx.Document(io.BytesIO(data))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if ext in {".html", ".htm"}:
        html = data.decode("utf-8", errors="ignore")
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
        html = re.sub(r"<br\s*/?>|</p>|</div>|</h\d>|</li>", "\n\n", html, flags=re.I)
        return re.sub(r"[ \t]+", " ", re.sub(r"<[^>]+>", " ", html))
    return data.decode("utf-8", errors="ignore")


def _units(text: str, size: int):
    """Yield paragraphs; split oversized ones into sentences, then hard-cut."""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            yield para
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            while len(sent) > size:
                yield sent[:size]
                sent = sent[size:]
            if sent:
                yield sent


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    """Greedy paragraph-aware chunking with a character overlap between chunks."""
    size = size or settings.chunk_size
    overlap = settings.chunk_overlap if overlap is None else overlap
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()

    chunks: list[str] = []
    current = ""
    for unit in _units(text, size):
        if current and len(current) + len(unit) + 2 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            if " " in tail:
                tail = tail.split(" ", 1)[1]  # start the overlap on a word boundary
            current = f"{tail}\n\n{unit}" if tail else unit
            if len(current) > size:
                current = unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks


def ingest_bytes(store, metrics, filename: str, data: bytes) -> dict:
    """Extract, chunk, embed and store one file. Re-ingesting a file replaces it."""
    t0 = time.perf_counter()
    text = extract_text(filename, data)
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError(f"No extractable text found in {filename}")
    doc_type = Path(filename).suffix.lstrip(".").lower() or "txt"
    n = store.add_document(filename, chunks, doc_type=doc_type)
    ms = (time.perf_counter() - t0) * 1000
    metrics.log_ingestion(source=filename, n_chunks=n, n_bytes=len(data), latency_ms=ms)
    metrics.log_event("ingest", f"📥 Ingested {filename} into {n} chunks ({ms / 1000:.1f}s)")
    return {"source": filename, "chunks": n, "chars": len(text), "latency_ms": ms}
