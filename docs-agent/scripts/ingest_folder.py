"""Bulk-ingest every supported file in a folder.

    python scripts/ingest_folder.py ./my_docs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import SUPPORTED, ingest_bytes  # noqa: E402
from app.metrics import MetricsStore  # noqa: E402
from app.vector_store import VectorStore  # noqa: E402


def main(folder: str) -> None:
    store, metrics = VectorStore(), MetricsStore()
    files = [p for p in sorted(Path(folder).rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED]
    if not files:
        print(f"No supported files in {folder} ({', '.join(sorted(SUPPORTED))})")
        return
    for p in files:
        try:
            r = ingest_bytes(store, metrics, p.name, p.read_bytes())
            print(f"✓ {p.name}: {r['chunks']} chunks ({r['latency_ms'] / 1000:.1f}s)")
        except Exception as e:
            print(f"✗ {p.name}: {e}")
    print(f"Done. Vector DB now holds {store.count()} chunks.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample_docs")
