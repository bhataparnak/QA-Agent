"""Documents page: upload, inspect, delete and test-search the vector DB."""
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app.ingest import SUPPORTED, ingest_bytes
from app.services import get_metrics, get_store, render_status_sidebar

st.set_page_config(page_title="Documents", page_icon="📄", layout="wide")
render_status_sidebar()
store, metrics = get_store(), get_metrics()
SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"

st.title("Documents")
st.write("Files are split into chunks, embedded with Ollama and stored in ChromaDB. "
         "Uploading a file with the same name replaces the old version.")


def run_ingest(files: list[tuple[str, bytes]]) -> None:
    bar = st.progress(0.0, text="Starting…")
    for i, (name, data) in enumerate(files, 1):
        bar.progress((i - 1) / len(files), text=f"Embedding {name}…")
        try:
            r = ingest_bytes(store, metrics, name, data)
            st.success(f"{name}: {r['chunks']} chunks in {r['latency_ms'] / 1000:.1f}s")
        except Exception as e:
            metrics.log_event("ingest", f"❌ Failed to ingest {name}: {e}", level="error")
            st.error(f"{name}: {e}")
    bar.progress(1.0, text="Done")


col_up, col_sample = st.columns([3, 1])
with col_up:
    uploads = st.file_uploader(
        "Add documents", type=sorted(ext.lstrip(".") for ext in SUPPORTED), accept_multiple_files=True
    )
    if uploads and st.button("Add to library", type="primary"):
        run_ingest([(f.name, f.getvalue()) for f in uploads])
with col_sample:
    st.write("Trying it out?")
    if SAMPLE_DIR.exists() and st.button("Load sample documents", use_container_width=True):
        run_ingest([(p.name, p.read_bytes()) for p in sorted(SAMPLE_DIR.iterdir()) if p.suffix.lower() in SUPPORTED])

st.divider()
st.subheader("Library")
sources = store.list_sources()
if not sources:
    st.info("No documents yet. Add some above.")
else:
    df = pd.DataFrame(sources)
    df["ingested_at"] = df["ingested_at"].map(lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M"))
    st.dataframe(df, hide_index=True, use_container_width=True)

    c1, c2 = st.columns([3, 1])
    to_delete = c1.selectbox("Remove a document", [s["source"] for s in sources], index=None, placeholder="Choose a document")
    if c2.button("Remove document", disabled=to_delete is None, use_container_width=True):
        store.delete_source(to_delete)
        metrics.log_event("delete", f"🗑️ Removed {to_delete}")
        st.rerun()

st.divider()
st.subheader("Test retrieval")
st.write("See exactly which chunks the agent would get for a query, and how similar they are.")
q = st.text_input("Search query")
k = st.slider("Results", 1, 10, 5)
if q:
    hits = store.search(q, k)
    if not hits:
        st.warning("No results. Is the library empty?")
    for h in hits:
        with st.container(border=True):
            st.markdown(f"**{h['source']}** (chunk {h['chunk']}), similarity `{h['score']:.3f}`")
            st.progress(max(0.0, min(1.0, h["score"])))
            st.text(h["text"][:800])
