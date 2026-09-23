"""Shared singletons for the Streamlit pages (one store/agent per server process)."""
import streamlit as st

from . import llm
from .agent import DocsAgent
from .metrics import MetricsStore
from .vector_store import VectorStore


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore()


@st.cache_resource
def get_metrics() -> MetricsStore:
    return MetricsStore()


@st.cache_resource
def get_agent() -> DocsAgent:
    return DocsAgent(get_store(), get_metrics())


@st.cache_data(ttl=15, show_spinner=False)
def ollama_health() -> dict:
    return llm.health()


def render_status_sidebar() -> None:
    from .config import settings

    with st.sidebar:
        st.subheader("System status")
        h = ollama_health()
        if h["error"]:
            st.error(f"Can't reach Ollama at {settings.ollama_host}. Start it with `ollama serve`.")
        elif h["missing"]:
            st.warning("Pull the missing models, then refresh:")
            st.code("\n".join(f"ollama pull {m}" for m in h["missing"]), language="bash")
        else:
            st.success("Ollama connected")
        st.caption(f"Chat model: `{settings.chat_model}`")
        st.caption(f"Embeddings: `{settings.embed_model}`")
        st.caption(f"Mode: `{settings.agent_mode}`")
        try:
            st.caption(f"Chunks in vector DB: **{get_store().count()}**")
        except Exception as e:
            st.caption(f"Vector DB unavailable: {e}")
