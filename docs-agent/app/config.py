"""Central configuration, read from environment variables / .env."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    chat_model: str = os.getenv("CHAT_MODEL", "llama3.1:8b")
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "300"))

    agent_mode: str = os.getenv("AGENT_MODE", "auto").lower()  # auto | rag
    require_retrieval: bool = _bool("REQUIRE_RETRIEVAL", "true")
    max_steps: int = int(os.getenv("MAX_STEPS", "4"))
    history_turns: int = int(os.getenv("HISTORY_TURNS", "4"))

    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/chroma")
    collection: str = os.getenv("COLLECTION", "documents")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    min_score: float = float(os.getenv("MIN_SCORE", "0.45"))

    metrics_db: str = os.getenv("METRICS_DB", "./data/metrics.db")


settings = Settings()
