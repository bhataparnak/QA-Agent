"""SQLite metrics store: every agent run, ingestion, feedback and event is logged here."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT,
    question TEXT,
    answer TEXT,
    mode TEXT,
    latency_ms REAL,
    retrieval_ms REAL,
    llm_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    tool_calls INTEGER,
    n_chunks INTEGER,
    top_score REAL,
    avg_score REAL,
    grounded INTEGER,
    sources TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_queries_ts ON queries(ts);

CREATE TABLE IF NOT EXISTS ingestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    source TEXT,
    n_chunks INTEGER,
    n_bytes INTEGER,
    latency_ms REAL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    query_id INTEGER,
    rating INTEGER
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    level TEXT,
    kind TEXT,
    message TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


class MetricsStore:
    def __init__(self, path: str | None = None):
        self.path = path or settings.metrics_db
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _insert(self, table: str, row: dict) -> int:
        row = {"ts": time.time(), **row}
        cols = ", ".join(row)
        marks = ", ".join(["?"] * len(row))
        with self._conn() as c:
            cur = c.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))
            return cur.lastrowid

    # ---------- write ----------
    def log_query(self, **row) -> int:
        if isinstance(row.get("sources"), (list, tuple)):
            row["sources"] = json.dumps(list(row["sources"]))
        return self._insert("queries", row)

    def log_ingestion(self, source: str, n_chunks: int, n_bytes: int, latency_ms: float) -> int:
        return self._insert(
            "ingestions",
            {"source": source, "n_chunks": n_chunks, "n_bytes": n_bytes, "latency_ms": latency_ms},
        )

    def log_event(self, kind: str, message: str, level: str = "info") -> int:
        return self._insert("events", {"kind": kind, "message": message, "level": level})

    def log_feedback(self, query_id: int, rating: int) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM feedback WHERE query_id = ?", (query_id,))
        self._insert("feedback", {"query_id": query_id, "rating": rating})

    def reset(self) -> None:
        with self._conn() as c:
            for t in ("queries", "ingestions", "feedback", "events"):
                c.execute(f"DELETE FROM {t}")

    # ---------- read ----------
    def _df(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query(sql, c, params=params)

    def queries(self, since_ts: float, until_ts: float | None = None) -> pd.DataFrame:
        until_ts = until_ts or time.time() + 1
        return self._df("SELECT * FROM queries WHERE ts >= ? AND ts < ? ORDER BY ts", (since_ts, until_ts))

    def feedback(self, since_ts: float) -> pd.DataFrame:
        return self._df("SELECT * FROM feedback WHERE ts >= ?", (since_ts,))

    def ingestions(self, since_ts: float) -> pd.DataFrame:
        return self._df("SELECT * FROM ingestions WHERE ts >= ? ORDER BY ts", (since_ts,))

    def events(self, limit: int = 30) -> pd.DataFrame:
        return self._df("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
