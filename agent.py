"""The document Q&A agent.

Two modes:
  * tools - the LLM decides when to call `search_documents` / `save_note` (ReAct-style loop)
  * rag   - classic retrieve-then-answer; used directly (AGENT_MODE=rag) or as a fallback when
            the model has no tool support or tries to answer without searching.

Every run is timed, token-counted and logged to the metrics store for the dashboard.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import ollama

from . import llm
from .config import settings
from .metrics import MetricsStore
from .vector_store import VectorStore

StepCallback = Callable[[str, str], None]

SYSTEM_PROMPT = """You are a support assistant that answers questions using the user's document library.

Rules:
- For any question about products, policies, procedures or facts, call `search_documents` first.
  You may search several times with different phrasings if the first results are weak.
- Answer ONLY from the retrieved passages. If they don't contain the answer, say you couldn't
  find it in the documents and suggest what the user could upload or ask instead.
- Cite sources inline using the file name in brackets, e.g. [pricing.pdf].
- Use `save_note` only when the user explicitly asks you to remember or save something.
- Be concise. Use short paragraphs or bullet points when it helps."""

RAG_SYSTEM_PROMPT = """You are a support assistant. Answer the user's question using ONLY the
context passages provided. If the context doesn't contain the answer, say you couldn't find it
in the documents. Cite sources inline using the file name in brackets, e.g. [pricing.pdf].
Be concise."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Semantic search over the user's uploaded documents. Returns the most "
            "relevant passages with their source file names and similarity scores (0-1).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A focused search query."},
                    "k": {"type": "integer", "description": "Number of passages (1-10). Default 5."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a short note or fact to the knowledge base so it can be found later. "
            "Only use when the user explicitly asks to remember or save something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short title for the note."},
                    "text": {"type": "string", "description": "The content to save."},
                },
                "required": ["title", "text"],
            },
        },
    },
]


@dataclass
class AgentResult:
    answer: str = ""
    sources: list[dict] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    mode: str = ""
    latency_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0
    top_score: float | None = None
    grounded: bool = False
    error: str | None = None
    query_id: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class _Run:
    """Per-request state, so one agent instance can serve many sessions safely."""
    res: AgentResult
    on_step: StepCallback | None
    hits: dict[str, dict] = field(default_factory=dict)


class DocsAgent:
    def __init__(self, store: VectorStore, metrics: MetricsStore):
        self.store = store
        self.metrics = metrics

    # ------------------------------------------------------------------ public
    def run(
        self,
        question: str,
        history: list[dict] | None = None,
        session_id: str | None = None,
        on_step: StepCallback | None = None,
    ) -> AgentResult:
        run = _Run(AgentResult(), on_step)
        t0 = time.perf_counter()
        history = self._trim_history(history or [])
        self._emit(run, "query", f"❓ Question: {question[:80]}")

        try:
            if settings.agent_mode == "rag":
                run.res.answer = self._run_rag(run, question, history)
            else:
                try:
                    run.res.answer = self._run_tools(run, question, history)
                except ollama.ResponseError as e:
                    if "does not support tools" not in str(e).lower():
                        raise
                    self._emit(run, "fallback", f"⚠️ {settings.chat_model} has no tool support, using classic RAG")
                    run.res.answer = self._run_rag(run, question, history)
        except Exception as e:
            run.res.error = f"{type(e).__name__}: {e}"
            run.res.answer = f"The agent hit an error and couldn't answer: {e}"
            self._emit(run, "error", f"❌ {run.res.error}", level="error")

        self._finalize(run, question, session_id, t0)
        return run.res

    # ------------------------------------------------------------------ modes
    def _run_tools(self, run: _Run, question: str, history: list[dict]) -> str:
        run.res.mode = "tools"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": question}]

        for _ in range(settings.max_steps):
            msg = self._chat(run, messages, tools=TOOLS).message
            if not msg.tool_calls:
                if settings.require_retrieval and run.res.tool_calls == 0 and self.store.count() > 0:
                    self._emit(run, "fallback", "↩️ Model answered without searching, grounding it with retrieval")
                    return self._run_rag(run, question, history)
                return (msg.content or "").strip()

            messages.append(msg)
            for call in msg.tool_calls:
                name = call.function.name
                args = dict(call.function.arguments or {})
                run.res.tool_calls += 1
                output = self._dispatch(run, name, args, question)
                messages.append({"role": "tool", "content": output, "tool_name": name})

        self._emit(run, "limit", "⏱️ Step limit reached, asking for a final answer")
        messages.append({"role": "user", "content": "Using what you found, give your final answer now."})
        return (self._chat(run, messages).message.content or "").strip()

    def _run_rag(self, run: _Run, question: str, history: list[dict]) -> str:
        run.res.mode = f"{run.res.mode}→rag" if run.res.mode else "rag"
        hits = self._search(run, question)
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": f"Context passages:\n\n{self._format_hits(hits)}\n\nQuestion: {question}"},
        ]
        return (self._chat(run, messages).message.content or "").strip()

    # ------------------------------------------------------------------ tools
    def _dispatch(self, run: _Run, name: str, args: dict, question: str) -> str:
        if name == "search_documents":
            try:
                k = int(args.get("k") or settings.top_k)
            except (TypeError, ValueError):
                k = settings.top_k
            query = str(args.get("query") or question)
            return self._format_hits(self._search(run, query, k))

        if name == "save_note":
            title = str(args.get("title") or "note").strip()
            text = str(args.get("text") or "").strip()
            if not text:
                return "Nothing saved: 'text' was empty."
            self.store.add_note(title, text)
            self._emit(run, "tool", f"📝 save_note(\"{title[:40]}\")")
            return f"Saved note '{title}' to the knowledge base."

        self._emit(run, "tool", f"⚠️ Unknown tool requested: {name}", level="warning")
        return f"Unknown tool: {name}. Available tools: search_documents, save_note."

    def _search(self, run: _Run, query: str, k: int | None = None) -> list[dict]:
        k = max(1, min(int(k or settings.top_k), 10))
        self._emit(run, "tool", f"🔎 search_documents(\"{query[:60]}\", k={k})")
        t = time.perf_counter()
        hits = self.store.search(query, k)
        run.res.retrieval_ms += (time.perf_counter() - t) * 1000
        for h in hits:
            prev = run.hits.get(h["id"])
            if prev is None or h["score"] > prev["score"]:
                run.hits[h["id"]] = h
        best = f"{hits[0]['score']:.2f}" if hits else "n/a"
        self._emit(run, "retrieval", f"📚 {len(hits)} passages retrieved, best similarity {best}")
        return hits

    @staticmethod
    def _format_hits(hits: list[dict]) -> str:
        if not hits:
            return "No relevant passages found in the documents."
        return "\n\n---\n\n".join(
            f"[{i}] source: {h['source']} (similarity {h['score']:.2f})\n{h['text']}"
            for i, h in enumerate(hits, 1)
        )

    # ------------------------------------------------------------------ plumbing
    def _chat(self, run: _Run, messages, tools=None):
        t = time.perf_counter()
        resp = llm.chat(messages, tools=tools)
        run.res.llm_ms += (time.perf_counter() - t) * 1000
        run.res.prompt_tokens += resp.prompt_eval_count or 0
        run.res.completion_tokens += resp.eval_count or 0
        return resp

    def _emit(self, run: _Run, kind: str, message: str, level: str = "info") -> None:
        run.res.steps.append(message)
        if run.on_step:
            run.on_step(kind, message)
        try:
            self.metrics.log_event(kind, message, level)
        except Exception:
            pass  # metrics must never break the agent

    @staticmethod
    def _trim_history(history: list[dict]) -> list[dict]:
        clean = [
            {"role": m["role"], "content": m["content"]}
            for m in history
            if m.get("role") in {"user", "assistant"} and isinstance(m.get("content"), str)
        ]
        return clean[-settings.history_turns * 2 :] if settings.history_turns else []

    def _finalize(self, run: _Run, question: str, session_id: str | None, t0: float) -> None:
        res = run.res
        res.latency_ms = (time.perf_counter() - t0) * 1000
        res.sources = sorted(run.hits.values(), key=lambda h: h["score"], reverse=True)
        scores = [h["score"] for h in res.sources]
        res.top_score = max(scores) if scores else None
        res.grounded = bool(scores) and res.top_score >= settings.min_score
        cited = sorted({h["source"] for h in res.sources if h["score"] >= settings.min_score})

        try:
            res.query_id = self.metrics.log_query(
                session_id=session_id,
                question=question,
                answer=res.answer[:2000],
                mode=res.mode,
                latency_ms=res.latency_ms,
                retrieval_ms=res.retrieval_ms,
                llm_ms=res.llm_ms,
                prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens,
                tool_calls=res.tool_calls,
                n_chunks=len(res.sources),
                top_score=res.top_score,
                avg_score=(sum(scores) / len(scores)) if scores else None,
                grounded=int(res.grounded),
                sources=cited,
                error=res.error,
            )
        except Exception:
            pass
        if not res.error:
            self._emit(run, "answer", f"✅ Answered in {res.latency_ms / 1000:.1f}s using {res.total_tokens} tokens")
