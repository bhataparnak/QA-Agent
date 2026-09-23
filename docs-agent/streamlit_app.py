"""Chat page: ask questions about your documents."""
import uuid

import streamlit as st

from app.services import get_agent, get_metrics, get_store, render_status_sidebar

st.set_page_config(page_title="Docs Agent", page_icon="🤖", layout="wide")
render_status_sidebar()

st.session_state.setdefault("session_id", str(uuid.uuid4()))
st.session_state.setdefault("messages", [])

with st.sidebar:
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


def save_feedback(query_id: int) -> None:
    value = st.session_state.get(f"fb_{query_id}")
    if value is not None:
        get_metrics().log_feedback(query_id, 1 if value == 1 else -1)


def render_assistant(m: dict) -> None:
    st.markdown(m["content"])
    meta = m.get("meta")
    if meta:
        line = (
            f"{meta['latency']:.1f}s | {meta['tool_calls']} tool calls | "
            f"{meta['tokens']} tokens | mode: {meta['mode']}"
        )
        if not meta["grounded"]:
            line += " | ⚠️ weak match in documents, verify this answer"
        st.caption(line)
    if m.get("sources"):
        with st.expander(f"Sources ({len(m['sources'])})"):
            for s in m["sources"]:
                st.markdown(f"**{s['source']}** (chunk {s['chunk']}, similarity `{s['score']:.2f}`)")
                preview = s["text"][:400] + ("…" if len(s["text"]) > 400 else "")
                st.caption(preview)
    if m.get("steps"):
        with st.expander("Agent trace"):
            for step in m["steps"]:
                st.write(step)
    if m.get("query_id"):
        st.feedback("thumbs", key=f"fb_{m['query_id']}", on_change=save_feedback, args=(m["query_id"],))


st.title("Ask your documents")
if get_store().count() == 0:
    st.info("Your document library is empty. Add files on the **Documents** page, then come back to ask questions.")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        render_assistant(m) if m["role"] == "assistant" else st.markdown(m["content"])

if prompt := st.chat_input("Ask a question about your documents"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Working on it…", expanded=True) as status:
            result = get_agent().run(
                prompt,
                history=history,
                session_id=st.session_state.session_id,
                on_step=lambda kind, msg: status.write(msg),
            )
            status.update(
                label=f"Done in {result.latency_ms / 1000:.1f}s" if not result.error else "Failed",
                state="error" if result.error else "complete",
                expanded=False,
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "sources": result.sources,
            "steps": result.steps,
            "query_id": result.query_id,
            "meta": {
                "latency": result.latency_ms / 1000,
                "tool_calls": result.tool_calls,
                "tokens": result.total_tokens,
                "mode": result.mode,
                "grounded": result.grounded,
            },
        }
    )
    st.rerun()
