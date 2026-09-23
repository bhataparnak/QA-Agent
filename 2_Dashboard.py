"""Real-time dashboard: agent traffic, latency, retrieval quality, tokens, feedback and live activity."""
import json
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from app.config import settings
from app.services import get_metrics, get_store, render_status_sidebar

st.set_page_config(page_title="Agent dashboard", page_icon="📊", layout="wide")
render_status_sidebar()
metrics, store = get_metrics(), get_store()

LOCAL_TZ = datetime.now().astimezone().tzinfo
WINDOWS = {"Last 15 minutes": 15, "Last hour": 60, "Last 24 hours": 1440, "Last 7 days": 10080}
COLORS = {"primary": "#2F6F8F", "accent": "#E0A526", "muted": "#9FB4C0", "bad": "#C8553D"}


def bucket_for(minutes: int) -> pd.Timedelta:
    if minutes <= 15:
        return pd.Timedelta(seconds=30)
    if minutes <= 60:
        return pd.Timedelta(minutes=2)
    if minutes <= 1440:
        return pd.Timedelta(minutes=30)
    return pd.Timedelta(hours=6)


def with_time(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty:
        df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(LOCAL_TZ)
    return df


def style(fig, height: int = 300):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=height, legend_title_text="")
    return fig


def delta(cur, prev, fmt="{:+.1f}"):
    if prev is None or cur is None or pd.isna(cur) or pd.isna(prev):
        return None
    return fmt.format(cur - prev)


# ------------------------------------------------------------------ controls
st.title("Agent dashboard")
c1, c2, c3 = st.columns([2, 2, 1])
window_label = c1.selectbox("Time window", list(WINDOWS), index=1)
refresh = c2.select_slider("Refresh every", options=[2, 5, 10, 30, 60], value=5, format_func=lambda s: f"{s}s")
live = c3.toggle("Live", value=True)
window_min = WINDOWS[window_label]


@st.fragment(run_every=refresh if live else None)
def render_dashboard(window_min: int) -> None:
    now = time.time()
    since = now - window_min * 60
    q = with_time(metrics.queries(since))
    prev = metrics.queries(since - window_min * 60, since)
    fb = metrics.feedback(since)
    sources = store.list_sources()

    st.caption(f"Updated {datetime.now().strftime('%H:%M:%S')}" + (f", refreshing every {refresh}s" if live else ""))

    # ---------------- KPIs
    n, n_prev = len(q), len(prev)
    lat = q["latency_ms"].mean() / 1000 if n else None
    lat_prev = prev["latency_ms"].mean() / 1000 if n_prev else None
    p95 = q["latency_ms"].quantile(0.95) / 1000 if n else None
    grounded = q["grounded"].mean() * 100 if n else None
    grounded_prev = prev["grounded"].mean() * 100 if n_prev else None
    score = q["top_score"].mean() if n else None
    score_prev = prev["top_score"].mean() if n_prev else None
    tokens = int((q["prompt_tokens"] + q["completion_tokens"]).sum()) if n else 0
    errors = q["error"].notna().mean() * 100 if n else None
    satisfaction = (fb["rating"] > 0).mean() * 100 if len(fb) else None

    k = st.columns(8)
    k[0].metric("Queries", n, delta=n - n_prev if n_prev or n else None)
    k[1].metric("Avg latency", f"{lat:.1f}s" if lat is not None else "–",
                delta=delta(lat, lat_prev, "{:+.1f}s"), delta_color="inverse")
    k[2].metric("p95 latency", f"{p95:.1f}s" if p95 is not None else "–")
    k[3].metric("Grounded answers", f"{grounded:.0f}%" if grounded is not None else "–",
                delta=delta(grounded, grounded_prev, "{:+.0f} pts"),
                help=f"Share of answers whose best retrieved chunk scored ≥ {settings.min_score}")
    k[4].metric("Avg top similarity", f"{score:.2f}" if score is not None and not pd.isna(score) else "–",
                delta=delta(score, score_prev, "{:+.2f}"))
    k[5].metric("Tokens", f"{tokens:,}")
    k[6].metric("Error rate", f"{errors:.0f}%" if errors is not None else "–")
    k[7].metric("Thumbs up", f"{satisfaction:.0f}%" if satisfaction is not None else "–",
                help=f"{len(fb)} ratings in this window")

    if q.empty:
        st.info(
            "No questions asked in this window yet. Ask something on the chat page, "
            "or run `python scripts/simulate_traffic.py` to generate demo traffic."
        )
    else:
        bucket = bucket_for(window_min)
        t = q.set_index("time")

        # ---------------- row 1: traffic + latency
        r1a, r1b = st.columns(2)
        traffic = pd.DataFrame({
            "Grounded": t["grounded"].resample(bucket).sum(),
            "Weak retrieval": (1 - t["grounded"]).resample(bucket).sum(),
        })
        fig = px.bar(traffic, title="Questions over time",
                     color_discrete_map={"Grounded": COLORS["primary"], "Weak retrieval": COLORS["accent"]})
        fig.update_layout(xaxis_title=None, yaxis_title="Questions")
        r1a.plotly_chart(style(fig), use_container_width=True)

        latency = pd.DataFrame({
            "Median": t["latency_ms"].resample(bucket).median() / 1000,
            "p95": t["latency_ms"].resample(bucket).quantile(0.95) / 1000,
        }).dropna()
        fig = px.line(latency, markers=True, title="Response time (seconds)",
                      color_discrete_map={"Median": COLORS["primary"], "p95": COLORS["bad"]})
        fig.update_layout(xaxis_title=None, yaxis_title="Seconds")
        r1b.plotly_chart(style(fig), use_container_width=True)

        # ---------------- row 2: time split, score distribution, tokens
        r2a, r2b, r2c = st.columns(3)
        retrieval = q["retrieval_ms"].sum()
        llm_time = q["llm_ms"].sum()
        other = max(q["latency_ms"].sum() - retrieval - llm_time, 0)
        split = pd.DataFrame({"stage": ["LLM generation", "Vector search", "Other"],
                              "ms": [llm_time, retrieval, other]})
        fig = px.pie(split, names="stage", values="ms", hole=0.55, title="Where the time goes",
                     color="stage", color_discrete_map={"LLM generation": COLORS["primary"],
                                                        "Vector search": COLORS["accent"], "Other": COLORS["muted"]})
        r2a.plotly_chart(style(fig), use_container_width=True)

        scores = q.dropna(subset=["top_score"])
        if scores.empty:
            r2b.info("No retrieval scores yet.")
        else:
            fig = px.histogram(scores, x="top_score", nbins=20, title="Best-match similarity per question",
                               color_discrete_sequence=[COLORS["primary"]])
            fig.add_vline(x=settings.min_score, line_dash="dash", line_color=COLORS["bad"],
                          annotation_text="grounding threshold")
            fig.update_layout(xaxis_title="Cosine similarity", yaxis_title="Questions")
            r2b.plotly_chart(style(fig), use_container_width=True)

        tok = pd.DataFrame({
            "Prompt": t["prompt_tokens"].resample(bucket).sum(),
            "Completion": t["completion_tokens"].resample(bucket).sum(),
        })
        fig = px.bar(tok, title="Token usage", color_discrete_map={"Prompt": COLORS["muted"],
                                                                   "Completion": COLORS["primary"]})
        fig.update_layout(xaxis_title=None, yaxis_title="Tokens")
        r2c.plotly_chart(style(fig), use_container_width=True)

        # ---------------- row 3: cited sources + knowledge base
        r3a, r3b = st.columns(2)

        def _parse(s):
            try:
                return json.loads(s) if s else []
            except (TypeError, ValueError):
                return []

        cited = q["sources"].map(_parse).explode().dropna()
        if cited.empty:
            r3a.info("No documents cited above the grounding threshold yet.")
        else:
            counts = cited.value_counts().head(10).sort_values()
            fig = px.bar(x=counts.values, y=counts.index, orientation="h", title="Most cited documents",
                         color_discrete_sequence=[COLORS["primary"]])
            fig.update_layout(xaxis_title="Answers citing it", yaxis_title=None)
            r3a.plotly_chart(style(fig), use_container_width=True)

    if sources:
        kb = pd.DataFrame(sources).sort_values("chunks").tail(12)
        fig = px.bar(kb, x="chunks", y="source", orientation="h", color="type",
                     title=f"Knowledge base: {sum(s['chunks'] for s in sources)} chunks in {len(sources)} documents")
        fig.update_layout(xaxis_title="Chunks", yaxis_title=None)
        target = r3b if not q.empty else st
        target.plotly_chart(style(fig), use_container_width=True)

    # ---------------- live activity + recent questions
    left, right = st.columns([2, 3])
    with left:
        st.subheader("Live activity")
        ev = metrics.events(25)
        if ev.empty:
            st.caption("Nothing yet.")
        else:
            ev["time"] = ev["ts"].map(lambda x: datetime.fromtimestamp(x).strftime("%H:%M:%S"))
            st.dataframe(ev[["time", "message"]], hide_index=True, use_container_width=True, height=380)
    with right:
        st.subheader("Recent questions")
        if q.empty:
            st.caption("Nothing yet.")
        else:
            recent = q.sort_values("ts", ascending=False).head(25).copy()
            recent["when"] = recent["time"].dt.strftime("%H:%M:%S")
            recent["latency (s)"] = (recent["latency_ms"] / 1000).round(1)
            recent["tokens"] = recent["prompt_tokens"] + recent["completion_tokens"]
            recent["grounded"] = recent["grounded"].map({1: "✅", 0: "⚠️"})
            st.dataframe(
                recent[["when", "question", "latency (s)", "top_score", "grounded", "tool_calls", "tokens", "mode"]],
                hide_index=True, use_container_width=True, height=380,
                column_config={"top_score": st.column_config.ProgressColumn(
                    "top similarity", min_value=0.0, max_value=1.0, format="%.2f")},
            )


render_dashboard(window_min)

with st.expander("Danger zone"):
    if st.button("Clear all metrics"):
        metrics.reset()
        st.rerun()
