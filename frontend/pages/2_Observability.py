"""Observability dashboard — recent runs, pass rate over time, cost per run.

Reads the ``run_traces`` table populated by every pipeline invocation.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

try:
    from dotenv import load_dotenv  # noqa: E402

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

from backend.db import list_run_traces  # noqa: E402
from backend.observability.tracing import LANGSMITH_ENABLED  # noqa: E402

st.set_page_config(page_title="Observability", page_icon="📈", layout="wide")
st.title("📈 Observability")

if LANGSMITH_ENABLED:
    st.caption("LangSmith tracing is **enabled** (LANGSMITH_API_KEY detected). "
               "Full traces are also available in the LangSmith UI.")
else:
    st.caption("Local structlog tracing. Set `LANGSMITH_API_KEY` to also stream traces to LangSmith.")

traces = list_run_traces(limit=500)
if not traces:
    st.info("No runs traced yet. Process a job on the main page first.")
    st.stop()

df = pd.DataFrame(traces)
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df = df.sort_values("created_at")

# ---- headline metrics ----
total = len(df)
success_rate = df["success"].mean() * 100 if total else 0
avg_latency = df["latency_ms"].mean() if total else 0
avg_tokens = df["total_tokens"].mean() if total else 0
fallbacks = int((df["cover_passed"] == 1).sum())  # cover passed includes fallback path

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total runs", total)
c2.metric("Success rate", f"{success_rate:.0f}%")
c3.metric("Avg latency", f"{avg_latency/1000:.2f}s")
c4.metric("Avg tokens/run", f"{avg_tokens:.0f}")

st.divider()

# ---- pass rate over time (rolling) ----
st.subheader("Scrape success over time")
ts = df.set_index("created_at")
if len(ts) >= 1:
    rolling = ts["scrape_passed"].rolling(window=min(10, len(ts)), min_periods=1).mean() * 100
    chart_df = pd.DataFrame({"scrape_pass_rate_%": rolling})
    st.line_chart(chart_df)

# ---- token cost over time ----
st.subheader("Tokens per run")
st.bar_chart(ts[["total_tokens"]])

st.divider()

# ---- recent runs table ----
st.subheader("Recent runs")
display = df.sort_values("created_at", ascending=False)[
    ["created_at", "application_id", "run_id", "success", "scrape_passed",
     "cover_passed", "strategy_used", "scrape_attempts", "latency_ms", "total_tokens",
     "total_llm_calls"]
].head(50).copy()
display["created_at"] = display["created_at"].dt.strftime("%Y-%m-%d %H:%M")
st.dataframe(
    display,
    use_container_width=True,
    column_config={
        "success": st.column_config.CheckboxColumn("ok"),
        "scrape_passed": st.column_config.CheckboxColumn("scrape"),
        "cover_passed": st.column_config.CheckboxColumn("cover"),
        "latency_ms": st.column_config.NumberColumn("latency (ms)", format="%.0f"),
    },
)

# ---- per-run step drilldown ----
st.subheader("🔬 Run step trace")
run_ids = display["run_id"].tolist()
if run_ids:
    chosen = st.selectbox("Select a run", run_ids)
    rec = next((t for t in traces if t["run_id"] == chosen), None)
    if rec and rec.get("trace"):
        steps = rec["trace"].get("steps", [])
        step_df = pd.DataFrame(steps)
        cols = [c for c in ["name", "latency_ms", "input", "output", "llm_calls",
                            "total_tokens", "error"] if c in step_df.columns]
        st.dataframe(step_df[cols], use_container_width=True)
