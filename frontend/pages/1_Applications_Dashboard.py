"""Applications Dashboard — table of all processed jobs with editable status & notes."""

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

from backend.db import (  # noqa: E402
    VALID_STATUSES,
    list_applications,
    update_notes,
    update_status,
)

st.set_page_config(page_title="Applications Dashboard", page_icon="📊", layout="wide")
st.title("📊 Applications Dashboard")

apps = list_applications()
if not apps:
    st.info("No applications yet. Process a job on the main page first.")
    st.stop()

# Summary metrics
counts = {s: 0 for s in VALID_STATUSES}
for a in apps:
    counts[a.get("status", "To Apply")] = counts.get(a.get("status", "To Apply"), 0) + 1
cols = st.columns(len(VALID_STATUSES) + 1)
cols[0].metric("Total", len(apps))
for col, status in zip(cols[1:], VALID_STATUSES):
    col.metric(status, counts.get(status, 0))

st.divider()

# Build an editable table
rows = []
for a in apps:
    created = (a.get("created_at") or "")[:10]
    rows.append(
        {
            "id": a["id"],
            "Company": a.get("company", ""),
            "Role": a.get("role", ""),
            "Match": a.get("match_score", 0),
            "Date": created,
            "Status": a.get("status", "To Apply"),
            "Notes": a.get("notes", "") or "",
        }
    )
df = pd.DataFrame(rows).set_index("id")

edited = st.data_editor(
    df,
    use_container_width=True,
    column_config={
        "Match": st.column_config.ProgressColumn(
            "Match", min_value=0, max_value=100, format="%d"
        ),
        "Status": st.column_config.SelectboxColumn(
            "Status", options=list(VALID_STATUSES), required=True
        ),
        "Notes": st.column_config.TextColumn("Notes", width="large"),
        "Company": st.column_config.TextColumn(disabled=True),
        "Role": st.column_config.TextColumn(disabled=True),
        "Date": st.column_config.TextColumn(disabled=True),
    },
    disabled=["Company", "Role", "Match", "Date"],
    key="apps_editor",
)

if st.button("💾 Save changes", type="primary"):
    changes = 0
    for app_id, row in edited.iterrows():
        original = df.loc[app_id]
        if row["Status"] != original["Status"]:
            update_status(int(app_id), row["Status"])
            changes += 1
        if row["Notes"] != original["Notes"]:
            update_notes(int(app_id), row["Notes"])
            changes += 1
    st.success(f"Saved {changes} change(s).")
    st.rerun()

st.divider()

# Detail viewer
st.subheader("🔍 View application package")
choice = st.selectbox(
    "Select an application",
    options=[a["id"] for a in apps],
    format_func=lambda i: next(
        f"#{a['id']} — {a['company']} / {a['role']}" for a in apps if a["id"] == i
    ),
)
selected = next(a for a in apps if a["id"] == choice)

t1, t2, t3 = st.tabs(["Job", "Tailored bullets", "Cover letter"])
with t1:
    st.json(selected.get("parsed_job") or {})
    st.markdown("**Matching skills:** " + ", ".join(selected.get("matching_skills") or []))
    st.markdown("**Missing skills:** " + ", ".join(selected.get("missing_skills") or []))
with t2:
    bullets = selected.get("tailored_bullets") or []
    if not bullets:
        st.write("No tailored bullets.")
    for i, b in enumerate(bullets, 1):
        st.markdown(f"**Bullet {i}** — *{b.get('rationale','')}*")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original**")
            st.markdown(
                f'<div style="padding:.4rem .7rem;background:#fafafa;border:1px solid #e0e0e0;'
                f'border-radius:5px;font-size:.91rem;">{b.get("original","")}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown("**Rewritten**")
            st.markdown(
                f'<div style="padding:.4rem .7rem;background:#f6fff6;border:1px solid #c8e6c9;'
                f'border-radius:5px;font-size:.91rem;">{b.get("rewritten","")}</div>',
                unsafe_allow_html=True,
            )
        st.write("")
with t3:
    st.text_area(
        "Cover letter", value=selected.get("cover_letter", ""), height=400, disabled=True
    )
