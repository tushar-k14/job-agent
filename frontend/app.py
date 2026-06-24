"""Streamlit frontend — main page.

Run from the repo root:  streamlit run frontend/app.py
"""

from __future__ import annotations

import os
import sys

# Make the `backend` package importable when Streamlit runs this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st  # noqa: E402

try:
    from dotenv import load_dotenv  # noqa: E402

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

from backend.db import init_db  # noqa: E402
from backend.graph import run_batch, run_single  # noqa: E402
from backend.graph.pipeline import run_from_text  # noqa: E402
from frontend.diff_utils import render_word_diff  # noqa: E402

st.set_page_config(page_title="Job Application Agent", page_icon="🤖", layout="wide")

init_db()


# --------------------------------------------------------------------------- #
# Sidebar: resume (entered once, kept in session state)
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    with st.sidebar:
        st.header("📄 Your Resume")
        st.caption("Paste once — it's reused for every job you process this session.")
        resume = st.text_area(
            "Resume text",
            value=st.session_state.get("resume_text", ""),
            height=400,
            placeholder="Paste your full resume here...",
            label_visibility="collapsed",
        )
        st.session_state["resume_text"] = resume

        if resume.strip():
            st.success(f"Resume loaded ({len(resume.split())} words).")
        else:
            st.warning("Add your resume to enable processing.")

        st.divider()
        missing = [
            k for k in ("DEEPSEEK_API_KEY", "GEMINI_API_KEY") if not os.getenv(k)
        ]
        if missing == ["DEEPSEEK_API_KEY", "GEMINI_API_KEY"]:
            st.error("No LLM API keys set. Configure DEEPSEEK_API_KEY (or GEMINI_API_KEY).")
        elif "DEEPSEEK_API_KEY" in missing:
            st.info("DeepSeek key missing — will use Gemini fallback.")


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #
def gauge_color(score: int) -> str:
    if score >= 75:
        return "#2e7d32"
    if score >= 50:
        return "#f9a825"
    return "#c62828"


def render_match_gauge(score: int) -> None:
    color = gauge_color(score)
    st.markdown(
        f"""
        <div style="text-align:center;padding:1rem;">
          <div style="font-size:4rem;font-weight:700;color:{color};">{score}<span style="font-size:1.5rem;">/100</span></div>
          <div style="background:#eee;border-radius:8px;height:24px;width:100%;overflow:hidden;">
            <div style="width:{score}%;background:{color};height:100%;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _as_list(value) -> list:
    """Coerce a state field to a plain list regardless of LangGraph wrapper type."""
    if not value:
        return []
    if isinstance(value, list):
        return list(value)
    return list(value)


def _as_dict(value) -> dict:
    """Coerce a state field to a plain dict."""
    if not value:
        return {}
    return dict(value)


def render_results(state: dict) -> None:
    # Normalise: LangGraph returns AddableValuesDict; convert to plain dict once.
    state = dict(state)

    if state.get("error"):
        st.error(f"Pipeline error: {state['error']}")
        # If we have no job data at all, stop here — nothing useful to show.
        if not state.get("parsed_job"):
            st.info(
                "**Tip:** Many job boards (LinkedIn, Indeed, Greenhouse) block automated "
                "scraping. Try:\n"
                "- Pasting the **direct job description URL** (not the listing page)\n"
                "- Using a company careers page URL instead of an aggregator\n"
                "- Copying the job text manually into a pastebin and pasting that URL"
            )
            return

    parsed = _as_dict(state.get("parsed_job"))
    tabs = st.tabs(
        ["📋 Job Analysis", "🎯 Match Score", "✍️ Tailored Resume", "📨 Cover Letter"]
    )

    # --- Job Analysis ---
    with tabs[0]:
        title = parsed.get("title") or "Unknown role"
        company = parsed.get("company") or "Unknown company"
        st.subheader(title)
        st.caption(f"**{company}**")
        if parsed.get("salary"):
            st.info(f"💰 Salary: {parsed['salary']}")

        required = _as_list(parsed.get("required_skills"))
        nice = _as_list(parsed.get("nice_to_haves"))
        responsibilities = _as_list(parsed.get("responsibilities"))

        if not required and not nice and not responsibilities:
            st.warning("No structured job data extracted — check the raw page or try another URL.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Required skills**")
                for s in required or ["—"]:
                    st.markdown(f"- {s}")
                st.markdown("**Nice to have**")
                for s in nice or ["—"]:
                    st.markdown(f"- {s}")
            with c2:
                st.markdown("**Responsibilities**")
                for s in responsibilities or ["—"]:
                    st.markdown(f"- {s}")

    # --- Match Score ---
    with tabs[1]:
        render_match_gauge(int(state.get("match_score") or 0))
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**✅ Matching skills**")
            matching = _as_list(state.get("matching_skills"))
            for s in matching or ["—"]:
                st.markdown(f"- {s}")
        with c2:
            st.markdown("**⚠️ Missing skills**")
            missing = _as_list(state.get("missing_skills"))
            for s in missing or ["—"]:
                st.markdown(f"- {s}")
        with c3:
            st.markdown("**🔄 Transferable**")
            transferable = _as_list(state.get("transferable_experiences"))
            for s in transferable or ["—"]:
                st.markdown(f"- {s}")

    # --- Tailored Resume (diff view) ---
    with tabs[2]:
        bullets = _as_list(state.get("tailored_bullets"))
        if not bullets:
            st.write("No tailored bullets were produced.")
        for i, b in enumerate(bullets, 1):
            b = dict(b)
            original = b.get("original", "")
            rewritten = b.get("rewritten", "")
            rationale = b.get("rationale", "")
            st.markdown(f"**Bullet {i}** — *{rationale}*")
            col_orig, col_new = st.columns(2)
            with col_orig:
                st.markdown("**Original**")
                st.markdown(
                    f'<div style="padding:.5rem .75rem;background:#fafafa;border:1px solid #e0e0e0;'
                    f'border-radius:6px;font-size:.92rem;">{original}</div>',
                    unsafe_allow_html=True,
                )
            with col_new:
                st.markdown("**Rewritten**")
                diff_html = render_word_diff(original, rewritten)
                st.markdown(
                    f'<div style="padding:.5rem .75rem;background:#f6fff6;border:1px solid #c8e6c9;'
                    f'border-radius:6px;font-size:.92rem;">{diff_html}</div>',
                    unsafe_allow_html=True,
                )
            st.write("")

    # --- Cover Letter (editable + download) ---
    with tabs[3]:
        letter = st.text_area(
            "Cover letter (editable)",
            value=state.get("cover_letter") or "",
            height=420,
            key=f"cover_{state.get('application_id', 'tmp')}",
        )
        st.download_button(
            "⬇️ Download as .txt",
            data=letter,
            file_name="cover_letter.txt",
            mime="text/plain",
            key=f"dl_{state.get('application_id', 'tmp')}",
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    render_sidebar()
    st.title("🤖 Autonomous Job Application Agent")
    st.caption("Paste a job URL and let the agents scrape, analyze, tailor, and write.")

    resume = st.session_state.get("resume_text", "").strip()
    mode = st.radio("Mode", ["Single job", "Batch (multiple URLs)"], horizontal=True)

    if mode == "Single job":
        input_mode = st.radio(
            "Job input method",
            ["URL (company careers page)", "Paste job description text"],
            horizontal=True,
            help="LinkedIn/Indeed/Glassdoor block scraping. Use a direct company careers URL or paste the JD text.",
        )

        url = ""
        raw_jd_text = ""

        if input_mode == "URL (company careers page)":
            url = st.text_input(
                "Job posting URL",
                placeholder="https://company.com/careers/job-id  (avoid LinkedIn/Indeed)",
            )
            st.caption("Works best with direct company careers pages (Greenhouse, Lever, Workday, etc.)")
        else:
            raw_jd_text = st.text_area(
                "Paste the full job description here",
                height=260,
                placeholder="Copy and paste the full job description text from any job posting...",
            )
            url = st.text_input(
                "Job URL (optional — for tracking only)",
                placeholder="https://...",
            )

        can_process = resume and (url or raw_jd_text)
        if st.button("🚀 Process", type="primary", disabled=not can_process):
            with st.spinner("Running the agent pipeline..."):
                if raw_jd_text.strip():
                    state = run_from_text(url, raw_jd_text, resume)
                else:
                    state = run_single(url, resume)
            st.session_state["last_result"] = state
            if state.get("error") and not state.get("parsed_job"):
                st.error(f"Failed: {state['error']}")
            else:
                st.success(f"Done — saved as application #{state.get('application_id')}.")

        if st.session_state.get("last_result"):
            st.divider()
            render_results(st.session_state["last_result"])

    else:
        urls_raw = st.text_area(
            "Job URLs (one per line)",
            height=160,
            placeholder="https://job-one\nhttps://job-two\nhttps://job-three",
        )
        urls = [u for u in urls_raw.splitlines() if u.strip()]
        if st.button(
            f"🚀 Process {len(urls)} jobs",
            type="primary",
            disabled=not (urls and resume),
        ):
            progress = st.progress(0.0, text="Starting...")
            results: list[dict] = []

            def on_complete(done: int, total: int, state: dict) -> None:
                company = (state.get("parsed_job") or {}).get("company", "?")
                progress.progress(done / total, text=f"{done}/{total} — {company}")

            with st.spinner("Processing batch in parallel..."):
                results = run_batch(urls, resume, on_complete=on_complete)
            progress.progress(1.0, text="Complete")
            st.success(f"Processed {len(results)} jobs. See the Dashboard page.")

            for state in results:
                title = (state.get("parsed_job") or {}).get("title", state.get("job_url"))
                with st.expander(
                    f"{title} — score {state.get('match_score', 0)}"
                    + (" ⚠️ error" if state.get("error") else "")
                ):
                    render_results(state)


if __name__ == "__main__":
    main()
