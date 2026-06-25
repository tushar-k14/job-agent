# Current Architecture (Phase 0 baseline)

> **Note:** This is the *historical Phase 0 baseline* captured before the refactor. All
> five phases of the refactor plan (§5) are now implemented — see the top-level
> [`README.md`](../README.md) for the current planner/executor/verifier architecture,
> Chroma memory, benchmark, and observability. This file is kept as the record of where
> the project started and why each change was made.

_Snapshot taken before the planner/executor/verifier refactor. This describes what
existed at the start, honestly, including what was missing for production use._

## 1. What this system actually is

An **Autonomous Job Application Agent**: given a job posting (URL or pasted text) plus
the candidate's resume, it produces a tailored application package — structured job
analysis, a fit score, truthfully-reframed resume bullets, and a cover letter — and
persists it to a tracker.

**Stack (as-built):**

| Concern        | Technology                                              |
|----------------|---------------------------------------------------------|
| Orchestration  | LangGraph (`StateGraph`, linear edges)                  |
| LLM            | DeepSeek (primary) → Gemini (fallback), via `requests`  |
| "Scraping"     | `requests` + BeautifulSoup (static HTML only)           |
| Persistence    | SQLite (`data/applications.db`)                          |
| Frontend       | Streamlit (main page + dashboard page)                  |
| Tests          | pytest, 78 tests, all mocked (no live network/LLM)      |
| Packaging      | Dockerfile + docker-compose (Streamlit on :8501)        |

> **Note on prior assumptions:** there is **no Playwright/Selenium**, **no vector store
> or embeddings**, and **no FastAPI/React**. "Scraping" is a single HTTP GET; "ranking"
> is one LLM call returning a 0–100 score. The upgrade plan accounts for this.

## 2. Nodes / steps that exist today

The graph is a **strictly linear chain** (no branches, no loops, no retries):

```
START → scraper → analysis → tailor → writer → tracker → END
```

| Node       | File                              | Responsibility | LLM? |
|------------|-----------------------------------|----------------|------|
| `scraper`  | `backend/agents/scraper.py`       | HTTP GET + BS4 clean → LLM extracts {title, company, required_skills, responsibilities, nice_to_haves, salary} | yes (`llm_json`) |
| `analysis` | `backend/agents/analysis.py`      | resume ↔ job → {match_score, matching_skills, missing_skills, transferable_experiences} | yes |
| `tailor`   | `backend/agents/tailor.py`        | rewrite ≤6 resume bullets (truthful reframe only) | yes |
| `writer`   | `backend/agents/cover_letter.py`  | 3-paragraph cover letter | yes (`llm_complete`) |
| `tracker`  | `backend/agents/tracker.py`       | persist full package to SQLite, stamp status | no |

(The cover-letter node is registered as `"writer"` because LangGraph forbids a node
name colliding with a state key, and `cover_letter` is a state key.)

## 3. How data flows

- A single mutable **`ApplicationState`** `TypedDict` (`backend/graph/state.py`,
  `total=False`) is threaded through every node. Each node returns a **partial dict**
  that LangGraph shallow-merges into the running state.
- **Error channel:** every node checks `state["error"]` at the top and short-circuits
  (returns `{}`) if a prior node failed — except `tracker`, which still runs so failed
  runs are recorded. There is no typed contract between nodes; they read/write loose
  dict keys.
- **Entry points** (`backend/graph/pipeline.py`):
  - `run_single(url, resume)` → builds initial state, `graph.invoke()`.
  - `run_from_text(url, raw_text, resume)` → **bypasses the graph entirely** and calls
    the node functions manually in sequence (used by the "paste JD" UI path). This is
    duplicated orchestration logic — a refactor smell.
  - `run_batch(urls, resume)` → thread-pool map-reduce; each URL runs its own
    `run_single` concurrently; results reduced to an ordered list.
- **Scrape strategy** is fixed: one set of generic BS4 rules strips
  `script/style/nav/header/footer`, truncates to 12k chars, hands raw text to the LLM.
  Known-blocked domains (LinkedIn/Indeed/Glassdoor/ZipRecruiter) are rejected up front.

## 4. What's missing for production use

| Gap | Today | Risk |
|-----|-------|------|
| **No verification** | Output is whatever the LLM returned; never checked | Empty/boilerplate scrapes and fabricated cover-letter claims pass through silently |
| **No retries / adaptive strategy** | One scrape attempt, one prompt | JS-rendered or unusual pages → empty `parsed_job`, no recovery |
| **No typed hand-offs** | Loose dict keys | Silent shape drift; a typo'd key fails quietly |
| **No memory** | Every run is cold | Re-discovers the same site quirks every time; no learning |
| **No evaluation** | Zero benchmark; README makes qualitative claims | Can't prove quality or catch regressions |
| **No observability** | `logging.info` only; no per-step tokens/latency/trace | Can't debug failures or measure cost |
| **No guardrails** | Bare `try/except` → stuff error string in state | No backoff on transient 429/5xx; no deterministic fallback |
| **Duplicated orchestration** | `run_from_text` reimplements the chain | Two code paths drift apart |

## 5. Refactor plan (proposed — no code yet)

### Phase 1 — Planner / Executor / Verifier
Replace the linear chain with an explicit agentic loop, **behavior-preserving** for the
happy path. Per the agreed design:

- **`planner`** — picks the extraction strategy for the target domain (selector-set A vs
  B vs regex vs LLM-from-raw-HTML) and decides whether enrichment is needed when the JD
  is too thin to tailor confidently.
- **`executor`** — runs the chosen strategy (scrape variant / enrichment fetch / draft
  generation).
- **`verifier`** — two real checks: (1) **scrape validity** (title/company/description
  present, non-empty, not an error/boilerplate page) → on fail, route back to planner
  with the failure reason for an alternate strategy (retries capped at N); (2)
  **cover-letter quality** via LLM-as-judge (no fabricated facts, right length, no
  generic filler) → on fail, regenerate with adjusted instructions.
- **Pydantic models** for every hand-off (`ScrapeStrategy`, `ExtractionResult`,
  `VerificationResult`, etc.) — no raw string parsing between nodes.
- Collapse `run_from_text` into the same graph (paste = a pre-seeded `raw_job_text` that
  skips the fetch but still flows through planner→executor→verifier).

### Phase 2 — Chroma vector memory (real but scoped)
In-process Chroma. Store outcomes + discovered site quirks (e.g. "domain X needs paste
mode", "selector B works for greenhouse"). Planner queries before acting; verifier/
executor write outcomes after. No new infra.

### Phase 3 — Evaluation harness
20–30 fixed tasks (real + mocked fixtures, incl. tricky edge cases), each with automated
pass/fail. `eval/run_benchmark.py` reports pass rate / latency / token cost. GitHub
Actions gate on every push. Real number in README.

### Phase 4 — Observability & guardrails
Tracing per step (LangSmith if key, else structlog/OTel) capturing I/O, tool calls,
tokens, latency. Guardrails: exponential backoff on transient failures, deterministic
fallback when verification keeps failing, Pydantic validation before any output is acted
on. Minimal Streamlit dashboard page: recent runs, pass rate over time, cost per run.

### Guiding constraints
Small reviewable commits; no new infra beyond what each phase needs; **Docker stays
green at the end of every phase**; README claims match reality (substance over
buzzwords).
