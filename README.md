# 🤖 Autonomous Job Application Agent

A LangGraph + Streamlit application that turns a job-posting URL and your resume into
a complete, tailored application package: structured job analysis, a fit score,
truthfully-reframed resume bullets, and a custom cover letter — all tracked in a
SQLite-backed dashboard.

LLM calls use **DeepSeek** as the primary provider with an automatic fallback to the
**free Gemini** tier, so a single missing/rate-limited key won't stop the pipeline.

---

## Architecture

A LangGraph state machine built as an explicit **planner → executor → verifier** loop
with two capped retry gates. Every hand-off between nodes is a validated **Pydantic**
model (`backend/schemas.py`), not loose dict parsing.

```
START → planner → executor → scrape_verifier ──pass──▶ analysis → tailor → writer → cover_letter_verifier ──pass──▶ tracker → END
            ▲                       │ fail                                                      │ fail
            └───── retry (≤3) ──────┘  (escalate extraction strategy)        regenerate (≤3) ──┘  (with corrective feedback)
```

| Node | Responsibility |
|------|----------------|
| **planner** | Chooses the extraction **strategy** for this attempt (and escalates it on retry); flags thin JDs for enrichment. Queries persistent memory for a strategy known to work on the target domain. *Deterministic — no LLM.* |
| **executor** | Runs the chosen strategy (fetches HTML **once**, caches across retries); supports paste-text and enrichment actions. |
| **scrape_verifier** | **Deterministic** gate: title/company/description present, not a captcha/error page → routes back to planner with the failure reason. |
| **analysis** | resume ↔ job → matching/missing/transferable skills, **match score 0–100**. |
| **tailor** | Rewrites existing bullets to fit the job — **reframes only, never invents**. |
| **writer** | 3-paragraph cover letter: *why the role / company / you*. Consumes verifier feedback on retry. |
| **cover_letter_verifier** | **LLM-as-judge** for subtle fabrication + length + filler, backed by a **deterministic entity-grounding** pre-check (every named entity in the letter must trace to the resume or job). |
| **tracker** | Persists the full package to SQLite, stamps status, and **writes the run outcome to vector memory**. |

**Extraction strategies** (`backend/agents/strategies.py`): `selector_structured`
(known ATS containers) → `generic_text` (boilerplate-stripped visible text) →
`llm_from_raw_html` (raw markup to the LLM), plus `paste_text` for pasted JDs. The
planner walks this escalation order across retries.

**Persistent vector memory** (`backend/memory/`, Chroma, in-process): records each run's
`{domain, strategy, success, reason, scores}`. The planner recalls the
highest-success-rate strategy per domain before acting. Defaults to a no-op embedding
(zero model download); set `JOB_AGENT_MEMORY_EMBED=1` for semantic embeddings.

`ApplicationState` (see [`backend/graph/state.py`](backend/graph/state.py)) threads the
job/resume inputs, parsed job, analysis fields, tailored bullets, cover letter, the
planner decision + verifier verdicts, retry counters, and an `error` channel.

### Batch mode (map-reduce)
Multiple URLs (one per line) are fanned out across a thread pool — each URL runs its
own isolated copy of the graph concurrently — then reduced into a list, with a live
Streamlit progress bar. See [`run_batch`](backend/graph/pipeline.py).

## Benchmark

A fixed suite of **24 tasks** with explicit, automated pass/fail criteria exercises the
agent's *structural correctness*: correct strategy selection & escalation, scrape-verifier
verdicts, capped retries, enrichment decisions, blocked-domain handling, and deterministic
cover-letter grounding (no fabrication). Two tiers:

- **Mock tier** (`python -m eval.run_benchmark`) — deterministic, no API keys, no cost.
  Gates CI on every push; the build fails if the pass rate drops below the threshold.
- **Live tier** (`--live`) — runs the real DeepSeek/Gemini pipeline for true quality
  measurement, on demand.

```
$ python -m eval.run_benchmark --threshold 0.95
  tier=mock  tasks=24  passed=24
  PASS RATE : 100.0%
  avg latency: ~135 ms/task
```

**Current benchmark (mock tier): 24/24 = 100% pass rate.** Runs in CI via
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

---

## Repo layout (monorepo)

```
job-agent/
├── backend/
│   ├── llm/            # DeepSeek primary + Gemini fallback client
│   ├── agents/         # planner, executor, verifier, analysis, tailor,
│   │                   #   cover_letter, tracker + strategies
│   ├── graph/          # ApplicationState + compiled P/E/V LangGraph + batch runner
│   ├── schemas.py      # Pydantic models for typed node hand-offs
│   ├── memory/         # Chroma persistent vector memory (site quirks + outcomes)
│   ├── guardrails/     # deterministic cover-letter entity-grounding check
│   └── db/             # SQLite persistence
├── eval/               # benchmark suite: fixtures, harness, scoring, run_benchmark.py
├── frontend/
│   ├── app.py          # main page: process single / batch, result tabs
│   ├── diff_utils.py   # word-level diff highlighting
│   └── pages/
│       └── 1_Applications_Dashboard.py
├── docs/               # current_architecture.md
├── .github/workflows/  # CI: tests + gated benchmark
├── data/               # SQLite DB + Chroma memory (gitignored, volume-mounted)
├── run_cli.py          # headless runner for testing
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup (local)

1. **Clone & enter**
   ```bash
   git clone <your-repo-url> job-agent && cd job-agent
   ```

2. **Create a virtualenv & install**
   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure keys**
   ```bash
   cp .env.example .env          # Windows: copy .env.example .env
   ```
   Edit `.env` and set at least one of `DEEPSEEK_API_KEY` / `GEMINI_API_KEY`.
   (Set both for primary + fallback.)

4. **Run the app**
   ```bash
   streamlit run frontend/app.py
   ```
   Open http://localhost:8501.

### CLI smoke test (no UI)
```bash
python run_cli.py --resume my_resume.txt --url https://example.com/job/123
python run_cli.py --resume my_resume.txt --urls urls.txt    # batch
```

---

## Usage

1. Paste your resume **once** in the sidebar (kept in session state).
2. **Single job:** paste a URL → **Process**. Review the four tabs:
   - **Job Analysis** — parsed requirements
   - **Match Score** — visual gauge + matching/missing/transferable skills
   - **Tailored Resume** — diff view (red strikethrough = removed, green = added)
   - **Cover Letter** — editable text area, copy via the code box, or download `.txt`
3. **Batch:** paste several URLs (one per line) → processed in parallel with a progress bar.
4. **Applications Dashboard** (left nav) — table of every job with company, role, match
   score, date, an editable **status** dropdown (To Apply / Applied / Interview /
   Rejected) and a **notes** field. Click **Save changes** to persist.

---

## Deployment

### Docker Compose (recommended)
```bash
cp .env.example .env   # fill in keys
docker compose up --build
```
App is served at http://localhost:8501. The SQLite DB is persisted to `./data` via a
bind mount, so applications survive restarts.

### Plain Docker
```bash
docker build -t job-application-agent .
docker run -p 8501:8501 --env-file .env -v "$(pwd)/data:/app/data" job-application-agent
```

### Notes for production
- Put the app behind a reverse proxy (nginx/Caddy) with TLS if exposing publicly.
- The DB is single-file SQLite — fine for personal use; swap the `backend/db` layer for
  Postgres if you need multi-user concurrency.
- Respect target sites' Terms of Service and `robots.txt` when scraping.

---

## How truthfulness is enforced
The **ResumeTailiorAgent** and **CoverLetterAgent** prompts explicitly forbid inventing
skills, employers, metrics, or outcomes — they may only **reframe experience already
present** in your resume. The diff view lets you verify every change before using it.

---

## Configuration reference

| Env var | Default | Purpose |
|---------|---------|---------|
| `DEEPSEEK_API_KEY` | — | Primary LLM key |
| `GEMINI_API_KEY` | — | Fallback LLM key |
| `DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek model id |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model id |
| `LLM_TIMEOUT` | `90` | Per-request timeout (s) |
| `JOB_AGENT_DB` | `data/applications.db` | SQLite path |
