# 🤖 Autonomous Job Application Agent

A LangGraph + Streamlit application that turns a job-posting URL and your resume into
a complete, tailored application package: structured job analysis, a fit score,
truthfully-reframed resume bullets, and a custom cover letter — all tracked in a
SQLite-backed dashboard.

LLM calls use **DeepSeek** as the primary provider with an automatic fallback to the
**free Gemini** tier, so a single missing/rate-limited key won't stop the pipeline.

---

## Architecture

A LangGraph state machine threads a single `ApplicationState` through five agent nodes:

```
job_url ─▶ ┌───────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────┐   ┌──────────────┐
           │ JobScraper    │──▶│ ResumeAnalysis   │──▶│ ResumeTailior    │──▶│ CoverLetter     │──▶│ Tracker      │──▶ SQLite
           │ (requests +   │   │ (match score,    │   │ (truthful bullet │   │ (3-paragraph    │   │ (persist +   │
           │  BeautifulSoup│   │  gaps, transfers)│   │  reframing)      │   │  letter)        │   │  set status) │
           │  + LLM extract)│  └──────────────────┘   └──────────────────┘   └─────────────────┘   └──────────────┘
           └───────────────┘
```

| Node | Responsibility |
|------|----------------|
| **JobScraperAgent** | Fetches the URL, strips boilerplate, LLM-extracts title, company, required skills, responsibilities, nice-to-haves, salary |
| **ResumeAnalysisAgent** | Compares resume ↔ job → matching skills, missing skills, transferable experiences, **match score 0–100** |
| **ResumeTailiorAgent** | Rewrites existing bullets to fit the job — **reframes only, never invents** experience |
| **CoverLetterAgent** | Writes a 3-paragraph letter: *why the role*, *why the company*, *why you* |
| **TrackerAgent** | Saves the full package to SQLite and stamps a status |

`ApplicationState` (see [`backend/graph/state.py`](backend/graph/state.py)) carries:
`job_url, raw_job_text, parsed_job, resume_text, match_score, tailored_bullets,
cover_letter, application_id, status` (+ analysis fields and an `error` channel).

### Batch mode (map-reduce)
Multiple URLs (one per line) are fanned out across a thread pool — each URL runs its
own isolated copy of the graph concurrently — then reduced into a list, with a live
Streamlit progress bar. See [`run_batch`](backend/graph/pipeline.py).

---

## Repo layout (monorepo)

```
job-agent/
├── backend/
│   ├── llm/            # DeepSeek primary + Gemini fallback client
│   ├── agents/         # the 5 agent nodes
│   ├── graph/          # ApplicationState + compiled LangGraph + batch runner
│   └── db/             # SQLite persistence
├── frontend/
│   ├── app.py          # main page: process single / batch, result tabs
│   ├── diff_utils.py   # word-level diff highlighting
│   └── pages/
│       └── 1_Applications_Dashboard.py
├── data/               # SQLite DB lives here (gitignored, volume-mounted)
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
