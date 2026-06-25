"""Shared LangGraph state definition for the job application pipeline."""

from __future__ import annotations

from typing import Any, List, Optional, TypedDict


class TailoredBullet(TypedDict):
    """One rewritten resume bullet, keeping the original for diffing."""

    original: str
    rewritten: str
    rationale: str


class ApplicationState(TypedDict, total=False):
    """State threaded through every node of the application graph.

    ``total=False`` so partial updates returned by individual nodes are valid;
    LangGraph merges each node's returned dict into the running state.
    """

    # Input
    job_url: str
    resume_text: str

    # JobScraperAgent
    raw_job_text: str
    parsed_job: dict  # title, company, required_skills, responsibilities, nice_to_haves, salary

    # ResumeAnalysisAgent
    match_score: int  # 0-100
    matching_skills: List[str]
    missing_skills: List[str]
    transferable_experiences: List[str]

    # ResumeTailiorAgent
    tailored_bullets: List[TailoredBullet]

    # CoverLetterAgent
    cover_letter: str

    # TrackerAgent
    application_id: int
    status: str

    # --- Planner / Executor / Verifier bookkeeping (Phase 1) ---
    # The paste-JD path pre-seeds raw_job_text and sets this so the planner skips fetching.
    pasted: bool
    # Current PlannerDecision (as a dict; the planner constructs the Pydantic model).
    plan: dict
    # Number of extraction attempts made so far (scrape gate retries).
    scrape_attempts: int
    # Number of cover-letter generation attempts so far (judge gate retries).
    cover_letter_attempts: int
    # Most recent verifier verdicts (dicts of VerificationResult), for tracing/dashboard.
    scrape_verdict: dict
    cover_letter_verdict: dict
    # True when the deterministic grounded-template fallback replaced the LLM letter.
    cover_letter_fallback_used: bool
    # Strategy actually used for the successful (or final) extraction.
    strategy_used: str
    # Fetched HTML cached across retries so we re-parse instead of re-downloading.
    cached_html: str

    # Pipeline bookkeeping
    error: Optional[str]
