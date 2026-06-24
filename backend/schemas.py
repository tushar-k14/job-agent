"""Pydantic models for typed hand-offs between the planner, executor, and verifier.

These replace loose dict-key passing between graph nodes. Every structured decision or
result that crosses a node boundary is one of these models, so a shape mismatch fails
loudly at construction time instead of silently producing a wrong key downstream.

The node functions still return plain dicts into the LangGraph ``ApplicationState``
(LangGraph merges dicts), but the *values* of the agentic keys are these models, and
each node validates its inputs by re-constructing the model from state.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Extraction strategy
# --------------------------------------------------------------------------- #
class ExtractionStrategy(str, Enum):
    """How the executor should try to turn a page into structured job fields.

    Ordered roughly from cheapest/most-specific to most-general fallback. The planner
    picks the first strategy; the verifier escalates to later ones on failure.
    """

    # Pre-seeded raw text (paste-JD path) — no fetch, go straight to LLM extraction.
    PASTE_TEXT = "paste_text"
    # Structured HTML: try known job-board container selectors (greenhouse/lever/workday).
    SELECTOR_STRUCTURED = "selector_structured"
    # Generic: strip boilerplate tags, hand all visible text to the LLM (today's default).
    GENERIC_TEXT = "generic_text"
    # Last resort: feed a slice of the raw HTML itself to the LLM (handles odd markup).
    LLM_FROM_RAW_HTML = "llm_from_raw_html"

    @classmethod
    def escalation_order(cls) -> List["ExtractionStrategy"]:
        """Default order the planner walks through across retries (excluding PASTE_TEXT)."""
        return [cls.SELECTOR_STRUCTURED, cls.GENERIC_TEXT, cls.LLM_FROM_RAW_HTML]


class PlannerDecision(BaseModel):
    """The planner's chosen plan for the current attempt."""

    strategy: ExtractionStrategy
    needs_enrichment: bool = Field(
        default=False,
        description="True when the JD is too thin to tailor confidently and the executor "
        "should pull supplementary company/role context.",
    )
    reason: str = Field(default="", description="Human-readable justification (for tracing).")
    attempt: int = Field(default=1, ge=1, description="1-based attempt number for this run.")


# --------------------------------------------------------------------------- #
# Parsed job (structured extraction result)
# --------------------------------------------------------------------------- #
class ParsedJob(BaseModel):
    """Structured job posting. Mirrors the legacy ``parsed_job`` dict shape exactly so
    downstream analysis/tailor/writer code keeps working unchanged."""

    title: str = "Unknown"
    company: str = "Unknown"
    required_skills: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    nice_to_haves: List[str] = Field(default_factory=list)
    salary: Optional[str] = None

    @classmethod
    def normalized(cls, raw: dict) -> "ParsedJob":
        """Build from a possibly-messy LLM dict, coercing nulls to defaults."""
        raw = raw or {}
        return cls(
            title=raw.get("title") or "Unknown",
            company=raw.get("company") or "Unknown",
            required_skills=raw.get("required_skills") or [],
            responsibilities=raw.get("responsibilities") or [],
            nice_to_haves=raw.get("nice_to_haves") or [],
            salary=raw.get("salary"),
        )


class ExtractionResult(BaseModel):
    """What the executor produced for an extraction attempt."""

    raw_text: str = ""
    parsed_job: Optional[ParsedJob] = None
    strategy_used: Optional[ExtractionStrategy] = None
    error: Optional[str] = None


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
class VerificationStage(str, Enum):
    SCRAPE = "scrape"
    COVER_LETTER = "cover_letter"


class VerificationResult(BaseModel):
    """Outcome of a verifier gate. ``passed`` drives the conditional graph edge."""

    stage: VerificationStage
    passed: bool
    reason: str = Field(default="", description="Specific failure reason, fed back to planner.")
    # For the LLM-as-judge cover-letter gate:
    fabrication_detected: bool = False
    length_ok: bool = True
    generic_filler: bool = False


class CoverLetterJudgement(BaseModel):
    """Raw LLM-as-judge output for the cover-letter quality gate (parsed from JSON)."""

    fabrication_detected: bool = Field(
        description="True if the letter claims experience/skills not supported by the resume."
    )
    length_ok: bool = Field(description="True if roughly 3 paragraphs / under ~350 words.")
    generic_filler: bool = Field(description="True if it leans on clichés/boilerplate.")
    passed: bool = Field(description="Overall verdict: usable as-is.")
    reason: str = Field(default="", description="One sentence explaining the verdict.")
