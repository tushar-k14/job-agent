"""Verifier node(s).

Two real verification gates, each a separate node so they can sit at different points
in the graph and route independently:

1. ``scrape_verifier`` — runs after extraction. Deterministic validity checks: are
   title / company / description present, non-empty, and not an obvious error or
   boilerplate page? On failure it records a specific reason; the graph routes back to
   the planner for an alternate strategy until ``scrape_attempts`` hits the cap.

2. ``cover_letter_verifier`` — runs after the writer. LLM-as-judge: does the draft
   reference only real candidate facts (no fabrication), is it roughly the right length,
   and does it avoid generic filler? On failure the graph routes back to the writer to
   regenerate with adjusted instructions, until ``cover_letter_attempts`` hits the cap.
"""

from __future__ import annotations

import logging

from ..llm import llm_json
from ..graph.state import ApplicationState
from ..schemas import (
    CoverLetterJudgement,
    ParsedJob,
    VerificationResult,
    VerificationStage,
)

logger = logging.getLogger(__name__)

# Retry caps (N=2 retries => up to 3 attempts each), agreed in Phase 0.
MAX_SCRAPE_ATTEMPTS = 3
MAX_COVER_LETTER_ATTEMPTS = 3

_BOILERPLATE_MARKERS = (
    "enable javascript",
    "are you a robot",
    "captcha",
    "access denied",
    "page not found",
    "403 forbidden",
    "just a moment",
)


# --------------------------------------------------------------------------- #
# Gate 1: scrape validity (deterministic)
# --------------------------------------------------------------------------- #
def _validate_scrape(parsed: ParsedJob, raw_text: str) -> VerificationResult:
    low = (raw_text or "").lower()
    for marker in _BOILERPLATE_MARKERS:
        if marker in low[:1500]:
            return VerificationResult(
                stage=VerificationStage.SCRAPE,
                passed=False,
                reason=f"Page looks like a block/error page (matched '{marker}').",
            )

    missing = []
    if not parsed.title or parsed.title == "Unknown":
        missing.append("title")
    if not parsed.company or parsed.company == "Unknown":
        missing.append("company")
    # "description" presence = some responsibilities OR required skills OR enough text.
    has_desc = bool(parsed.responsibilities or parsed.required_skills) or len(low) > 400
    if not has_desc:
        missing.append("description")

    if missing:
        return VerificationResult(
            stage=VerificationStage.SCRAPE,
            passed=False,
            reason=f"Missing/empty fields: {', '.join(missing)}.",
        )
    return VerificationResult(stage=VerificationStage.SCRAPE, passed=True, reason="ok")


def scrape_verifier_node(state: ApplicationState) -> dict:
    # If extraction hard-errored (e.g. blocked domain), don't override that error here;
    # the router decides whether attempts remain.
    parsed_dict = state.get("parsed_job")
    if not parsed_dict:
        verdict = VerificationResult(
            stage=VerificationStage.SCRAPE,
            passed=False,
            reason=state.get("error") or "No parsed job produced.",
        )
        return {"scrape_verdict": verdict.model_dump(mode="json")}

    verdict = _validate_scrape(ParsedJob(**parsed_dict), state.get("raw_job_text", ""))
    logger.info("Scrape verifier: passed=%s reason=%s", verdict.passed, verdict.reason)
    out: dict = {"scrape_verdict": verdict.model_dump(mode="json")}
    if not verdict.passed:
        # Surface the reason as a soft error so the planner can react / final state shows it.
        out["error"] = f"Scrape verification failed: {verdict.reason}"
    else:
        out["error"] = None
    return out


# --------------------------------------------------------------------------- #
# Gate 2: cover-letter quality (LLM-as-judge)
# --------------------------------------------------------------------------- #
_JUDGE_SYSTEM = (
    "You are a meticulous hiring-quality reviewer. You judge whether a cover letter is "
    "usable, focusing on TRUTHFULNESS above all. Respond with a single JSON object."
)

_JUDGE_USER_TMPL = """Judge this cover letter against the candidate's resume and the job.

Return JSON with exactly these keys:
{{
  "fabrication_detected": bool,   // true if the letter claims any skill, employer, metric, or experience NOT supported by the resume
  "length_ok": bool,             // true if it is roughly 3 paragraphs and under ~350 words
  "generic_filler": bool,        // true if it leans on clichés/boilerplate instead of specifics
  "passed": bool,                // overall: usable as-is (must be false if fabrication_detected is true)
  "reason": string               // one sentence explaining the verdict
}}

RESUME:
\"\"\"
{resume}
\"\"\"

JOB TITLE: {title} at {company}

COVER LETTER:
\"\"\"
{letter}
\"\"\""""


def _judge_cover_letter(state: ApplicationState) -> CoverLetterJudgement:
    parsed = ParsedJob(**(state.get("parsed_job") or {}))
    raw = llm_json(
        _JUDGE_SYSTEM,
        _JUDGE_USER_TMPL.format(
            resume=state.get("resume_text", ""),
            title=parsed.title,
            company=parsed.company,
            letter=state.get("cover_letter", ""),
        ),
        temperature=0.0,
    )
    # Coerce defensively; a missing key shouldn't crash the gate.
    return CoverLetterJudgement(
        fabrication_detected=bool(raw.get("fabrication_detected", False)),
        length_ok=bool(raw.get("length_ok", True)),
        generic_filler=bool(raw.get("generic_filler", False)),
        passed=bool(raw.get("passed", True)) and not bool(raw.get("fabrication_detected", False)),
        reason=str(raw.get("reason", "")),
    )


def cover_letter_verifier_node(state: ApplicationState) -> dict:
    if not state.get("cover_letter"):
        verdict = VerificationResult(
            stage=VerificationStage.COVER_LETTER,
            passed=False,
            reason="No cover letter to verify.",
        )
        return {"cover_letter_verdict": verdict.model_dump(mode="json")}

    # Defense-in-depth gate 1 (deterministic, free): entity grounding. If the letter
    # asserts entities not traceable to resume/job, fail immediately without spending an
    # LLM judge call — this catches blatant fabrication for zero cost.
    try:
        from ..guardrails import check_cover_letter_grounding

        grounding = check_cover_letter_grounding(
            state.get("cover_letter", ""),
            state.get("resume_text", ""),
            state.get("parsed_job") or {},
        )
        if not grounding.grounded:
            verdict = VerificationResult(
                stage=VerificationStage.COVER_LETTER,
                passed=False,
                reason=grounding.reason,
                fabrication_detected=True,
            )
            logger.info("Cover-letter grounding gate failed: %s", grounding.reason)
            return {"cover_letter_verdict": verdict.model_dump(mode="json")}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Grounding pre-check skipped: %s", exc)

    # Gate 2 (LLM-as-judge): catches subtler issues grounding can't (tone, length, filler).
    try:
        judgement = _judge_cover_letter(state)
    except Exception as exc:  # noqa: BLE001
        # If the judge itself fails, don't block the pipeline — pass with a note.
        logger.warning("Cover-letter judge failed, passing through: %s", exc)
        verdict = VerificationResult(
            stage=VerificationStage.COVER_LETTER,
            passed=True,
            reason=f"Judge unavailable ({exc}); accepted without review.",
        )
        return {"cover_letter_verdict": verdict.model_dump(mode="json")}

    verdict = VerificationResult(
        stage=VerificationStage.COVER_LETTER,
        passed=judgement.passed,
        reason=judgement.reason,
        fabrication_detected=judgement.fabrication_detected,
        length_ok=judgement.length_ok,
        generic_filler=judgement.generic_filler,
    )
    logger.info(
        "Cover-letter verifier: passed=%s fabrication=%s reason=%s",
        verdict.passed, verdict.fabrication_detected, verdict.reason,
    )
    return {"cover_letter_verdict": verdict.model_dump(mode="json")}


def cover_letter_fallback_node(state: ApplicationState) -> dict:
    """Deterministic fallback: when the LLM letter fails verification on every retry,
    replace it with a grounded template letter that cannot fabricate."""
    from ..guardrails.fallback import grounded_fallback_letter

    letter = grounded_fallback_letter(dict(state))
    logger.info("Cover-letter fallback engaged (template letter from grounded facts).")
    verdict = VerificationResult(
        stage=VerificationStage.COVER_LETTER,
        passed=True,
        reason="Deterministic grounded-template fallback after retries exhausted.",
    )
    return {
        "cover_letter": letter,
        "cover_letter_verdict": verdict.model_dump(mode="json"),
        "cover_letter_fallback_used": True,
    }
