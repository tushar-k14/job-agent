"""CoverLetterAgent.

Writes a focused 3-paragraph cover letter:
  1. Why this role
  2. Why this company
  3. Why you specifically
grounded in the parsed job + the candidate's real resume.
"""

from __future__ import annotations

import json
import logging

from ..llm import llm_complete
from ..graph.state import ApplicationState

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a professional career writer. Write authentic, specific cover letters "
    "in a confident but not arrogant voice. Ground every claim in the candidate's "
    "actual resume; never fabricate experience. Avoid clichés and filler."
)

_USER_TMPL = """Write a cover letter for this application. It MUST be exactly three paragraphs:

Paragraph 1 — Why this ROLE: what about the position's responsibilities excites the
candidate and how their background fits.
Paragraph 2 — Why this COMPANY: connect to the company (use only what is given/known;
do not invent specifics like funding rounds or product names you are unsure of).
Paragraph 3 — Why YOU specifically: the strongest, most relevant proof points from the
resume that make this candidate a standout.

Do not include a date or mailing address. You may open with "Dear Hiring Team," and
close with "Sincerely,". Keep it under ~320 words. Return only the letter text.

JOB (structured):
{job_json}

ANALYSIS:
- Matching skills: {matching}
- Transferable experiences: {transferable}

RESUME:
\"\"\"
{resume}
\"\"\""""


def _feedback_from_verdict(state: ApplicationState) -> str:
    """Turn the last cover-letter verdict into corrective instructions for a retry."""
    verdict = state.get("cover_letter_verdict")
    if not verdict:
        return ""
    notes = []
    if verdict.get("fabrication_detected"):
        notes.append(
            "The previous draft contained claims not supported by the resume. Use ONLY "
            "facts present in the resume; remove any invented skills, metrics, or experience."
        )
    if not verdict.get("length_ok", True):
        notes.append("The previous draft was the wrong length. Keep it to 3 paragraphs, under ~320 words.")
    if verdict.get("generic_filler"):
        notes.append("The previous draft was too generic. Replace clichés with specific, resume-grounded details.")
    if not notes and verdict.get("reason"):
        notes.append(f"Address this feedback: {verdict['reason']}")
    if not notes:
        return ""
    return "\n\nCORRECTIONS REQUIRED (the prior draft was rejected):\n- " + "\n- ".join(notes)


def cover_letter_node(state: ApplicationState) -> dict:
    if state.get("error"):
        return {}
    parsed_job = state.get("parsed_job", {})
    resume = state.get("resume_text", "")

    attempt = state.get("cover_letter_attempts", 0) + 1
    feedback = _feedback_from_verdict(state)
    logger.info("Writing cover letter for %s (attempt %s)", parsed_job.get("title"), attempt)
    try:
        letter = llm_complete(
            _SYSTEM,
            _USER_TMPL.format(
                job_json=json.dumps(parsed_job, ensure_ascii=False, indent=2),
                matching=", ".join(state.get("matching_skills", [])) or "n/a",
                transferable=", ".join(state.get("transferable_experiences", []))
                or "n/a",
                resume=resume,
            )
            + feedback,
            temperature=0.6 if attempt == 1 else 0.4,  # tighten on retry
            max_tokens=800,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Cover letter generation failed: {exc}"}

    return {"cover_letter": letter.strip(), "cover_letter_attempts": attempt}
