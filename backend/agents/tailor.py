"""ResumeTailiorAgent.

Rewrites existing resume bullet points so they speak more directly to the job,
WITHOUT inventing experience. The model is instructed to only reframe what is
already present. Returns original/rewritten/rationale triples for a diff view.
"""

from __future__ import annotations

import json
import logging

from ..llm import llm_json
from ..graph.state import ApplicationState

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are an expert resume editor. You rewrite a candidate's existing resume "
    "bullet points so they better align with a specific job, while preserving "
    "strict truthfulness. CRITICAL RULES:\n"
    "- Only reframe, reword, or re-emphasize experience that is ALREADY in the "
    "original bullet. Never invent skills, tools, metrics, employers, or outcomes.\n"
    "- If a bullet has no numbers, do not fabricate numbers.\n"
    "- Keep each rewrite concise and results-oriented.\n"
    "Respond with a single JSON object and nothing else."
)

_USER_TMPL = """Select up to {n} of the most relevant resume bullet points and rewrite each
to better match the job. Use language and keywords from the job where the candidate's
real experience genuinely supports it.

Return JSON with exactly this shape:
{{
  "tailored_bullets": [
    {{
      "original": string,    // the exact original bullet text from the resume
      "rewritten": string,   // the improved version (truthful reframing only)
      "rationale": string    // 1 sentence: why this rewrite helps for THIS job
    }}
  ]
}}

JOB (structured):
{job_json}

RESUME:
\"\"\"
{resume}
\"\"\""""

_MAX_BULLETS = 6


def tailor_node(state: ApplicationState) -> dict:
    if state.get("error"):
        return {}
    parsed_job = state.get("parsed_job", {})
    resume = state.get("resume_text", "")

    logger.info("Tailoring resume bullets for: %s", parsed_job.get("title"))
    try:
        result = llm_json(
            _SYSTEM,
            _USER_TMPL.format(
                n=_MAX_BULLETS,
                job_json=json.dumps(parsed_job, ensure_ascii=False, indent=2),
                resume=resume,
            ),
            temperature=0.5,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Resume tailoring failed: {exc}"}

    bullets = result.get("tailored_bullets") or []
    cleaned = [
        {
            "original": b.get("original", ""),
            "rewritten": b.get("rewritten", ""),
            "rationale": b.get("rationale", ""),
        }
        for b in bullets
        if isinstance(b, dict) and b.get("original") and b.get("rewritten")
    ]
    return {"tailored_bullets": cleaned}
