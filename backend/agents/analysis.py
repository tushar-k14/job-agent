"""ResumeAnalysisAgent.

Compares the user's resume against the parsed job and produces a match assessment:
matching skills, missing skills, transferable experiences, and a 0-100 score.
"""

from __future__ import annotations

import json
import logging

from ..llm import llm_json
from ..graph.state import ApplicationState

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are an experienced technical recruiter and resume analyst. Evaluate how "
    "well a candidate's resume matches a job. Be honest and specific. Base every "
    "judgement only on what the resume actually says. Respond with a single JSON "
    "object and nothing else."
)

_USER_TMPL = """Analyze this candidate against the job below.

Return JSON with EXACTLY these four keys — all four are required, use an empty list [] if nothing applies:
{{
  "match_score": integer,                  // 0-100 overall fit
  "matching_skills": [string],             // skills/tools/qualifications the resume clearly demonstrates that the job requires
  "missing_skills": [string],              // required skills/tools listed in the job that are NOT evidenced anywhere in the resume (never omit this key, even if the list is empty)
  "transferable_experiences": [string]     // resume experiences that map to this role even if the terminology differs
}}

Rules:
- Every required skill from the job must appear in either matching_skills or missing_skills.
- Do not put the same item in both lists.
- missing_skills must be present as a key even when the candidate is a strong match.

JOB (structured):
{job_json}

RESUME:
\"\"\"
{resume}
\"\"\""""


def analysis_node(state: ApplicationState) -> dict:
    if state.get("error"):
        return {}
    parsed_job = state.get("parsed_job", {})
    resume = state.get("resume_text", "")
    if not resume.strip():
        return {"error": "No resume text provided for analysis."}

    logger.info("Analyzing resume vs job: %s", parsed_job.get("title"))
    try:
        result = llm_json(
            _SYSTEM,
            _USER_TMPL.format(
                job_json=json.dumps(parsed_job, ensure_ascii=False, indent=2),
                resume=resume,
            ),
            temperature=0.2,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Resume analysis failed: {exc}"}

    score = result.get("match_score", 0)
    try:
        score = max(0, min(100, int(score)))
    except (TypeError, ValueError):
        score = 0

    return {
        "match_score": score,
        "matching_skills": result.get("matching_skills") or [],
        "missing_skills": result.get("missing_skills") or [],
        "transferable_experiences": result.get("transferable_experiences") or [],
    }
