"""Deterministic fallbacks for when verification fails repeatedly.

When the LLM-driven cover letter fails the quality gate on every retry (e.g. it keeps
fabricating), we don't ship an unverified letter. Instead we synthesize a conservative,
**template** cover letter from grounded facts only — the job title/company and the
candidate's matching skills (which by construction come from the resume). This is
guaranteed to pass the entity-grounding check and never invents anything.

It is intentionally plainer than an LLM draft: the point is a safe, honest floor, clearly
better than shipping a fabricated letter or nothing at all.
"""

from __future__ import annotations

from .grounding import check_cover_letter_grounding


def build_template_cover_letter(
    *,
    title: str,
    company: str,
    matching_skills: list[str],
    transferable: list[str] | None = None,
) -> str:
    title = title if title and title != "Unknown" else "this role"
    company = company if company and company != "Unknown" else "your team"
    skills = [s for s in (matching_skills or []) if s]
    transferable = [t for t in (transferable or []) if t]

    skills_clause = (
        f"my experience with {_oxford(skills)}"
        if skills
        else "the relevant experience on my resume"
    )
    transfer_clause = (
        f" I also bring {_oxford(transferable)}, which transfers directly to this work."
        if transferable
        else ""
    )

    return (
        f"Dear Hiring Team,\n\n"
        f"I am writing to apply for {title} at {company}. The role aligns closely with "
        f"{skills_clause}, and I am confident I can contribute from day one.\n\n"
        f"What draws me to {company} is the opportunity to apply and deepen these skills "
        f"on problems that matter.{transfer_clause}\n\n"
        f"I would welcome the chance to discuss how my background fits your needs. Thank "
        f"you for your consideration.\n\n"
        f"Sincerely,"
    )


def _oxford(items: list[str]) -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def grounded_fallback_letter(state: dict) -> str:
    """Build a template letter from the state and assert it is grounded."""
    parsed = state.get("parsed_job") or {}
    letter = build_template_cover_letter(
        title=parsed.get("title", ""),
        company=parsed.get("company", ""),
        matching_skills=state.get("matching_skills", []),
        transferable=state.get("transferable_experiences", []),
    )
    # Safety net: if somehow not grounded (e.g. a skill string contains an odd token),
    # fall back to the most minimal possible letter.
    result = check_cover_letter_grounding(letter, state.get("resume_text", ""), parsed)
    if not result.grounded:
        company = parsed.get("company") or "your team"
        title = parsed.get("title") or "this role"
        return (
            f"Dear Hiring Team,\n\nI am writing to apply for {title} at {company}. "
            f"Please find my qualifications in the attached resume; I believe my "
            f"background is a strong match for your needs.\n\nThank you for your "
            f"consideration.\n\nSincerely,"
        )
    return letter
