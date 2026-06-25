"""Deterministic pass/fail scoring of a task's final state against its expectations."""

from __future__ import annotations

from backend.guardrails import check_cover_letter_grounding
from .fixtures import BenchmarkTask


def score_task(task: BenchmarkTask, state: dict) -> list[str]:
    """Return a list of failure messages. Empty list == task passed."""
    exp = task.expect
    failures: list[str] = []

    scrape_verdict = state.get("scrape_verdict") or {}
    actual_scrape_passed = bool(scrape_verdict.get("passed"))
    if actual_scrape_passed != exp.scrape_passed:
        failures.append(
            f"scrape_passed: expected {exp.scrape_passed}, got {actual_scrape_passed} "
            f"(reason: {scrape_verdict.get('reason')})"
        )

    if exp.has_error is not None:
        has_error = bool(state.get("error"))
        if has_error != exp.has_error:
            failures.append(f"has_error: expected {exp.has_error}, got {has_error} ({state.get('error')})")

    # attempts within bounds
    attempts = state.get("scrape_attempts", 0)
    if attempts < exp.min_scrape_attempts:
        failures.append(f"scrape_attempts {attempts} < min {exp.min_scrape_attempts}")
    if attempts > exp.max_scrape_attempts:
        failures.append(f"scrape_attempts {attempts} > max {exp.max_scrape_attempts}")

    # final strategy
    if exp.final_strategy is not None:
        used = state.get("strategy_used")
        if used != exp.final_strategy:
            failures.append(f"final_strategy: expected {exp.final_strategy}, got {used}")

    # enrichment decision (read from the last plan)
    if exp.needs_enrichment is not None:
        plan = state.get("plan") or {}
        got = bool(plan.get("needs_enrichment"))
        if got != exp.needs_enrichment:
            failures.append(f"needs_enrichment: expected {exp.needs_enrichment}, got {got}")

    # blocked domain → error should mention scraping/blocking and zero successful parse
    if exp.is_blocked_domain:
        err = (state.get("error") or "").lower()
        if "scrap" not in err and "block" not in err:
            failures.append(f"blocked domain: error doesn't indicate blocking ({state.get('error')})")

    # deterministic fallback engaged?
    if exp.cover_letter_fallback is not None:
        used = bool(state.get("cover_letter_fallback_used"))
        if used != exp.cover_letter_fallback:
            failures.append(f"cover_letter_fallback: expected {exp.cover_letter_fallback}, got {used}")

    # deterministic grounding check on the produced cover letter
    if exp.cover_letter_grounded is not None:
        letter = state.get("cover_letter", "")
        result = check_cover_letter_grounding(letter, task.resume, state.get("parsed_job") or {})
        if result.grounded != exp.cover_letter_grounded:
            failures.append(
                f"cover_letter_grounded: expected {exp.cover_letter_grounded}, "
                f"got {result.grounded} ({result.reason})"
            )

    return failures
