"""Benchmark fixtures: 24 fixed tasks with explicit, automatable pass/fail criteria.

Each task is a ``BenchmarkTask``. A task carries:
- the input (a job URL + canned HTML, or pasted text),
- a canned LLM script (so the MOCK tier is deterministic and free),
- the candidate resume,
- and ``expect`` assertions describing the correct AGENT BEHAVIOR.

Categories (deliberately include tricky edge cases):
- good_*       : well-formed pages on various ATS layouts → scrape passes first try
- bad_*        : captcha / empty / error pages → scrape verifier fails → retries → recorded failure
- thin_*       : sparse JD → planner flags enrichment
- paste_*      : pasted text path → PASTE_TEXT strategy
- blocked_*    : known-blocked domains → hard error, no fetch
- fabrication_*: cover letter asserts entities absent from source → grounding catches it
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

RESUME = (
    "Jane Doe — Software Engineer. 4 years experience. "
    "Built REST APIs with FastAPI and Python at TechStartup. "
    "Designed PostgreSQL schemas. Containerized services with Docker, deployed on AWS. "
    "Wrote pytest suites. Collaborated with React frontend team. "
    "Skills: Python, FastAPI, PostgreSQL, Docker, AWS, pytest, Git."
)

# A reusable "good" parsed-job the mock LLM returns for well-formed pages.
_GOOD_PARSED = {
    "title": "Backend Engineer",
    "company": "Acme Corp",
    "required_skills": ["Python", "FastAPI", "PostgreSQL"],
    "responsibilities": ["Build and maintain REST APIs", "Own backend services"],
    "nice_to_haves": ["Docker", "AWS"],
    "salary": "$120k–$150k",
}

_EMPTY_PARSED = {
    "title": None, "company": None,
    "required_skills": [], "responsibilities": [], "nice_to_haves": [], "salary": None,
}

_THIN_PARSED = {
    "title": "Engineer", "company": "Startup",
    "required_skills": [], "responsibilities": [], "nice_to_haves": [], "salary": None,
}

_GOOD_COVER = (
    "Dear Hiring Team,\n\nI am excited to apply for the Backend Engineer role at Acme "
    "Corp, where building and owning REST APIs is central. I have built REST APIs with "
    "FastAPI and Python.\n\nAcme Corp's focus resonates with my background.\n\nMy work "
    "with PostgreSQL, Docker, and AWS makes me a strong fit.\n\nSincerely,"
)

# A cover letter that fabricates entities not in the resume/job (Google, Kubernetes, TensorFlow).
_FABRICATED_COVER = (
    "Dear Hiring Team,\n\nI am excited about Acme Corp. At Google I led a Kubernetes "
    "migration and used TensorFlow extensively.\n\nYour mission inspires me.\n\nMy "
    "experience makes me a fit.\n\nSincerely,"
)

_ANALYSIS = {
    "match_score": 82,
    "matching_skills": ["Python", "FastAPI", "PostgreSQL"],
    "missing_skills": [],
    "transferable_experiences": ["React collaboration"],
}

_JUDGE_PASS = {
    "fabrication_detected": False, "length_ok": True, "generic_filler": False,
    "passed": True, "reason": "ok",
}
_JUDGE_FAIL = {
    "fabrication_detected": True, "length_ok": True, "generic_filler": False,
    "passed": False, "reason": "claims unsupported experience",
}


def _good_html(title="Backend Engineer", company="Acme Corp", container="main"):
    body = (
        f"<h1>{title}</h1><h2>{company}</h2>"
        "<p>We are hiring. Responsibilities: build and maintain REST APIs, own backend "
        "services across the engineering organization. Required: Python, FastAPI, "
        "PostgreSQL. Nice to have: Docker, AWS. Salary: $120k-$150k.</p>"
    )
    if container == "main":
        return f"<html><body><main>{body}</main></body></html>"
    if container == "greenhouse":
        return f"<html><body><div id='content'>{body}</div></body></html>"
    if container == "lever":
        return f"<html><body><div class='posting'>{body}</div></body></html>"
    return f"<html><body>{body}</body></html>"


@dataclass
class Expect:
    """Expected agent behavior for a task (all automatable, deterministic)."""

    scrape_passed: bool
    # Strategy expected to be used on the FINAL successful attempt (or first if paste).
    final_strategy: Optional[str] = None
    needs_enrichment: Optional[bool] = None
    has_error: Optional[bool] = None
    min_scrape_attempts: int = 1
    max_scrape_attempts: int = 3
    cover_letter_grounded: Optional[bool] = None  # deterministic grounding check
    is_blocked_domain: bool = False
    cover_letter_fallback: Optional[bool] = None  # deterministic fallback engaged?


@dataclass
class BenchmarkTask:
    id: str
    category: str
    resume: str
    expect: Expect
    # URL path:
    url: str = ""
    html: str = ""
    # Paste path:
    pasted_text: str = ""
    # Canned LLM responses (mock tier):
    parsed_job: dict = field(default_factory=dict)
    analysis: dict = field(default_factory=lambda: dict(_ANALYSIS))
    cover_letter: str = _GOOD_COVER
    judge: dict = field(default_factory=lambda: dict(_JUDGE_PASS))
    # If extraction should fail to produce a usable job on the first k attempts:
    fail_extraction_until_attempt: int = 0  # 0 = never force-fail


def all_tasks() -> list[BenchmarkTask]:
    tasks: list[BenchmarkTask] = []

    # ---- good pages, various ATS layouts (scrape passes first try) ----
    for i, container in enumerate(["main", "greenhouse", "lever", "generic"], 1):
        tasks.append(BenchmarkTask(
            id=f"good_{container}",
            category="good",
            url=f"https://careers.example{i}.com/job/{i}",
            html=_good_html(container=container),
            parsed_job=dict(_GOOD_PARSED),
            resume=RESUME,
            expect=Expect(
                scrape_passed=True,
                needs_enrichment=False,
                has_error=False,
                min_scrape_attempts=1, max_scrape_attempts=1,
                cover_letter_grounded=True,
            ),
        ))

    # ---- good page but with a salary-less / minimal-but-valid posting ----
    tasks.append(BenchmarkTask(
        id="good_no_salary",
        category="good",
        url="https://careers.example9.com/job/9",
        html=_good_html(),
        parsed_job={**_GOOD_PARSED, "salary": None},
        resume=RESUME,
        expect=Expect(scrape_passed=True, has_error=False, cover_letter_grounded=True),
    ))

    # ---- bad pages: empty/error/captcha → verifier fails → retries → recorded failure ----
    tasks.append(BenchmarkTask(
        id="bad_empty_page",
        category="bad",
        url="https://broken.example.com/job/1",
        html="<html><body></body></html>",
        parsed_job=dict(_EMPTY_PARSED),
        resume=RESUME,
        fail_extraction_until_attempt=99,  # always empty
        expect=Expect(
            scrape_passed=False, has_error=True,
            min_scrape_attempts=3, max_scrape_attempts=3,
        ),
    ))
    tasks.append(BenchmarkTask(
        id="bad_captcha_wall",
        category="bad",
        url="https://protected.example.com/job/2",
        html="<html><body><h1>Just a moment</h1><p>Please complete the CAPTCHA to continue.</p></body></html>",
        parsed_job=dict(_EMPTY_PARSED),
        resume=RESUME,
        fail_extraction_until_attempt=99,
        expect=Expect(
            scrape_passed=False, has_error=True,
            min_scrape_attempts=3, max_scrape_attempts=3,
        ),
    ))
    tasks.append(BenchmarkTask(
        id="bad_missing_company",
        category="bad",
        url="https://partial.example.com/job/3",
        html=_good_html(),
        parsed_job={**_GOOD_PARSED, "company": None},
        resume=RESUME,
        fail_extraction_until_attempt=99,
        expect=Expect(scrape_passed=False, has_error=True, min_scrape_attempts=3),
    ))

    # ---- recovery: fails first attempt, succeeds on retry (escalated strategy) ----
    tasks.append(BenchmarkTask(
        id="recover_on_second_attempt",
        category="recovery",
        url="https://flaky.example.com/job/1",
        html=_good_html(),
        parsed_job=dict(_GOOD_PARSED),
        resume=RESUME,
        fail_extraction_until_attempt=2,  # attempt 1 empty, attempt 2 good
        expect=Expect(
            scrape_passed=True, has_error=False,
            min_scrape_attempts=2, max_scrape_attempts=2,
            final_strategy="generic_text",
            cover_letter_grounded=True,
        ),
    ))
    tasks.append(BenchmarkTask(
        id="recover_on_third_attempt",
        category="recovery",
        url="https://veryflaky.example.com/job/1",
        html=_good_html(),
        parsed_job=dict(_GOOD_PARSED),
        resume=RESUME,
        fail_extraction_until_attempt=3,
        expect=Expect(
            scrape_passed=True, has_error=False,
            min_scrape_attempts=3, max_scrape_attempts=3,
            final_strategy="llm_from_raw_html",
            cover_letter_grounded=True,
        ),
    ))

    # ---- thin JD → planner flags enrichment ----
    # Cover letter is intentionally grounded against the THIN parsed job ("Startup")
    # so the grounding assertion is consistent with this task's source material.
    tasks.append(BenchmarkTask(
        id="thin_jd_enrichment",
        category="thin",
        url="https://sparse.example.com/job/1",
        html="<html><body><main>Engineer at Startup. Apply now.</main></body></html>",
        parsed_job=dict(_THIN_PARSED),
        resume=RESUME,
        cover_letter=(
            "Dear Hiring Team,\n\nI am excited about the Engineer role at Startup. I have "
            "built REST APIs with Python and FastAPI.\n\nStartup's mission resonates with "
            "me.\n\nMy Python and Docker experience makes me a strong fit.\n\nSincerely,"
        ),
        expect=Expect(
            scrape_passed=True, has_error=False,
            needs_enrichment=True,
            cover_letter_grounded=True,
        ),
    ))

    # ---- paste path ----
    tasks.append(BenchmarkTask(
        id="paste_basic",
        category="paste",
        pasted_text=(
            "Backend Engineer at Acme Corp. Build REST APIs. Required: Python, FastAPI, "
            "PostgreSQL. Nice: Docker, AWS."
        ),
        parsed_job=dict(_GOOD_PARSED),
        resume=RESUME,
        expect=Expect(
            scrape_passed=True, has_error=False,
            final_strategy="paste_text",
            min_scrape_attempts=1, max_scrape_attempts=1,
            cover_letter_grounded=True,
        ),
    ))
    tasks.append(BenchmarkTask(
        id="paste_thin",
        category="paste",
        pasted_text="Engineer at Startup.",
        parsed_job=dict(_THIN_PARSED),
        resume=RESUME,
        expect=Expect(
            scrape_passed=True, has_error=False,
            final_strategy="paste_text",
            needs_enrichment=True,
        ),
    ))

    # ---- blocked domains → hard error, no fetch ----
    for i, domain in enumerate([
        "https://www.linkedin.com/jobs/view/123",
        "https://www.indeed.com/viewjob?jk=abc",
        "https://www.glassdoor.com/job/xyz",
    ], 1):
        tasks.append(BenchmarkTask(
            id=f"blocked_{i}",
            category="blocked",
            url=domain,
            html="",
            parsed_job=dict(_EMPTY_PARSED),
            resume=RESUME,
            expect=Expect(
                scrape_passed=False, has_error=True, is_blocked_domain=True,
                min_scrape_attempts=1,
            ),
        ))

    # ---- fabrication: letter asserts ungrounded entities → grounding catches it ----
    tasks.append(BenchmarkTask(
        id="fabrication_detected",
        category="fabrication",
        url="https://careers.examplef.com/job/1",
        html=_good_html(),
        parsed_job=dict(_GOOD_PARSED),
        resume=RESUME,
        cover_letter=_FABRICATED_COVER,
        judge=dict(_JUDGE_FAIL),
        expect=Expect(
            scrape_passed=True, has_error=False,
            # Phase 4: fabrication is not just detected but REMEDIATED — retries exhaust,
            # the deterministic fallback engages, and the FINAL letter is grounded.
            cover_letter_fallback=True,
            cover_letter_grounded=True,
        ),
    ))
    tasks.append(BenchmarkTask(
        id="fabrication_clean_passes",
        category="fabrication",
        url="https://careers.examplef.com/job/2",
        html=_good_html(),
        parsed_job=dict(_GOOD_PARSED),
        resume=RESUME,
        cover_letter=_GOOD_COVER,
        expect=Expect(
            scrape_passed=True, has_error=False,
            cover_letter_grounded=True,
        ),
    ))

    # ---- a few more good variants to reach ~24 and exercise different resumes ----
    tasks.append(BenchmarkTask(
        id="good_ml_role",
        category="good",
        url="https://careers.exampleml.com/job/1",
        html=_good_html(title="ML Engineer", company="DataCo"),
        parsed_job={
            "title": "ML Engineer", "company": "DataCo",
            "required_skills": ["Python", "PyTorch"], "responsibilities": ["Train models daily"],
            "nice_to_haves": ["AWS"], "salary": None,
        },
        resume=RESUME + " Also used PyTorch for a side project.",
        cover_letter=(
            "Dear Hiring Team,\n\nI am drawn to the ML Engineer role at DataCo. I have "
            "used Python and PyTorch.\n\nDataCo's work excites me.\n\nMy Python "
            "background fits.\n\nSincerely,"
        ),
        expect=Expect(scrape_passed=True, has_error=False, cover_letter_grounded=True),
    ))
    for i in range(1, 5):
        tasks.append(BenchmarkTask(
            id=f"good_extra_{i}",
            category="good",
            url=f"https://careers.extra{i}.com/job/{i}",
            html=_good_html(),
            parsed_job=dict(_GOOD_PARSED),
            resume=RESUME,
            expect=Expect(scrape_passed=True, has_error=False, cover_letter_grounded=True),
        ))

    # ---- paste-path fabrication (grounding must work regardless of input path) ----
    tasks.append(BenchmarkTask(
        id="fabrication_paste",
        category="fabrication",
        pasted_text="Backend Engineer at Acme Corp. Required: Python, FastAPI.",
        parsed_job=dict(_GOOD_PARSED),
        resume=RESUME,
        cover_letter=_FABRICATED_COVER,
        judge=dict(_JUDGE_FAIL),
        expect=Expect(
            scrape_passed=True, has_error=False,
            final_strategy="paste_text",
            cover_letter_fallback=True,
            cover_letter_grounded=True,
        ),
    ))

    return tasks
