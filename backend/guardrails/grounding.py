"""Deterministic fabrication check for cover letters.

The idea (no LLM required): every *salient named entity* the cover letter mentions —
companies, technologies/skills, proper-noun tokens — must trace back to the source
material the letter is allowed to draw on (the candidate's resume + the parsed job).
Anything the letter asserts that does NOT appear in source is a candidate fabrication.

This is intentionally conservative and explainable:
- It compares against a UNION of resume text + parsed job fields (the legitimate sources).
- It only flags *capitalized multi-letter tokens* and a curated tech-skill vocabulary, so
  ordinary prose ("I am excited to...") never trips it.
- A small stop-list removes sentence-start words and salutations that are capitalized for
  grammar, not because they're entities.

Used in two places:
1. The benchmark, as a deterministic, automatable "no fabrication" pass/fail.
2. At runtime (Phase 4) as a cheap pre-check BEFORE the LLM-as-judge — if grounding
   already finds an ungrounded entity, we can reject/regenerate without spending a judge
   call. Defense in depth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# Tokens that are capitalized for grammar/politeness, not because they're entities.
_STOPWORDS = {
    "Dear", "Hiring", "Team", "Sincerely", "Regards", "Best", "I", "I'm", "I've",
    "My", "The", "A", "An", "As", "At", "In", "On", "Of", "To", "For", "With", "And",
    "This", "That", "These", "Those", "It", "We", "Our", "Your", "You", "Their",
    "Manager", "Sir", "Madam", "Thank", "Thanks", "Yours", "Faithfully", "Role",
    "Company", "Position", "Engineer", "Developer", "Senior", "Junior", "Lead",
    "Please", "Find", "What", "Why", "How", "When", "Where", "Who", "Confident",
    "Writing", "Experience", "Opportunity", "Background", "Qualifications",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
}

# A curated set of common technologies so we catch lowercase tech mentions too
# (e.g. "kubernetes", "pytorch") that the capitalized-token rule would miss.
_TECH_VOCAB = {
    "python", "java", "javascript", "typescript", "go", "golang", "rust", "ruby",
    "c++", "c#", "scala", "kotlin", "swift", "php", "perl", "r",
    "react", "angular", "vue", "svelte", "node", "nodejs", "django", "flask",
    "fastapi", "spring", "rails", "express", "nextjs", "nest",
    "postgresql", "postgres", "mysql", "mongodb", "redis", "sqlite", "cassandra",
    "elasticsearch", "kafka", "rabbitmq", "celery", "graphql", "grpc", "rest",
    "docker", "kubernetes", "k8s", "terraform", "ansible", "jenkins", "circleci",
    "aws", "gcp", "azure", "lambda", "s3", "ec2", "ecs", "eks",
    "pytorch", "tensorflow", "keras", "sklearn", "pandas", "numpy", "spark",
    "hadoop", "airflow", "dbt", "snowflake", "databricks", "tableau",
    "git", "github", "gitlab", "linux", "kubernetes", "prometheus", "grafana",
}

# Generic technical vocabulary that is NOT a fabricatable entity. Saying "API design"
# or "REST endpoints" asserts no specific unverifiable experience the way naming a
# company (Google) or a specific tool the candidate lacks (Kubernetes) would. Excluding
# these avoids false-positive fabrication flags on ordinary engineering prose.
_GENERIC_TECH = {
    "api", "apis", "rest", "restful", "sql", "nosql", "http", "https", "json", "xml",
    "html", "css", "ci", "cd", "cicd", "sdk", "cli", "ui", "ux", "orm", "crud", "mvc",
    "tcp", "udp", "ip", "dns", "ssl", "tls", "url", "uri", "saas", "paas", "iaas",
    "etl", "ml", "ai", "llm", "nlp", "db", "ide", "os", "vm", "vpc", "cdn", "qa",
    "oop", "tdd", "bdd", "pr", "prs", "mvp", "poc", "kpi", "sla", "b2b", "b2c",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9.+#-]*")
_CAP_TOKEN_RE = re.compile(r"\b([A-Z][A-Za-z0-9.+#&-]{1,})\b")


def _normalize(tok: str) -> str:
    return tok.lower().strip(".,;:!?()[]\"'")


def _singularize(tok: str) -> str:
    """Cheap plural→singular: 'apis'→'api', 'schemas'→'schema'. Guards short tokens."""
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _is_sentence_initial(text: str, start: int) -> bool:
    """True if the token at ``start`` is the first word of a sentence (so its capital is
    grammatical, not a proper-noun signal). Looks back past whitespace for a sentence
    boundary (start-of-text, newline, or . ! ? : ; " ( ‑ )."""
    i = start - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    if i < 0:
        return True
    return text[i] in ".!?:;\n\r\"'([—–-•*"


def extract_candidate_entities(text: str) -> set[str]:
    """Salient entities a letter might assert: mid-sentence proper nouns + known tech terms.

    Two rules keep this high-precision (few false positives, so it doesn't wrongly trip
    the fallback):
    - A capitalized word is an entity candidate only if it appears MID-sentence — a
      sentence-initial capital ("Ensuring…", "With…") is grammar, not an entity.
    - Generic technical vocabulary (API, REST, SQL, …) is never an entity — you can't
      "fabricate" saying "API design".
    The LLM-as-judge gate remains the backstop for subtler cases this misses.
    """
    entities: set[str] = set()

    for m in _CAP_TOKEN_RE.finditer(text):
        tok = m.group(1)
        if tok in _STOPWORDS:
            continue
        if _is_sentence_initial(text, m.start()):
            continue
        norm = _normalize(tok)
        if norm in _GENERIC_TECH or _singularize(norm) in _GENERIC_TECH:
            continue
        if len(norm) >= 2:
            entities.add(norm)

    # Known specific technologies regardless of case/position (kubernetes, pytorch, …).
    for m in _TOKEN_RE.finditer(text):
        norm = _normalize(m.group(0))
        if norm in _GENERIC_TECH:  # generic vocab is never a fabricatable entity
            continue
        if norm in _TECH_VOCAB:
            entities.add(norm)

    return entities


def _source_vocabulary(sources: Iterable[str]) -> set[str]:
    """All tokens in the source material, stored in both surface and singular form so
    plural/singular differences ('APIs' in resume vs 'API' in letter) still match."""
    vocab: set[str] = set()
    for src in sources:
        if not src:
            continue
        for m in _TOKEN_RE.finditer(src):
            norm = _normalize(m.group(0))
            vocab.add(norm)
            vocab.add(_singularize(norm))
    return vocab


def _is_grounded(entity: str, source_vocab: set[str]) -> bool:
    """Entity is grounded if it (or its singular form) appears in source."""
    return entity in source_vocab or _singularize(entity) in source_vocab


@dataclass
class GroundingResult:
    grounded: bool
    ungrounded_entities: list[str] = field(default_factory=list)
    checked_entities: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if self.grounded:
            return "All asserted entities trace to source material."
        return "Ungrounded entities (not in resume or job): " + ", ".join(
            sorted(self.ungrounded_entities)
        )


def check_cover_letter_grounding(
    cover_letter: str,
    resume_text: str,
    parsed_job: dict | None = None,
    *,
    extra_allowed: Iterable[str] = (),
) -> GroundingResult:
    """Return whether every salient entity in the letter is grounded in source.

    ``parsed_job`` fields (title, company, skills, responsibilities, nice_to_haves) plus
    the resume text form the legitimate source vocabulary. ``extra_allowed`` lets callers
    whitelist additional tokens (rarely needed).
    """
    parsed_job = parsed_job or {}
    job_strings: list[str] = []
    for key in ("title", "company", "salary"):
        val = parsed_job.get(key)
        if isinstance(val, str):
            job_strings.append(val)
    for key in ("required_skills", "responsibilities", "nice_to_haves"):
        vals = parsed_job.get(key) or []
        job_strings.extend(v for v in vals if isinstance(v, str))

    source_vocab = _source_vocabulary([resume_text, *job_strings, *extra_allowed])

    entities = extract_candidate_entities(cover_letter)
    ungrounded = sorted(e for e in entities if not _is_grounded(e, source_vocab))

    return GroundingResult(
        grounded=not ungrounded,
        ungrounded_entities=ungrounded,
        checked_entities=sorted(entities),
    )
