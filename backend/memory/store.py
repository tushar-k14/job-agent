"""Persistent vector memory backed by Chroma (in-process, no external infra).

Scope (deliberately "real but scoped" — see project decisions): the agent remembers
**what worked per site** and **why runs failed**, so the planner can act on prior
experience instead of cold-starting every time.

Two complementary access patterns:

1. **Deterministic site strategy** — keyed lookups by domain ("for greenhouse.io,
   selector_structured succeeded 4/5 times"). This is exact, not fuzzy, so it drives the
   planner's first-attempt strategy reliably. Stored as Chroma metadata and aggregated.

2. **Semantic recall** — embeddings over free-text trajectory summaries ("thin JD,
   enrichment helped", "captcha wall on attempt 1"). Used for surfacing similar past
   situations in the dashboard / future planner heuristics.

Chroma persists to ``data/chroma/`` (a PersistentClient), so memory survives restarts
and Docker volume mounts. If Chroma is unavailable for any reason, every function
degrades to a no-op / empty result so the pipeline never hard-depends on memory.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import Counter
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("JOB_AGENT_CHROMA_DIR", os.path.join("data", "chroma"))
_OUTCOMES_COLLECTION = "run_outcomes"

_lock = threading.Lock()
_store: Optional["MemoryStore"] = None


class _NoopEmbedding:
    """A deterministic zero-vector embedding function.

    Chroma requires *an* embedding function, but our access pattern is metadata
    filtering (``collection.get(where=...)``), never similarity search. Returning fixed
    8-dim zero vectors keeps Chroma happy with no model download. Implemented as a plain
    callable matching Chroma's ``EmbeddingFunction.__call__(input) -> Embeddings``.
    """

    _DIM = 8

    def __call__(self, input):  # noqa: A002 - Chroma's parameter name is `input`
        return [[0.0] * self._DIM for _ in input]

    # Chroma 0.5.x introspects these for collection config persistence.
    def name(self) -> str:  # pragma: no cover - trivial
        return "noop"

    @staticmethod
    def build_from_config(config):  # pragma: no cover - trivial
        return _NoopEmbedding()

    def get_config(self):  # pragma: no cover - trivial
        return {}


def domain_of(url: str) -> str:
    """Normalized registrable-ish host, e.g. 'boards.greenhouse.io'. '' for paste/empty."""
    if not url or url in ("pasted", "paste"):
        return ""
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


class MemoryStore:
    """Thin wrapper around a Chroma collection of run outcomes."""

    def __init__(self, persist_dir: str = CHROMA_DIR):
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._init_error: Optional[str] = None
        self._connect()

    def _connect(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings

            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            # Our primary access pattern (per-domain strategy recall + known-bad checks)
            # is METADATA filtering, which needs no embeddings. We therefore default to a
            # no-op embedding function so Chroma does not download the ~79MB ONNX model at
            # import time (important for fast, offline CI and Docker). Set
            # JOB_AGENT_MEMORY_EMBED=1 to enable real embeddings for semantic recall.
            if os.getenv("JOB_AGENT_MEMORY_EMBED") == "1":
                # Real semantic embeddings (Chroma default ONNX MiniLM, one-time download).
                self._collection = self._client.get_or_create_collection(
                    name=_OUTCOMES_COLLECTION,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                self._collection = self._client.get_or_create_collection(
                    name=_OUTCOMES_COLLECTION,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=_NoopEmbedding(),
                )
            logger.info("Memory: Chroma ready at %s", self.persist_dir)
        except Exception as exc:  # noqa: BLE001
            self._init_error = str(exc)
            logger.warning("Memory disabled (Chroma init failed): %s", exc)

    @property
    def available(self) -> bool:
        return self._collection is not None

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def record_outcome(
        self,
        *,
        url: str,
        strategy: str,
        success: bool,
        reason: str,
        match_score: Optional[int] = None,
        cover_letter_passed: Optional[bool] = None,
    ) -> None:
        if not self.available:
            return
        domain = domain_of(url)
        now = datetime.now(timezone.utc).isoformat()
        summary = (
            f"domain={domain or 'pasted'} strategy={strategy} "
            f"success={success} reason={reason}"
        )
        doc_id = f"{domain or 'pasted'}-{now}-{abs(hash(reason)) % 100000}"
        try:
            self._collection.add(
                ids=[doc_id],
                documents=[summary],
                metadatas=[
                    {
                        "domain": domain or "pasted",
                        "strategy": strategy or "unknown",
                        "success": bool(success),
                        "reason": reason or "",
                        "match_score": int(match_score) if match_score is not None else -1,
                        "cover_letter_passed": (
                            bool(cover_letter_passed)
                            if cover_letter_passed is not None
                            else False
                        ),
                        "ts": now,
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory: failed to record outcome: %s", exc)

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def best_strategy_for_domain(self, url: str, min_samples: int = 1) -> Optional[str]:
        """Return the strategy with the best historical success rate for this domain.

        Returns None when there is no domain (paste path) or insufficient history.
        """
        if not self.available:
            return None
        domain = domain_of(url)
        if not domain:
            return None
        try:
            res = self._collection.get(where={"domain": domain})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory: domain query failed: %s", exc)
            return None

        metadatas = res.get("metadatas") or []
        if len(metadatas) < min_samples:
            return None

        # success counts and totals per strategy
        succ: Counter = Counter()
        total: Counter = Counter()
        for m in metadatas:
            strat = m.get("strategy", "unknown")
            total[strat] += 1
            if m.get("success"):
                succ[strat] += 1

        # Pick strategy with highest success rate, breaking ties by sample count.
        best, best_rate, best_n = None, -1.0, 0
        for strat, n in total.items():
            rate = succ[strat] / n if n else 0.0
            if (rate, n) > (best_rate, best_n):
                best, best_rate, best_n = strat, rate, n
        # Only recommend if it has actually succeeded at least once.
        if best and succ[best] > 0:
            logger.info(
                "Memory: recall %s → %s (%.0f%% over %d runs)",
                domain, best, best_rate * 100, best_n,
            )
            return best
        return None

    def domain_known_bad(self, url: str, threshold: int = 3) -> bool:
        """True if a domain has failed >= threshold times with zero successes."""
        if not self.available:
            return False
        domain = domain_of(url)
        if not domain:
            return False
        try:
            res = self._collection.get(where={"domain": domain})
        except Exception:  # noqa: BLE001
            return False
        metadatas = res.get("metadatas") or []
        if len(metadatas) < threshold:
            return False
        return not any(m.get("success") for m in metadatas)

    def stats(self) -> dict:
        if not self.available:
            return {"available": False, "count": 0}
        try:
            count = self._collection.count()
        except Exception:  # noqa: BLE001
            count = 0
        return {"available": True, "count": count, "dir": self.persist_dir}


def get_memory() -> MemoryStore:
    global _store
    with _lock:
        if _store is None:
            _store = MemoryStore()
        return _store


# Convenience module-level helpers used by the nodes -------------------------- #
def recall_site_strategy(url: str) -> Optional[str]:
    return get_memory().best_strategy_for_domain(url)


def record_outcome(**kwargs) -> None:
    get_memory().record_outcome(**kwargs)
