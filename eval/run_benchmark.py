"""Run the agent benchmark suite and report pass rate, latency, and token cost.

Usage:
    python -m eval.run_benchmark                  # MOCK tier (default): free, deterministic
    python -m eval.run_benchmark --live           # LIVE tier: real DeepSeek/Gemini
    python -m eval.run_benchmark --threshold 0.9  # fail process if pass rate < 0.90
    python -m eval.run_benchmark --json out.json  # also write machine-readable results

In CI we run the MOCK tier with a threshold; the build fails (exit 1) if the pass rate
drops below it. The LIVE tier is for local / on-demand quality measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

# Ensure repo root on path when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.fixtures import all_tasks  # noqa: E402
from eval.harness import run_task  # noqa: E402
from eval.scoring import score_task  # noqa: E402


def _isolate_state_dirs() -> None:
    """Point DB + Chroma at throwaway temp dirs so the benchmark never touches real data."""
    tmp = tempfile.mkdtemp(prefix="benchmark_")
    os.environ["JOB_AGENT_DB"] = os.path.join(tmp, "bench.db")
    os.environ["JOB_AGENT_CHROMA_DIR"] = os.path.join(tmp, "chroma")
    # Reset singletons if already imported.
    try:
        import backend.db.database as dbmod
        dbmod.DB_PATH = os.environ["JOB_AGENT_DB"]
    except Exception:  # noqa: BLE001
        pass
    try:
        import backend.memory.store as ms
        ms._store = None
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Job-agent benchmark")
    ap.add_argument("--live", action="store_true", help="use real LLMs (costs money)")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="fail (exit 1) if pass rate < threshold (0..1)")
    ap.add_argument("--json", dest="json_out", default="", help="write results JSON here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.live:
        os.environ.setdefault("DEEPSEEK_API_KEY", "")
        os.environ.setdefault("GEMINI_API_KEY", "")
    _isolate_state_dirs()

    tasks = all_tasks()
    results = []
    total_latency = 0.0
    total_tokens = 0
    passed_count = 0

    for task in tasks:
        out = run_task(task, live=args.live)
        failures = score_task(task, out["state"])
        ok = not failures
        passed_count += int(ok)
        total_latency += out["latency_s"]
        total_tokens += out["tokens"]
        results.append({
            "id": task.id, "category": task.category, "passed": ok,
            "failures": failures, "latency_s": round(out["latency_s"], 4),
            "approx_tokens": out["tokens"],
        })
        if not args.quiet:
            mark = "PASS" if ok else "FAIL"
            print(f"[{mark}] {task.id:<28} ({task.category})")
            for f in failures:
                print(f"        - {f}")

    n = len(tasks)
    pass_rate = passed_count / n if n else 0.0
    avg_latency = total_latency / n if n else 0.0
    avg_tokens = total_tokens / n if n else 0

    summary = {
        "tier": "live" if args.live else "mock",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": n,
        "passed": passed_count,
        "pass_rate": round(pass_rate, 4),
        "avg_latency_s": round(avg_latency, 4),
        "avg_approx_tokens": round(avg_tokens),
        "results": results,
    }

    print("\n" + "=" * 56)
    print(f"  tier={summary['tier']}  tasks={n}  passed={passed_count}")
    print(f"  PASS RATE : {pass_rate * 100:.1f}%")
    print(f"  avg latency: {avg_latency * 1000:.1f} ms/task")
    print(f"  avg tokens : ~{avg_tokens:.0f}/task ({'real' if args.live else 'estimated'})")
    print("=" * 56)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"wrote {args.json_out}")

    if args.threshold and pass_rate < args.threshold:
        print(f"\nFAILED: pass rate {pass_rate:.2%} < threshold {args.threshold:.2%}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
