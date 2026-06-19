"""Headless CLI runner — handy for testing the pipeline without Streamlit.

Usage:
    python run_cli.py --resume resume.txt --url https://job-posting
    python run_cli.py --resume resume.txt --urls urls.txt   # batch, one URL per line
"""

from __future__ import annotations

import argparse
import json
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

from backend.graph import run_batch, run_single


def _summary(state: dict) -> dict:
    return {
        "application_id": state.get("application_id"),
        "company": (state.get("parsed_job") or {}).get("company"),
        "role": (state.get("parsed_job") or {}).get("title"),
        "match_score": state.get("match_score"),
        "tailored_bullets": len(state.get("tailored_bullets") or []),
        "status": state.get("status"),
        "error": state.get("error"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Job Application Agent CLI")
    ap.add_argument("--resume", required=True, help="path to resume text file")
    ap.add_argument("--url", help="single job posting URL")
    ap.add_argument("--urls", help="file with one job URL per line (batch)")
    args = ap.parse_args()

    with open(args.resume, "r", encoding="utf-8") as fh:
        resume = fh.read()

    if args.url:
        result = run_single(args.url, resume)
        print(json.dumps(_summary(result), indent=2))
    elif args.urls:
        with open(args.urls, "r", encoding="utf-8") as fh:
            urls = [ln.strip() for ln in fh if ln.strip()]
        results = run_batch(
            urls,
            resume,
            on_complete=lambda d, t, s: print(f"[{d}/{t}] done", file=sys.stderr),
        )
        print(json.dumps([_summary(r) for r in results], indent=2))
    else:
        ap.error("provide either --url or --urls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
