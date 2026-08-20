#!/usr/bin/env python3
"""Fetch papers citing the DDXPlus paper via the Semantic Scholar Graph API,
rank them, apply manual curation, and write docs/citations.json.

Stdlib-only so it can run in CI without installing dependencies.

Usage:
    python3 fetch_citations.py [--api-key KEY]

Set the S2_API_KEY environment variable (or --api-key) to raise the
unauthenticated rate limit if you hit 429s.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path

PAPER_ID = "arXiv:2205.09148"  # the DDXPlus paper
API = "https://api.semanticscholar.org/graph/v1/paper"
CITATION_FIELDS = "title,year,venue,authors,citationCount,influentialCitationCount,isInfluential,externalIds"

# dropped automatically unless the paper is explicitly pinned in citations.curation.json
EXCLUDE_TITLE_KEYWORDS = ["survey", "review"]

# substrings matched case-insensitively against the venue field to award the score bonus in score()
TOP_TIER_VENUE_KEYWORDS = [
    "neural information processing systems", "neurips", "nips",
    "international conference on machine learning", "icml",
    "international conference on learning representations", "iclr",
    "association for computational linguistics", "acl",
    "empirical methods in natural language processing", "emnlp",
    "computer vision and pattern recognition", "cvpr",
    "conference on language modeling", "colm",
    "nature medicine", "nature", "science", "the lancet",
    "new england journal of medicine", "jama",
]

DOCS_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = DOCS_DIR / "citations.json"
CURATION_PATH = DOCS_DIR / "citations.curation.json"


def fetch_json(url: str, api_key: str | None, retries: int = 6, backoff: float = 5.0) -> dict:
    headers = {"x-api-key": api_key} if api_key else {}
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            # unauthenticated requests share a small rate-limit pool -- back off and retry
            if e.code == 429 and attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
                continue
            raise


def fetch_paper_summary(api_key: str | None) -> dict:
    url = f"{API}/{PAPER_ID}?fields=title,citationCount"
    return fetch_json(url, api_key)


def fetch_all_citations(api_key: str | None) -> list[dict]:
    citations, offset = [], 0
    while True:
        params = urllib.parse.urlencode({"fields": CITATION_FIELDS, "limit": 1000, "offset": offset})
        url = f"{API}/{PAPER_ID}/citations?{params}"
        page = fetch_json(url, api_key)
        citations += [c["citingPaper"] for c in page.get("data", []) if c.get("citingPaper")]
        if "next" not in page:
            break
        offset = page["next"]
        time.sleep(1)  # be polite to the shared rate pool between pages
    return citations


def paper_key(p: dict) -> str:
    ext = p.get("externalIds") or {}
    return ext.get("ArXiv") or ext.get("DOI") or p.get("paperId") or p.get("title", "")


def dedupe(papers: list[dict]) -> list[dict]:
    seen, out = set(), []
    for p in papers:
        key = paper_key(p) or p.get("title", "").strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def is_top_tier_venue(venue: str) -> bool:
    lowered = (venue or "").lower()
    return any(kw in lowered for kw in TOP_TIER_VENUE_KEYWORDS)


def score(paper: dict) -> float:
    cites = paper.get("citationCount") or 0
    year = paper.get("year") or date.today().year
    age_years = max(date.today().year - year + 1, 1)
    cites_per_year = cites / age_years
    venue_bonus = 1.5 if is_top_tier_venue(paper.get("venue")) else 0
    return math.log1p(cites) + 2 * bool(paper.get("isInfluential")) + 0.5 * cites_per_year + venue_bonus


def paper_url(p: dict) -> str:
    ext = p.get("externalIds") or {}
    if ext.get("DOI"):
        return f"https://doi.org/{ext['DOI']}"
    if ext.get("ArXiv"):
        return f"https://arxiv.org/abs/{ext['ArXiv']}"
    return f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}"


def load_curation() -> dict:
    if CURATION_PATH.exists():
        return json.loads(CURATION_PATH.read_text())
    return {"pin": [], "hide": []}


def is_excluded_by_title(title: str) -> bool:
    lowered = title.lower()
    return any(kw in lowered for kw in EXCLUDE_TITLE_KEYWORDS)


def to_entry(p: dict) -> dict:
    return {
        "title": p["title"],
        "authors": [a["name"] for a in (p.get("authors") or [])],
        "venue": p.get("venue") or "",
        "year": p.get("year"),
        "url": paper_url(p),
        "citationCount": p.get("citationCount") or 0,
        "influential": bool(p.get("isInfluential")),
        "key": paper_key(p),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", default=os.environ.get("S2_API_KEY"))
    parser.add_argument("--limit", type=int, default=40, help="max papers to keep in citations.json")
    args = parser.parse_args()

    summary = fetch_paper_summary(args.api_key)
    raw = dedupe([p for p in fetch_all_citations(args.api_key) if p.get("title")])

    curation = load_curation()
    hidden = set(curation.get("hide", []))
    pinned = curation.get("pin", [])

    entries = [
        to_entry(p) for p in raw
        if paper_key(p) not in hidden
        and (paper_key(p) in pinned or not is_excluded_by_title(p["title"]))
    ]
    entries.sort(key=lambda e: score({**e, "citationCount": e["citationCount"], "isInfluential": e["influential"]}), reverse=True)

    pinned_entries = sorted(
        (e for e in entries if e["key"] in pinned),
        key=lambda e: pinned.index(e["key"]),
    )
    rest = [e for e in entries if e["key"] not in pinned]

    seen_keys, ordered = set(), []
    for e in pinned_entries + rest:
        if e["key"] in seen_keys:
            continue
        seen_keys.add(e["key"])
        ordered.append(e)
    ordered = ordered[: args.limit]

    out = {
        "updated": datetime.now(timezone.utc).date().isoformat(),
        "source": "Semantic Scholar",
        "totalCitations": summary.get("citationCount", len(entries)),
        "papers": ordered,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {len(out['papers'])} papers ({out['totalCitations']} total citations) to {OUT_PATH}")


if __name__ == "__main__":
    main()
