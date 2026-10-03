"""Download a small, reproducible AI-in-education abstract corpus from OpenAlex."""

from __future__ import annotations

import argparse
import csv
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


QUERIES = [
    ("generative_ai", "generative artificial intelligence feedback higher education", ("generative ai", "generative artificial intelligence", "chatgpt")),
    ("learning_analytics", "learning analytics student performance higher education", ("learning analytics", "educational data mining")),
    ("intelligent_tutoring", "intelligent tutoring systems higher education", ("intelligent tutor", "ai tutor")),
    ("ethics_policy", "artificial intelligence privacy fairness higher education", ("ethic", "privacy", "fairness", "policy", "responsible ai")),
]
TITLE_MARKERS = (
    "education", "educational", "student", "teaching", "teacher", "tutor",
    "classroom", "university", "academic", "learning analytics", "school",
    "assessment", "pedagogy", "course", "college",
)
AI_MARKERS = (
    "artificial intelligence", "generative ai", "chatgpt", "intelligent tutor",
    "learning analytics", "machine learning", " ai ", " ai:", " ai-",
)


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions = [(position, word) for word, occurrences in index.items() for position in occurrences]
    return " ".join(word for _, word in sorted(positions))


def fetch(query: str, api_key: str | None) -> list[dict]:
    params = {
        "search": query,
        "filter": "from_publication_date:2020-01-01,to_publication_date:2026-10-03,has_abstract:true,type:article",
        "per_page": "100",
    }
    if api_key:
        params["api_key"] = api_key
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "ITA-mini-project/0.1 (student literature demo)"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["results"]


def build_corpus(max_per_query: int, api_key: str | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for topic_label, query, topic_markers in QUERIES:
        accepted = 0
        for work in fetch(query, api_key):
            title = (work.get("display_name") or "").strip()
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            if not title or len(abstract.split()) < 60:
                continue
            normalized_title = " " + title.lower() + " "
            if not any(marker in normalized_title for marker in TITLE_MARKERS):
                continue
            if not any(marker in normalized_title for marker in AI_MARKERS):
                continue
            if not any(marker in normalized_title for marker in topic_markers):
                continue
            identifier = work.get("doi") or work.get("id")
            if identifier in seen:
                continue
            seen.add(identifier)
            rows.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "authors": ", ".join(
                        author["author"]["display_name"]
                        for author in work.get("authorships", [])[:5]
                        if author.get("author", {}).get("display_name")
                    ),
                    "year": str(work.get("publication_year") or ""),
                    "doi": work.get("doi") or "",
                    "url": work.get("id") or "",
                    "topic_label": topic_label,
                    "search_query": query,
                }
            )
            accepted += 1
            if accepted >= max_per_query:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch AI-in-education abstracts from OpenAlex")
    parser.add_argument("--output", type=Path, default=Path("data/ai_education_papers.csv"))
    parser.add_argument("--per-query", type=int, default=6)
    args = parser.parse_args()
    if args.per_query < 1 or args.per_query > 10:
        parser.error("--per-query must be between 1 and 10")
    rows = build_corpus(args.per_query, os.environ.get("OPENALEX_API_KEY"))
    if not rows:
        raise RuntimeError("OpenAlex returned no usable abstracts.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} paper abstracts to {args.output}")
    for row in rows:
        print(f"{row['year']} | {row['title'][:100]}")


if __name__ == "__main__":
    main()
