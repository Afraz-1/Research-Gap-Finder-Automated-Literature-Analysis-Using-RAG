"""Reproducible OpenAlex title/abstract search and CSV import."""

from __future__ import annotations

import csv
import io
import json
import os
import urllib.parse
import urllib.request


def read_csv(content: bytes) -> list[dict]:
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    if not reader.fieldnames or not {"title", "abstract"}.issubset(reader.fieldnames):
        raise ValueError("CSV must have title and abstract columns.")
    records = list(reader)
    if not records:
        raise ValueError("CSV has no paper rows.")
    return records


def _abstract(index: dict | None) -> str:
    if not index:
        return ""
    tokens = [(position, word) for word, positions in index.items() for position in positions]
    return " ".join(word for _, word in sorted(tokens))


def search_openalex(query: str, from_year: int, to_year: int,
                    limit: int = 100) -> tuple[list[dict], dict]:
    """Fetch a bounded, logged search. Search scope is title and abstract."""
    if not query.strip():
        raise ValueError("Search query is required.")
    if from_year > to_year or not 1900 <= from_year <= 2100 or not 1900 <= to_year <= 2100:
        raise ValueError("Invalid publication-year range.")
    if not 1 <= limit <= 500:
        raise ValueError("Search limit must be between 1 and 500.")
    filters = (
        f"from_publication_date:{from_year}-01-01,"
        f"to_publication_date:{to_year}-12-31,has_abstract:true,type:article"
    )
    params = {
        "search.title_and_abstract": query.strip(),
        "filter": filters,
        "per_page": str(min(100, limit)),
        "cursor": "*",
    }
    api_key = os.environ.get("OPENALEX_API_KEY")
    records: list[dict] = []
    pages = 0
    while len(records) < limit:
        request_params = dict(params)
        if api_key:
            request_params["api_key"] = api_key
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(request_params)
        request = urllib.request.Request(
            url, headers={"User-Agent": "ResearchEvidenceExplorer/1.0 (literature review workspace)"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
        pages += 1
        for work in payload.get("results", []):
            abstract = _abstract(work.get("abstract_inverted_index"))
            if not abstract or not work.get("display_name"):
                continue
            records.append({
                "title": work["display_name"],
                "abstract": abstract,
                "authors": ", ".join(
                    item.get("author", {}).get("display_name", "")
                    for item in work.get("authorships", [])[:10]
                    if item.get("author", {}).get("display_name")
                ),
                "year": str(work.get("publication_year") or ""),
                "doi": work.get("doi") or "",
                "url": (work.get("primary_location") or {}).get("landing_page_url")
                       or work.get("id", ""),
                "openalex_id": work.get("id") or "",
            })
            if len(records) >= limit:
                break
        next_cursor = payload.get("meta", {}).get("next_cursor")
        if not next_cursor or not payload.get("results"):
            break
        params["cursor"] = next_cursor
    return records, {
        "query_parameter": "search.title_and_abstract",
        "query": query.strip(), "filter": filters, "limit": limit,
        "pages_fetched": pages, "endpoint": "https://api.openalex.org/works",
    }
