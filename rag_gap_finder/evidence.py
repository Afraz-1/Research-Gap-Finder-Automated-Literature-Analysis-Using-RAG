"""Conservative candidate discovery from source-stated limitations."""

from __future__ import annotations

import re
import sqlite3

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_gap_finder.workspace import add_candidate, list_papers


LIMITATION_CUE = re.compile(
    r"\b(?:limitations?|future (?:work|research|studies)|further (?:work|research|studies)"
    r"|remains? (?:unclear|unknown|unexplored)|not yet|underexplored|lack(?:s|ing)?"
    r"|small sample|more research|additional research)\b",
    re.IGNORECASE,
)
SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.DOTALL)


def limitation_sentences(abstract: str) -> list[tuple[str, int]]:
    found = []
    for match in SENTENCE.finditer(abstract):
        sentence = match.group().strip()
        if 35 <= len(sentence) <= 700 and LIMITATION_CUE.search(sentence):
            found.append((sentence, match.start() + len(match.group()) - len(match.group().lstrip())))
    return found


def discover_candidates(con: sqlite3.Connection, project_id: int) -> int:
    before = con.execute("SELECT COUNT(*) FROM candidates WHERE project_id=?", (project_id,)).fetchone()[0]
    for paper in list_papers(con, project_id):
        if paper["decision"] != "include":
            continue
        for sentence, offset in limitation_sentences(paper["abstract"]):
            add_candidate(con, project_id, paper["id"], sentence, offset)
    after = con.execute("SELECT COUNT(*) FROM candidates WHERE project_id=?", (project_id,)).fetchone()[0]
    return after - before


def related_papers(candidate: dict, included: list[dict], limit: int = 4) -> list[dict]:
    """Other relevant abstracts to inspect; similarity does not imply contradiction."""
    other = [p for p in included if p["id"] != candidate["source_paper_id"]]
    if not other:
        return []
    texts = [candidate["source_sentence"]] + [p["title"] + " " + p["abstract"] for p in other]
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(texts)
    except ValueError:
        return []
    scores = cosine_similarity(matrix[0], matrix[1:]).ravel()
    ranked = sorted(range(len(other)), key=lambda i: (-scores[i], other[i]["id"]))[:limit]
    return [{**other[i], "similarity": round(float(scores[i]), 3)} for i in ranked if scores[i] > 0]
