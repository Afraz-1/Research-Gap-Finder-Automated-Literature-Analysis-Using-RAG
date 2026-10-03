"""Module 4: aspect coverage, cautious gap hypotheses, and limitation evidence."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import nltk

from rag_gap_finder.analyze import read_passages


ASPECTS = {
    "learning outcomes": ("learning outcome", "academic performance", "achievement", "learning gain"),
    "privacy and fairness": ("privacy", "data protection", "bias", "fairness", "ethical", "ethics"),
    "teacher role": ("teacher", "instructor", "educator", "pedagog"),
    "long-term evaluation": ("long-term", "longitudinal", "follow-up", "over time", "sustained"),
    "accessibility and equity": ("accessib", "disabil", "equity", "inclusive", "digital divide"),
}
POSITIVE = ("improv", "benefit", "effectiv", "promising", "enhanc", "positive", "support")
NEGATIVE = ("challeng", "concern", "risk", "limitation", "bias", "lack", "threat", "problem")
LIMITATION = re.compile(r"\b(?:limitation|limited|lack|few|little|future research|future work|further research|gap|underexplored|not yet)\b", re.I)


def _stance(sentence: str) -> str:
    lower = sentence.lower()
    positive = any(word in lower for word in POSITIVE)
    negative = any(word in lower for word in NEGATIVE)
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "neutral"


def inspect_gaps(passages: list[dict]) -> dict:
    by_paper: dict[str, dict] = {}
    for passage in passages:
        paper = by_paper.setdefault(
            passage["paper_id"],
            {"title": passage["title"], "url": passage["url"], "year": passage["year"],
             "topic_label": passage["topic_label"], "text": ""},
        )
        paper["text"] += " " + passage["raw_text"]
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for paper_id, paper in by_paper.items():
        groups[paper["topic_label"] or "unlabeled"].append((paper_id, paper))

    coverage = {}
    aspect_sentences = []
    limitations = []
    for group, papers in groups.items():
        coverage[group] = {aspect: 0 for aspect in ASPECTS}
        for paper_id, paper in papers:
            sentences = nltk.sent_tokenize(paper["text"])
            for aspect, markers in ASPECTS.items():
                matching = [sentence.strip() for sentence in sentences if any(marker in sentence.lower() for marker in markers)]
                if matching:
                    coverage[group][aspect] += 1
                    aspect_sentences.append({
                        "paper_id": paper_id, "title": paper["title"], "url": paper["url"],
                        "group": group, "aspect": aspect, "stance": _stance(matching[0]),
                        "sentence": matching[0],
                    })
            for sentence in sentences:
                if LIMITATION.search(sentence):
                    limitations.append({
                        "paper_id": paper_id, "title": paper["title"], "url": paper["url"],
                        "group": group, "sentence": sentence.strip(),
                    })

    candidates = []
    for group, papers in groups.items():
        n = len(papers)
        if n < 4:
            continue
        for aspect, count in coverage[group].items():
            if count > n // 3:
                continue
            citations = [
                {"title": paper["title"], "url": paper["url"], "year": paper["year"]}
                for _, paper in papers[:3]
            ]
            matching_limitation = next(
                (item for item in limitations if item["group"] == group and any(
                    marker in item["sentence"].lower() for marker in ASPECTS[aspect])),
                None,
            )
            candidates.append({
                "topic": group,
                "aspect": aspect,
                "count": count,
                "total": n,
                "hypothesis": (
                    f"Only {count} of {n} {group.replace('_', ' ')} abstracts mention "
                    f"{aspect}. Check whether this is an underexplored question in a larger search."
                ),
                "sample_papers": citations,
                "related_limitation": matching_limitation,
            })
    candidates.sort(key=lambda item: (item["count"] / item["total"], item["topic"], item["aspect"]))
    return {
        "paper_count": len(by_paper),
        "group_sizes": {group: len(papers) for group, papers in groups.items()},
        "coverage": coverage,
        "candidates": candidates,
        "aspect_sentences": aspect_sentences,
        "explicit_limitations": limitations,
        "warning": "Counts refer only to selected abstracts and keyword matches. Validate each hypothesis with more papers and full texts.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Find candidate gaps in a small corpus")
    parser.add_argument("--input", type=Path, default=Path("data/processed/ai_education_passages.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/gaps.json"))
    args = parser.parse_args()
    report = inspect_gaps(read_passages(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Checked {report['paper_count']} papers; found {len(report['candidates'])} hypotheses")
    for candidate in report["candidates"][:8]:
        print(candidate["hypothesis"])
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
