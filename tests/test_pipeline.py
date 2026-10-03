from rag_gap_finder.fetch_papers import reconstruct_abstract
from rag_gap_finder.gaps import inspect_gaps


def test_openalex_abstract_reconstructed_in_word_order():
    assert reconstruct_abstract({"learning": [2], "AI": [0], "supports": [1]}) == "AI supports learning"


def test_gap_report_counts_distinct_papers_and_uses_cautious_language():
    passages = [
        {
            "paper_id": str(i), "title": f"Paper {i}", "url": f"https://example.org/{i}",
            "year": "2024", "topic_label": "learning_analytics",
            "raw_text": text,
        }
        for i, text in enumerate([
            "Student performance improved. Privacy was discussed.",
            "A dashboard showed learning gains.",
            "Learning analytics improved outcomes.",
            "The study examined academic performance.",
        ])
    ]
    report = inspect_gaps(passages)
    assert report["coverage"]["learning_analytics"]["privacy and fairness"] == 1
    candidate = next(item for item in report["candidates"] if item["aspect"] == "privacy and fairness")
    assert candidate["count"] == 1 and candidate["total"] == 4
    assert "Check whether" in candidate["hypothesis"]
