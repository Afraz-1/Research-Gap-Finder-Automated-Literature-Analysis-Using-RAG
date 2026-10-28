import io
import json

from rag_gap_finder.fetch_papers import reconstruct_abstract
from rag_gap_finder.gaps import inspect_gaps
from rag_gap_finder.retrieve import focused_excerpt, ollama_answer_stream


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


def test_focused_excerpt_keeps_relevant_complete_sentence():
    text = (
        "The paper introduces a dashboard. "
        "Student privacy is a concern for the dashboard. "
        "Teachers used it in a classroom."
    )
    excerpt = focused_excerpt("What about privacy?", text, max_sentences=1)
    assert excerpt == "Student privacy is a concern for the dashboard."


def test_ollama_answer_stream_uses_numbered_evidence(monkeypatch):
    captured = {}

    def fake_open(request, timeout):
        captured["payload"] = json.loads(request.data)
        return io.BytesIO(
            b'{"response":"Privacy matters ","done":false}\n'
            b'{"response":"[1].","done":false}\n'
            b'{"response":"","done":true}\n'
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    evidence = [{
        "rank": 1, "title": "Privacy in Education", "year": "2024",
        "location": "abstract", "raw_text": "Student privacy is a concern.",
    }]
    answer = "".join(ollama_answer_stream("What about privacy?", evidence))
    assert answer == "Privacy matters [1]."
    assert captured["payload"]["stream"] is True
    assert "[1] Privacy in Education" in captured["payload"]["prompt"]
