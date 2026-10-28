import json

import pytest

from rag_gap_finder.discovery import read_csv, search_openalex
from rag_gap_finder.evidence import discover_candidates, limitation_sentences, related_papers
from rag_gap_finder.workspace import (
    create_project, create_search_run, database, finish_search_run, import_papers,
    list_candidates, list_papers, list_search_runs, review_candidate, set_screening,
)


def test_search_screen_candidate_review_and_reopen(tmp_path):
    path = tmp_path / "review.sqlite3"
    rows = [
        {"title": "Longitudinal Study", "abstract":
         "Teachers used a feedback tool. Future research should examine outcomes after one year.",
         "doi": "https://doi.org/10.1234/EXAMPLE", "year": "2024"},
        {"title": "Related Study", "abstract":
         "A study of feedback tools found improved learning outcomes after one year.",
         "doi": "10.1234/other", "year": "2025"},
    ]
    with database(path) as con:
        project_id = create_project(con, "Feedback review", "What happens after one year?", "", "")
        run_id = create_search_run(con, project_id, "CSV import", "papers.csv", {"scope": "abstracts"})
        assert import_papers(con, project_id, run_id, rows) == 2
        assert import_papers(con, project_id, run_id, rows) == 0
        finish_search_run(con, run_id, "completed", 2, 2)
        papers = list_papers(con, project_id)
        assert len(papers) == 2
        for paper in papers:
            set_screening(con, paper["id"], "include")
        assert discover_candidates(con, project_id) == 1
        assert discover_candidates(con, project_id) == 0
        candidate = list_candidates(con, project_id)[0]
        assert candidate["source_sentence"] == "Future research should examine outcomes after one year."
        assert candidate["abstract"][candidate["start_offset"]:].startswith(candidate["source_sentence"])
        assert related_papers(candidate, list_papers(con, project_id))[0]["title"] == "Related Study"
        with pytest.raises(ValueError, match="counterevidence"):
            review_candidate(con, candidate["id"], "accepted", "pending", "Seems useful")
        review_candidate(con, candidate["id"], "accepted", "checked", "Related study may answer this.")
    with database(path) as con:
        assert list_candidates(con, project_id)[0]["status"] == "accepted"
        assert json.loads(list_search_runs(con, project_id)[0]["filters_json"])["scope"] == "abstracts"


def test_exclusion_requires_reason_and_hides_candidate(tmp_path):
    with database(tmp_path / "review.sqlite3") as con:
        project = create_project(con, "Test", "Question", "", "")
        run = create_search_run(con, project, "CSV", "file", {})
        import_papers(con, project, run, [{"title": "A", "abstract":
            "Future research should investigate this issue in larger studies."}])
        paper = list_papers(con, project)[0]
        with pytest.raises(ValueError, match="reason"):
            set_screening(con, paper["id"], "exclude")
        set_screening(con, paper["id"], "include")
        assert discover_candidates(con, project) == 1
        set_screening(con, paper["id"], "exclude", "Out of scope")
        assert list_candidates(con, project) == []


def test_csv_validation_and_sentence_detection():
    assert read_csv(b"title,abstract\nA,Some findings\n")[0]["title"] == "A"
    with pytest.raises(ValueError, match="title and abstract"):
        read_csv(b"title,body\nA,Some findings\n")
    assert limitation_sentences("The method works. Future work should test a larger sample.") == [
        ("Future work should test a larger sample.", 18)
    ]


def test_openalex_query_paginates_and_reconstructs_abstract(monkeypatch):
    calls = []
    class Response:
        def __init__(self, payload):
            self.payload = payload
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return json.dumps(self.payload).encode()

    def fake_open(request, timeout):
        calls.append(request.full_url)
        page = len(calls)
        return Response({
            "results": [{"id": f"https://openalex.org/W{page}", "display_name": f"Paper {page}",
                         "abstract_inverted_index": {"study": [1], "A": [0]},
                         "publication_year": 2024}],
            "meta": {"next_cursor": "next" if page == 1 else None},
        })
    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    rows, details = search_openalex("feedback learning", 2020, 2026, limit=2)
    assert [r["abstract"] for r in rows] == ["A study", "A study"]
    assert len(calls) == 2
    assert "search.title_and_abstract=" in calls[0]
    assert details["pages_fetched"] == 2
