import csv

import pytest

from rag_gap_finder.preprocess import load_csv, preprocess_text


def test_preprocess_keeps_meaningful_lemmas_and_removes_noise():
    clean, tokens = preprocess_text("The studies were using models [12] at https://example.org.")
    assert "study" in tokens
    assert "use" in tokens
    assert "model" in tokens
    assert "the" not in tokens
    assert "example" not in tokens
    assert clean == " ".join(tokens)


def test_csv_preserves_source_and_rejects_duplicate_papers(tmp_path):
    path = tmp_path / "papers.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["title", "abstract"])
        writer.writeheader()
        writer.writerow({"title": "Paper A", "abstract": "The studies used models."})
    paper = load_csv(path)[0]
    assert paper.raw_text == "The studies used models."
    assert paper.location == "abstract"
    assert paper.source == str(path)

    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["Paper A", "A second abstract."])
    with pytest.raises(ValueError, match="Duplicate paper"):
        load_csv(path)
