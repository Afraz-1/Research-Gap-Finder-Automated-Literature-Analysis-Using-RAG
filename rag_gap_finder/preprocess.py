"""Module 1: ingest research papers and prepare traceable text passages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


# A small fixed list keeps the demo reproducible and requires no extra download.
STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have "
    "in into is it its may more most of on or our out over that the their these this "
    "those to was were which while will with would we you your they them than then "
    "such using use based study paper research results method methods".split()
) - {"using", "use", "based", "study", "paper", "research", "results", "method", "methods"}

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
CITATION_RE = re.compile(r"\[(?:\d+\s*[,;–-]?\s*)+\]")
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


@dataclass(frozen=True)
class Passage:
    paper_id: str
    title: str
    authors: str
    year: str
    doi: str
    url: str
    topic_label: str
    source: str
    location: str
    raw_text: str
    clean_text: str
    tokens: list[str]


def _wordnet_pos(tag: str) -> str:
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}.get(
        tag[:1], wordnet.NOUN
    )


def preprocess_text(text: str) -> tuple[str, list[str]]:
    """Clean, tokenize, remove stopwords, and lemmatize English text."""
    text = URL_RE.sub(" ", text)
    text = CITATION_RE.sub(" ", text)
    text = re.sub(r"-\s*\n\s*", "", text)  # repair PDF line-wrap hyphenation
    text = text.lower()
    words = WORD_RE.findall(text)
    words = [word for word in words if word not in STOPWORDS and len(word) > 1]
    if not words:
        return "", []
    try:
        tagged = pos_tag(words)
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word, _wordnet_pos(tag)) for word, tag in tagged]
    except LookupError as exc:
        raise RuntimeError(
            "NLTK language data is missing. Run: "
            "python -m nltk.downloader wordnet averaged_perceptron_tagger_eng"
        ) from exc
    return " ".join(tokens), tokens


def _paper_id(title: str, doi: str) -> str:
    key = (doi or title).strip().lower()
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _passage(**kwargs: str) -> Passage:
    clean_text, tokens = preprocess_text(kwargs["raw_text"])
    return Passage(**kwargs, clean_text=clean_text, tokens=tokens)


def load_csv(path: Path) -> list[Passage]:
    """Read one paper abstract per CSV row; preserve metadata and source text."""
    passages: list[Passage] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"title", "abstract"}.issubset(reader.fieldnames):
            raise ValueError("CSV needs 'title' and 'abstract' columns.")
        for row_number, row in enumerate(reader, start=2):
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            if not title or not abstract:
                raise ValueError(f"CSV row {row_number} needs a title and abstract.")
            doi = (row.get("doi") or "").strip()
            paper_id = _paper_id(title, doi)
            if paper_id in seen:
                raise ValueError(f"Duplicate paper at CSV row {row_number}: {title}")
            seen.add(paper_id)
            passages.append(
                _passage(
                    paper_id=paper_id,
                    title=title,
                    authors=(row.get("authors") or "").strip(),
                    year=(row.get("year") or "").strip(),
                    doi=doi,
                    url=(row.get("url") or "").strip(),
                    topic_label=(row.get("topic_label") or "").strip(),
                    source=str(path),
                    location="abstract",
                    raw_text=abstract,
                )
            )
    if not passages:
        raise ValueError("CSV has no papers.")
    return passages


def load_pdfs(directory: Path) -> list[Passage]:
    """Extract text by PDF page so later retrieval can cite a precise location."""
    import pymupdf

    files = sorted(directory.glob("*.pdf"))
    if not files:
        raise ValueError(f"No PDFs found in {directory}")
    passages: list[Passage] = []
    for path in files:
        paper_id = _paper_id(path.stem, "")
        with pymupdf.open(path) as pdf:
            page_count = 0
            for page_number, page in enumerate(pdf, start=1):
                raw_text = page.get_text("text", sort=True).strip()
                if not raw_text:
                    continue
                page_count += 1
                passages.append(
                    _passage(
                        paper_id=paper_id,
                        title=path.stem,
                        authors="",
                        year="",
                        doi="",
                        url="",
                        topic_label="",
                        source=str(path),
                        location=f"page {page_number}",
                        raw_text=raw_text,
                    )
                )
            if not page_count:
                raise ValueError(f"No extractable text in {path}; OCR may be needed.")
    return passages


def load_corpus(path: Path) -> list[Passage]:
    if path.is_file() and path.suffix.lower() == ".csv":
        return load_csv(path)
    if path.is_dir():
        return load_pdfs(path)
    raise ValueError("Input must be a CSV file or a directory containing PDFs.")


def save_jsonl(passages: list[Passage], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for passage in passages:
            handle.write(json.dumps(asdict(passage), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and preprocess research papers")
    parser.add_argument("--input", type=Path, required=True, help="CSV file or PDF directory")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    args = parser.parse_args()
    passages = load_corpus(args.input)
    save_jsonl(passages, args.output)
    papers = len({passage.paper_id for passage in passages})
    print(f"Processed {papers} papers into {len(passages)} passages: {args.output}")
    first = passages[0]
    # Windows terminals may use a legacy encoding; keep previews printable there.
    safe_title = first.title.encode("ascii", "backslashreplace").decode("ascii")
    safe_raw = first.raw_text[:180].encode("ascii", "backslashreplace").decode("ascii")
    print(f"Example: {safe_title} ({first.location})")
    print(f"Raw: {safe_raw}")
    print(f"Tokens: {first.tokens[:20]}")


if __name__ == "__main__":
    main()
