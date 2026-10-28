"""Module 3: TF-IDF retrieval, NER, TextRank, and optional local LLM synthesis."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

import networkx as nx
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_gap_finder.analyze import read_passages
from rag_gap_finder.preprocess import preprocess_text


def textrank_summary(text: str, sentence_count: int = 2) -> str:
    """Rank sentences by similarity to other sentences, retaining source order."""
    sentences = nltk.sent_tokenize(text)[:40]
    if len(sentences) <= sentence_count:
        return " ".join(sentences)
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:sentence_count])
    similarities = cosine_similarity(matrix)
    graph = nx.from_numpy_array(similarities)
    scores = nx.pagerank(graph, weight="weight")
    chosen = sorted(sorted(scores, key=scores.get, reverse=True)[:sentence_count])
    return " ".join(sentences[index] for index in chosen)


def named_entities(text: str, max_words: int = 250) -> list[dict[str, str]]:
    """Show NLTK's person, organization, and location candidates; they need review."""
    try:
        words = nltk.word_tokenize(text)[:max_words]
        tree = nltk.ne_chunk(nltk.pos_tag(words))
    except LookupError as exc:
        raise RuntimeError(
            "NLTK NER data is missing. Run: python -m nltk.downloader "
            "punkt_tab maxent_ne_chunker_tab words"
        ) from exc
    found = []
    seen = set()
    for node in tree:
        if isinstance(node, nltk.Tree):
            entity = " ".join(word for word, _ in node.leaves())
            key = (entity.lower(), node.label())
            if key not in seen:
                seen.add(key)
                found.append({"text": entity, "type": node.label()})
    return found[:10]


def retrieve(passages: list[dict], question: str, top_k: int = 5) -> list[dict]:
    query, _ = preprocess_text(question)
    if not query:
        raise ValueError("Question contains no useful search words.")
    corpus = [p["clean_text"] for p in passages]
    matrix = TfidfVectorizer().fit_transform(corpus + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked = scores.argsort()[::-1][:top_k]
    return [
        {
            "rank": rank,
            "score": round(float(scores[index]), 3),
            "paper_id": passages[index]["paper_id"],
            "title": passages[index]["title"],
            "year": passages[index]["year"],
            "url": passages[index]["url"],
            "location": passages[index]["location"],
            "source": passages[index]["source"],
            "summary": textrank_summary(passages[index]["raw_text"]),
            "entities": named_entities(passages[index]["raw_text"]),
            "raw_text": passages[index]["raw_text"],
        }
        for rank, index in enumerate(ranked, start=1)
        if scores[index] > 0
    ]


def extractive_answer(question: str, evidence: list[dict]) -> str:
    """A deterministic answer that only combines retrieved source sentences."""
    if not evidence:
        return "No matching passage was found in this corpus."
    lines = [f"Question: {question}", "Relevant evidence in this corpus:"]
    for item in evidence[:5]:
        lines.append(f"[{item['rank']}] {item['summary']}")
    lines.append("These passages suggest directions to investigate; they do not establish that a research gap exists.")
    return "\n".join(lines)


def focused_excerpt(question: str, text: str, max_sentences: int = 3, max_chars: int = 850) -> str:
    """Keep complete source sentences most relevant to the question."""
    sentences = [sentence.strip() for sentence in nltk.sent_tokenize(text) if sentence.strip()]
    if not sentences:
        return ""
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(sentences + [question])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        ranked = sorted(range(len(sentences)), key=lambda index: (-scores[index], index))
    except ValueError:
        ranked = list(range(len(sentences)))
    selected = []
    size = 0
    for index in ranked:
        sentence = sentences[index]
        if selected and size + len(sentence) > max_chars:
            continue
        selected.append(index)
        size += len(sentence) + 1
        if len(selected) >= max_sentences:
            break
    return " ".join(sentences[index] for index in sorted(selected))


def ollama_answer_stream(question: str, evidence: list[dict], model: str = "llama3.2"):
    """Yield a grounded local-model answer as Ollama produces it."""
    if not evidence:
        yield "No matching passage was found in this corpus."
        return
    sources = "\n\n".join(
        f"[{item['rank']}] {item['title']} ({item['year']}), {item['location']}: "
        f"{focused_excerpt(question, item['raw_text'])}"
        for item in evidence[:5]
    )
    prompt = (
        "You are a research assistant. Answer the question directly in 2-4 sentences "
        "using ONLY the numbered source excerpts below. End each factual sentence "
        "with citations such as [1]. If the evidence is insufficient, say so. "
        "Treat possible research gaps as hypotheses. Write at most 100 words.\n\n"
        f"Question: {question}\n\nSources:\n{sources}\n\nAnswer:"
    )
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": True,
        "options": {"temperature": 0, "num_predict": 220},
    }).encode()
    request = urllib.request.Request(
        "http://localhost:11434/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            for line in response:
                if line.strip():
                    chunk = json.loads(line).get("response", "")
                    if chunk:
                        yield chunk
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("Local Ollama is unavailable. Start Ollama or use --generator extractive.") from exc


def ollama_answer(question: str, evidence: list[dict], model: str = "llama3.2") -> str:
    """Collect the streamed answer for command-line callers."""
    return "".join(ollama_answer_stream(question, evidence, model)).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a grounded question about the paper corpus")
    parser.add_argument("question")
    parser.add_argument("--input", type=Path, default=Path("data/processed/ai_education_passages.jsonl"))
    parser.add_argument("--generator", choices=["extractive", "ollama"], default="extractive")
    parser.add_argument("--output", type=Path, default=Path("data/processed/last_answer.json"))
    args = parser.parse_args()
    evidence = retrieve(read_passages(args.input), args.question)
    answer = ollama_answer(args.question, evidence) if args.generator == "ollama" else extractive_answer(args.question, evidence)
    result = {"question": args.question, "generator": args.generator, "answer": answer, "evidence": evidence}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(answer.encode("ascii", "backslashreplace").decode("ascii"))
    print(f"Saved {len(evidence)} source passages to {args.output}")


if __name__ == "__main__":
    main()
