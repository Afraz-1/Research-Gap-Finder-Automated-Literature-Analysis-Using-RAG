"""Module 2: vectorize, classify, cluster, and model topics in the corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def read_passages(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("No passages found. Run the preprocessing module first.")
    return rows


def group_papers(passages: list[dict]) -> list[dict]:
    papers: dict[str, dict] = {}
    for passage in passages:
        paper = papers.setdefault(
            passage["paper_id"],
            {key: passage[key] for key in ("paper_id", "title", "year", "url", "topic_label")},
        )
        paper.setdefault("tokens", []).extend(passage["tokens"])
    return list(papers.values())


def _top_terms(matrix, names, row: int, count: int = 8) -> list[dict]:
    vector = matrix[row].toarray().ravel()
    indices = np.argsort(vector)[::-1]
    return [{"term": str(names[i]), "weight": round(float(vector[i]), 3)} for i in indices if vector[i] > 0][:count]


def _topic_terms(model, names, count: int = 8) -> list[list[str]]:
    return [[str(names[i]) for i in np.argsort(component)[::-1][:count]] for component in model.components_]


def _classification(papers: list[dict]) -> dict:
    labeled = [p for p in papers if p["topic_label"]]
    counts = Counter(p["topic_label"] for p in labeled)
    if len(counts) < 2 or min(counts.values()) < 3:
        return {"status": "skipped: needs at least 2 classes with 3 labeled papers each"}
    texts = [" ".join(p["tokens"]) for p in labeled]
    labels = [p["topic_label"] for p in labeled]
    indices = list(range(len(labeled)))
    train, test = train_test_split(indices, test_size=max(len(counts), round(0.33 * len(indices))),
                                   random_state=42, stratify=labels)
    algorithms = {
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    }
    results = {}
    for name, algorithm in algorithms.items():
        model = make_pipeline(TfidfVectorizer(), algorithm)
        model.fit([texts[i] for i in train], [labels[i] for i in train])
        truth = [labels[i] for i in test]
        predicted = model.predict([texts[i] for i in test]).tolist()
        results[name] = {
            "accuracy": round(float(accuracy_score(truth, predicted)), 3),
            "macro_f1": round(float(f1_score(truth, predicted, average="macro", zero_division=0)), 3),
            "predictions": [
                {"title": labeled[i]["title"], "actual": labels[i], "predicted": prediction}
                for i, prediction in zip(test, predicted)
            ],
        }
    return {
        "status": "illustrative only: tiny, title-curated labels; held-out data is small",
        "training_papers": len(train),
        "test_papers": len(test),
        "class_counts": dict(counts),
        "models": results,
    }


def analyze(passages: list[dict]) -> dict:
    papers = group_papers(passages)
    if len(papers) < 4:
        raise ValueError("At least 4 papers are needed for clustering and topic comparison.")
    token_lists = [p["tokens"] for p in papers]
    if not any(token_lists):
        raise ValueError("No usable words were extracted from the papers.")
    bow_vectorizer = CountVectorizer(analyzer=lambda tokens: tokens)
    tfidf_vectorizer = TfidfVectorizer(analyzer=lambda tokens: tokens)
    bow = bow_vectorizer.fit_transform(token_lists)
    tfidf = tfidf_vectorizer.fit_transform(token_lists)
    bow_names = bow_vectorizer.get_feature_names_out()
    tfidf_names = tfidf_vectorizer.get_feature_names_out()

    # Word2Vec learns local word context; on 24 abstracts it is a teaching example only.
    w2v = Word2Vec(sentences=token_lists, vector_size=50, window=5, min_count=1,
                   workers=1, seed=42, epochs=100)
    neighbors = {}
    for word in ("student", "feedback", "learning", "privacy", "tutor"):
        if word in w2v.wv:
            neighbors[word] = [
                {"term": term, "similarity": round(float(score), 3)}
                for term, score in w2v.wv.most_similar(word, topn=5)
            ]

    distance = pairwise_distances(tfidf, metric="cosine")
    np.fill_diagonal(distance, 0)
    n_topics = min(4, len(papers), len(bow_names))
    hierarchy = AgglomerativeClustering(n_clusters=n_topics, metric="precomputed", linkage="average")
    hierarchy_labels = hierarchy.fit_predict(distance)
    density_labels = DBSCAN(eps=0.75, min_samples=2, metric="precomputed").fit_predict(distance)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=30).fit(bow)
    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=42, max_iter=500).fit(tfidf)
    lda_weights = lda.transform(bow)
    nmf_weights = nmf.transform(tfidf)

    paper_results = []
    for i, paper in enumerate(papers):
        paper_results.append({
            **{key: paper[key] for key in ("paper_id", "title", "year", "url", "topic_label")},
            "token_count": len(paper["tokens"]),
            "bow_terms": _top_terms(bow, bow_names, i),
            "tfidf_terms": _top_terms(tfidf, tfidf_names, i),
            "hierarchical_cluster": int(hierarchy_labels[i]),
            "dbscan_cluster": int(density_labels[i]),
            "lda_topic": int(np.argmax(lda_weights[i])),
            "nmf_topic": int(np.argmax(nmf_weights[i])),
        })
    return {
        "paper_count": len(papers),
        "vocabulary_size": len(bow_names),
        "papers": paper_results,
        "word2vec_neighbors": neighbors,
        "lda_topics": _topic_terms(lda, bow_names),
        "nmf_topics": _topic_terms(nmf, tfidf_names),
        "dbscan_noise_count": int(sum(label == -1 for label in density_labels)),
        "classification": _classification(papers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore representations and models")
    parser.add_argument("--input", type=Path, default=Path("data/processed/ai_education_passages.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/analysis.json"))
    args = parser.parse_args()
    report = analyze(read_passages(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Analyzed {report['paper_count']} papers, {report['vocabulary_size']} terms")
    print(f"LDA topics: {report['lda_topics']}")
    print(f"NMF topics: {report['nmf_topics']}")
    print(f"DBSCAN noise papers: {report['dbscan_noise_count']}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
