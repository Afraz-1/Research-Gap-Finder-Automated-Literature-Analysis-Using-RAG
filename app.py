"""Interactive walkthrough for the Introduction to Text Analysis mini project."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from rag_gap_finder.analyze import analyze, read_passages
from rag_gap_finder.gaps import inspect_gaps
from rag_gap_finder.preprocess import load_corpus
from rag_gap_finder.retrieve import extractive_answer, ollama_answer_stream, retrieve


st.set_page_config(page_title="Research Gap Finder", layout="wide")
st.title("Research Gap Finder")
st.caption("Automated literature analysis using retrieval augmented generation • AI in education demo")


@st.cache_data(show_spinner="Analyzing the paper collection...")
def built_in_data():
    processed = Path("data/processed/ai_education_passages.jsonl")
    passages = (read_passages(processed) if processed.exists()
                else [vars(p) for p in load_corpus(Path("data/ai_education_papers.csv"))])
    return passages, analyze(passages), inspect_gaps(passages)


uploaded = st.sidebar.file_uploader(
    "Optional: use your own CSV or text-based PDFs", type=["csv", "pdf"], accept_multiple_files=True
)
st.sidebar.caption("CSV needs title and abstract columns. Upload one CSV or several PDFs.")
if uploaded:
    kinds = {Path(file.name).suffix.lower() for file in uploaded}
    if kinds == {".csv"} and len(uploaded) != 1 or len(kinds) > 1:
        st.error("Choose one CSV, or choose PDFs only.")
        st.stop()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for file in uploaded:
            (root / Path(file.name).name).write_bytes(file.getvalue())
        input_path = root / uploaded[0].name if kinds == {".csv"} else root
        try:
            passages = [vars(p) for p in load_corpus(input_path)]
            analysis = analyze(passages)
            gaps = inspect_gaps(passages)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
    st.sidebar.success(f"Loaded {analysis['paper_count']} papers")
else:
    try:
        passages, analysis, gaps = built_in_data()
    except FileNotFoundError:
        st.error("Run the preprocessing command in README.md before opening the app.")
        st.stop()

corpus_tab, methods_tab, ask_tab, gaps_tab = st.tabs(
    ["1. Corpus and preprocessing", "2. Text analysis", "3. Retrieval and RAG", "4. Candidate gaps"]
)

with corpus_tab:
    st.subheader("What the pipeline reads")
    c1, c2, c3 = st.columns(3)
    c1.metric("Papers", analysis["paper_count"])
    c2.metric("Passages", len(passages))
    c3.metric("Unique terms", analysis["vocabulary_size"])
    st.dataframe(
        pd.DataFrame(analysis["papers"])[["title", "year", "topic_label", "token_count", "url"]],
        width="stretch", hide_index=True,
    )
    selected = st.selectbox("Inspect a passage", range(len(passages)),
                            format_func=lambda i: f"{passages[i]['title']} — {passages[i]['location']}")
    passage = passages[selected]
    left, right = st.columns(2)
    left.markdown("**Original wording**")
    left.write(passage["raw_text"])
    right.markdown("**After cleaning, tokenization, stopword removal, and lemmatization**")
    right.code(" ".join(passage["tokens"][:100]), language=None)
    st.info("The original wording stays available for citations. The tokens are used for statistical models.")

with methods_tab:
    st.subheader("Representations and models")
    st.caption("BoW counts word occurrences; TF-IDF emphasizes distinctive words. Word2Vec learns context-based word vectors.")
    selected_paper = st.selectbox("Inspect one paper's vectors", analysis["papers"],
                                  format_func=lambda p: p["title"])
    left, right = st.columns(2)
    left.markdown("**Top Bag-of-Words terms**")
    left.dataframe(pd.DataFrame(selected_paper["bow_terms"]), hide_index=True)
    right.markdown("**Top TF-IDF terms**")
    right.dataframe(pd.DataFrame(selected_paper["tfidf_terms"]), hide_index=True)
    st.markdown("**Word2Vec neighbors (illustrative on a small corpus)**")
    word = st.selectbox("Word", list(analysis["word2vec_neighbors"]))
    st.dataframe(pd.DataFrame(analysis["word2vec_neighbors"][word]), hide_index=True)
    left, right = st.columns(2)
    left.markdown("**LDA topics (probabilistic, BoW)**")
    left.dataframe(pd.DataFrame({"topic": range(len(analysis["lda_topics"])),
                                "words": [", ".join(t) for t in analysis["lda_topics"]]}), hide_index=True)
    right.markdown("**NMF topics (matrix factorization, TF-IDF)**")
    right.dataframe(pd.DataFrame({"topic": range(len(analysis["nmf_topics"])),
                                 "words": [", ".join(t) for t in analysis["nmf_topics"]]}), hide_index=True)
    st.markdown("**Clustering**")
    st.dataframe(pd.DataFrame(analysis["papers"])[["title", "hierarchical_cluster", "dbscan_cluster"]],
                 hide_index=True, width="stretch")
    st.caption(f"DBSCAN marked {analysis['dbscan_noise_count']} papers as noise (-1); its density threshold is sensitive on a small corpus.")
    st.markdown("**Supervised classification (tiny demonstration)**")
    classification = analysis["classification"]
    st.warning(classification["status"])
    if "models" in classification:
        st.dataframe(pd.DataFrame([
            {"model": name, "accuracy": result["accuracy"], "macro F1": result["macro_f1"]}
            for name, result in classification["models"].items()
        ]), hide_index=True)
        model_name = st.selectbox("Inspect held-out predictions", list(classification["models"]))
        st.dataframe(pd.DataFrame(classification["models"][model_name]["predictions"]), hide_index=True)

with ask_tab:
    st.subheader("Ask a question over the papers")
    question = st.text_input("Question", "What privacy concerns arise in AI education?")
    generator = st.radio("Answer method", ["Extractive TextRank", "Local Llama 3.2 transformer"], horizontal=True)
    just_streamed = False
    if st.button("Retrieve evidence and answer", type="primary"):
        try:
            with st.spinner("Retrieving and analyzing passages..."):
                evidence = retrieve(passages, question, top_k=5)
            if generator.startswith("Local"):
                with st.spinner("Llama is reading evidence on the CPU. Words will appear as it writes..."):
                    answer = st.write_stream(ollama_answer_stream(question, evidence))
                just_streamed = True
            else:
                answer = extractive_answer(question, evidence)
            st.session_state["last_result"] = (answer, evidence)
        except Exception as exc:
            st.error(str(exc))
    if "last_result" in st.session_state:
        answer, evidence = st.session_state["last_result"]
        st.markdown("**Answer**")
        if not just_streamed:
            st.write(answer)
        st.markdown("**Ranked evidence — verify the citations**")
        for item in evidence:
            with st.expander(f"[{item['rank']}] {item['title']} • cosine {item['score']:.3f}"):
                st.write(f"{item['year']} • {item['location']}")
                if item["url"]:
                    st.link_button("Open paper record", item["url"])
                st.markdown("**TextRank summary**")
                st.write(item["summary"])
                st.markdown("**NER candidates**")
                st.write(item["entities"] or "None detected")
                st.markdown("**Original passage**")
                st.write(item["raw_text"])
    st.caption("Retrieval uses TF-IDF and cosine similarity. The local Llama option synthesizes retrieved text; check each citation against the passage.")

with gaps_tab:
    st.subheader("Candidate gap explorer")
    st.warning(gaps["warning"])
    matrix = pd.DataFrame(gaps["coverage"]).T
    matrix.insert(0, "papers in group", [gaps["group_sizes"][group] for group in matrix.index])
    st.markdown("**Aspect mentions per topic group**")
    st.dataframe(matrix, width="stretch")
    st.markdown("**Hypotheses to check with a broader literature search**")
    for candidate in gaps["candidates"][:10]:
        with st.expander(candidate["hypothesis"]):
            st.write("Sample papers in this group:")
            for paper in candidate["sample_papers"]:
                if paper["url"]:
                    st.markdown(f"- [{paper['title']}]({paper['url']}) ({paper['year']})")
                else:
                    st.write(f"- {paper['title']} ({paper['year']})")
            if candidate["related_limitation"]:
                st.write("Related limitation sentence:", candidate["related_limitation"]["sentence"])
    if gaps["candidates"]:
        selected_gap = st.selectbox(
            "Investigate one hypothesis with retrieval",
            gaps["candidates"],
            format_func=lambda item: f"{item['topic'].replace('_', ' ')}: {item['aspect']}",
        )
        use_local_model = st.checkbox("Use local Llama 3.2 to synthesize the evidence", value=False)
        just_streamed_gap = False
        if st.button("Investigate selected gap"):
            gap_question = (
                f"What do these abstracts say about {selected_gap['aspect']} in "
                f"{selected_gap['topic'].replace('_', ' ')}? What remains uncertain?"
            )
            try:
                with st.spinner("Retrieving and analyzing passages..."):
                    gap_evidence = retrieve(passages, gap_question, top_k=5)
                if use_local_model:
                    with st.spinner("Llama is reading evidence on the CPU. Words will appear as it writes..."):
                        gap_answer = st.write_stream(ollama_answer_stream(gap_question, gap_evidence))
                    just_streamed_gap = True
                else:
                    gap_answer = extractive_answer(gap_question, gap_evidence)
                st.session_state["gap_result"] = (gap_answer, gap_evidence)
            except Exception as exc:
                st.error(str(exc))
        if "gap_result" in st.session_state:
            gap_answer, gap_evidence = st.session_state["gap_result"]
            if not just_streamed_gap:
                st.write(gap_answer)
            for item in gap_evidence:
                label = f"[{item['rank']}] {item['title']} ({item['year']}) — {item['location']}"
                st.markdown(f"- [{label}]({item['url']})" if item["url"] else f"- {label}")
    st.markdown("**Aspect-based stance example**")
    st.caption("A simple keyword heuristic labels positive, negative, mixed, or neutral wording near each aspect; these are not verified sentiment labels.")
    st.dataframe(pd.DataFrame(gaps["aspect_sentences"])[["aspect", "stance", "title", "sentence"]].head(30),
                 hide_index=True, width="stretch")
    st.markdown("**Explicit limitation sentences found in abstracts**")
    st.dataframe(pd.DataFrame(gaps["explicit_limitations"])[["title", "sentence"]].head(30),
                 hide_index=True, width="stretch")
