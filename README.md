# Research Evidence Explorer V1

The new review workspace is in [`research_app.py`](research_app.py). It saves search runs,
screens papers, extracts source-stated research directions from included abstracts, and
exports a reviewable report. Start it with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run research_app.py --server.address 127.0.0.1
```

See [`docs/V1.md`](docs/V1.md) for the V1 workflow, validation, and scope. The syllabus
demonstration remains below and can still be run with `app.py`.

# Research Gap Finder: syllabus demonstration

An Introduction to Text Analysis mini project. The system reads a focused set of research papers, retrieves relevant evidence for a question, and proposes **candidate** research gaps with links back to the source papers. A low frequency topic or missing keyword is a lead for review, not proof that a gap exists.

The included demonstration corpus contains 24 real AI-in-education article abstracts fetched from [OpenAlex](https://help.openalex.org/api/) on 3 October 2026. It has six papers in each of four title-curated groups: generative AI, learning analytics, intelligent tutoring, and ethics/policy. `data/ai_education_papers.csv` keeps the paper title, year, DOI/OpenAlex URL, abstract, and group label. It is a teaching sample, not a systematic review.

## Four-day build plan

| Day | Module | Course concepts | Demonstration |
| --- | --- | --- | --- |
| 1 | Corpus + preprocessing | PDF/abstract extraction, cleaning, tokenization, stopwords, lemmatization | Show raw text beside cleaned tokens and source location |
| 2 | Represent and organize | BoW, TF-IDF, Word2Vec; Naive Bayes/SVM/Decision Tree on labeled papers; hierarchical/DBSCAN clustering; LDA/NMF topics | Compare words, predicted categories, clusters, and topics |
| 3 | Retrieval + grounded answer | TF-IDF, cosine similarity, NER, TextRank and local transformer synthesis | Ask a question and inspect ranked source passages |
| 4 | Candidate gaps + evaluation + demo | RAG, aspect coverage, simple aspect stance, citations, manual validation | Explain why each suggestion follows from retrieved evidence and where the evidence is weak |

The classification component needs category labels. The included corpus uses title-curated groups; its scores are illustrative because the labels are selected by topical search and the held-out set contains only eight papers. Product-review sentiment is a course example rather than a natural fit for research papers; this project adapts it to **aspect-based stance** in abstract sentences. The keyword stance labels are exploratory, not validated sentiment judgments.

## Module 1: build a traceable corpus

Each record keeps its original wording for quotations and a separate cleaned version for text analysis. A CSV row is one abstract passage. A PDF is split by page, so a retrieved passage can cite its page number. Scanned PDFs need OCR before this module can read them.

### Input choices

Use a CSV with `title` and `abstract` columns, plus optional `authors`, `year`, `doi`, and `url`. Or put text-based PDFs in one directory. Keep all papers within one research area so comparisons are meaningful. A small initial target is 15–30 papers, ideally with titles, abstracts, and years; quality matters more than volume.

### Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m nltk.downloader wordnet averaged_perceptron_tagger_eng punkt_tab maxent_ne_chunker_tab words
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Use the project `.venv` interpreter for every command. This keeps the project packages separate from Anaconda, TensorFlow, and other projects. The app builds its analysis from the included CSV automatically. The NLTK download is only needed once. You can replace the corpus path with your own CSV or a directory of PDFs. The app also accepts one CSV or multiple PDFs through its sidebar. The preprocessing output is JSON Lines: one line per passage with paper metadata, source location, raw text, normalized text, and tokens. `data/sample_papers.csv` is synthetic and intended only for testing.

To run the modules individually:

```powershell
.\.venv\Scripts\python.exe -m rag_gap_finder.preprocess --input data/ai_education_papers.csv --output data/processed/ai_education_passages.jsonl
.\.venv\Scripts\python.exe -m rag_gap_finder.analyze
.\.venv\Scripts\python.exe -m rag_gap_finder.gaps
```

To refresh the real corpus, run `.\.venv\Scripts\python.exe -m rag_gap_finder.fetch_papers` before preprocessing. [OpenAlex authentication guidance](https://help.openalex.org/api/authentication/) says basic queries can work without a key; if needed, set the `OPENALEX_API_KEY` environment variable first. Refreshing changes the paper sample and therefore the model outputs.

### What to say in the viva

1. **Cleaning** removes URLs, citation markers, and punctuation that usually do not help word matching.
2. **Tokenization** turns text into a list of words.
3. **Stopword removal** removes common words such as “the” while keeping content words.
4. **Lemmatization** reduces grammatical forms: “studies” → “study”, “using” → “use”.
5. We retain raw text because citations and readable answers must show the paper's original wording. Preprocessing changes a representation; it does not rewrite the source.

The tokenizer is English-oriented. PDF reading order can be imperfect, especially in two-column papers. We will inspect a few outputs before using them as evidence.

## Module 2: representations and models

- **Bag of Words** counts each token per paper. It ignores order and meaning, but gives a transparent frequency baseline.
- **TF-IDF** raises the weight of a word when it is common in one paper and rarer across the collection. Its vectors also power retrieval.
- **Word2Vec** learns which words appear in similar contexts. This corpus is too small for stable semantic claims; the neighbor list is a demonstration.
- **Classification** predicts one of four supplied topic labels. Naive Bayes uses word probabilities, linear SVM finds a separating boundary, and a decision tree splits by features. The code fits TF-IDF inside each training pipeline to avoid test leakage. Compare macro F1 as well as accuracy; treat the eight-paper test as illustrative.
- **Clustering** groups papers without labels. Hierarchical clustering always assigns a group; DBSCAN can mark papers as noise when its density threshold finds no close neighbors. These groups need not equal the supplied labels.
- **Topic modeling** uses LDA on counts and NMF on TF-IDF. The displayed top words help a person interpret topics; topic numbers have no inherent meaning.

## Module 3: retrieval augmented generation

```text
question → preprocess → TF-IDF/cosine ranking → top source passages
                                              ↓
                              TextRank + NER + answer generator
                                              ↓
                               answer with numbered evidence
```

`.\.venv\Scripts\python.exe -m rag_gap_finder.retrieve "What privacy concerns arise in AI education?"` gives a deterministic answer made from retrieved TextRank sentences. Add `--generator ollama` to ask the locally installed `llama3.2` transformer to synthesize the evidence. This requires Ollama to be running. In the app, choose the answer method in tab 3. The generator sees query-focused sentences from at most five numbered passages; the app streams its answer as it writes. On a CPU, the first words can still take several seconds. **Always open the evidence and check its claims:** a language model can still misread or overstate an abstract.

NER extracts candidate people, organizations, and places. NLTK's generic NER may mislabel domain terms such as “ChatGPT,” so these are aids to reading rather than trusted metadata.

## Module 4: gap hypotheses

The project looks for mentions of five aspects: learning outcomes, privacy/fairness, teacher role, long-term evaluation, and accessibility/equity. For each topical group, it counts how many abstracts mention each aspect, extracts explicit limitation sentences, and labels the tone of one matching sentence with a simple positive/negative word heuristic. Low counts become **questions to check**, not declarations that research does not exist. In tab 4, choose a hypothesis and retrieve relevant papers; the optional local model can synthesize those passages with citations. A proper claim requires a broader search, full-text reading, and checking newer work.

For example, if zero of six learning-analytics abstracts mention privacy, you can say: “Privacy was absent from the six selected abstracts, so I would investigate privacy in a larger learning-analytics corpus.” You cannot say: “Nobody has researched privacy in learning analytics.”

## What each course experiment contributes

| Experiment | Where to show it |
| --- | --- |
| 1. Preprocessing | App tab 1: original text beside cleaned tokens |
| 2. BoW, TF-IDF, Word2Vec | Tab 2: term weights and word neighbors |
| 3. Naive Bayes, SVM, Decision Tree | Tab 2: held-out predictions and metrics |
| 4. Hierarchical and DBSCAN | Tab 2: cluster IDs and DBSCAN noise |
| 5. LDA and NMF | Tab 2: top words per topic |
| 6. TF-IDF, cosine, NER | Tab 3: ranked passages, scores, and entities |
| 7. Aspect-based sentiment | Tab 4: aspect sentence stance; explain why it is only a heuristic for papers |
| 8. TextRank and transformers | Tab 3: extractive summary and optional local Llama synthesis |

## Four-day study and presentation schedule

1. **Day 1:** Run preprocessing. Explain every transformation using one before/after abstract.
2. **Day 2:** Explore tab 2. Practice explaining why TF-IDF differs from BoW and why supervised labels differ from clusters/topics.
3. **Day 3:** Ask three questions in tab 3. Trace scores, passages, TextRank summaries, entities, and citations. Test the local Llama option if available.
4. **Day 4:** Review tab 4. Pick two candidate gaps, inspect their linked abstracts, and prepare to explain the limits of keyword counts and a small sample.

Run `.\.venv\Scripts\python.exe -m pytest -q` for the small verification suite. Do not present the 24-abstract sample or model scores as a comprehensive literature review.

The next module will fit BoW/TF-IDF and Word2Vec **only on the chosen corpus**, then add visual comparisons and models. The retrieval module will rank passages with cosine similarity. The final module will generate a cautious, cited synthesis from those results and make it easy to review the suggested gaps.
