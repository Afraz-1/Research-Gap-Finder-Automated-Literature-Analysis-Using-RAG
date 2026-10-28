"""Research Evidence Explorer V1: reproducible abstract review workspace."""

from __future__ import annotations

import os
import json
import hashlib
from collections import Counter
from pathlib import Path

import streamlit as st

from rag_gap_finder.discovery import read_csv, search_openalex
from rag_gap_finder.evidence import discover_candidates, related_papers
from rag_gap_finder.workspace import (
    create_project, create_search_run, database, finish_search_run, import_papers,
    list_candidates, list_papers, list_projects, list_search_runs,
    review_candidate, set_screening,
)


DB_PATH = Path(os.environ.get("REE_DB_PATH", "data/research_evidence.sqlite3"))
SAMPLE_PATH = Path(__file__).parent / "data" / "ai_education_papers.csv"


def report_markdown(project: dict, papers: list[dict], runs: list[dict],
                    candidates: list[dict]) -> str:
    counts = Counter(p["decision"] for p in papers)
    lines = [
        f"# {project['name']}", "", f"**Research question:** {project['question']}", "",
        "## Review scope", "", f"Inclusion: {project['inclusion'] or 'Not specified'}", "",
        f"Exclusion: {project['exclusion'] or 'Not specified'}", "",
        "This V1 analyzes abstracts only. Its candidates are source-stated research directions, "
        "not proof that a gap exists in the wider literature.", "",
        "## Search record", "",
    ]
    for run in runs:
        lines.append(
            f"- {run['executed_at']} | {run['source']} | query: {run['query']} | "
            f"status: {run['status']} | fetched: {run['fetched_count']} | "
            f"new: {run['new_count']} | filters: {run['filters_json']}"
        )
    lines += ["", "## Screening", "",
              f"{len(papers)} unique papers; {counts['include']} included; "
              f"{counts['exclude']} excluded; {counts['unsure']} unsure; "
              f"{counts['unscreened']} unscreened.", ""]
    if counts["unscreened"] or counts["unsure"]:
        lines += ["**Review incomplete:** some papers are unscreened or unsure. "
                  "The included set and research directions may change.", ""]
    for paper in papers:
        if paper["decision"] == "exclude":
            lines.append(f"- Excluded: {paper['title']} — {paper['reason']}")
    lines += ["", "## Reviewed research directions", ""]
    accepted = [c for c in candidates if c["status"] == "accepted"]
    if not accepted:
        lines.append("No candidates have been accepted by a reviewer.")
    for candidate in accepted:
        source = candidate["doi"] or candidate["url"] or "source link unavailable"
        lines += [
            f"### Candidate {candidate['id']}", "",
            f"**Source statement:** {candidate['source_sentence']}", "",
            f"**Source:** {candidate['title']} ({candidate['year']}); {source}; abstract.", "",
            f"**Reviewer assessment:** {candidate['reviewer_note']}", "",
            "Counterevidence review marked complete by the reviewer. This status records a "
            "review action; it does not establish that no contrary study exists.", "",
        ]
    return "\n".join(lines)


st.set_page_config(page_title="Research Evidence Explorer", layout="wide")
st.title("Research Evidence Explorer")
st.caption("A traceable workspace for investigating research directions from literature")

with database(DB_PATH) as con:
    projects = list_projects(con)

with st.sidebar:
    st.header("Workspace")
    with st.expander("Create research project", expanded=not projects):
        with st.form("new_project"):
            name = st.text_input("Project name")
            question = st.text_area("Research question")
            inclusion = st.text_area("Inclusion criteria", placeholder="Topic, study type, population...")
            exclusion = st.text_area("Exclusion criteria", placeholder="Out of scope, duplicates...")
            submitted = st.form_submit_button("Create project")
        if submitted:
            try:
                with database(DB_PATH) as con:
                    project_id = create_project(con, name, question, inclusion, exclusion)
                st.session_state["selected_project"] = project_id
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

if not projects:
    st.info("Create a research project in the sidebar to begin.")
    st.stop()

ids = [p["id"] for p in projects]
selected_id = st.session_state.get("selected_project")
if selected_id not in ids:
    selected_id = ids[0]
selected_id = st.sidebar.selectbox(
    "Current project", ids, index=ids.index(selected_id),
    format_func=lambda value: next(p["name"] for p in projects if p["id"] == value),
)
st.session_state["selected_project"] = selected_id
project = next(p for p in projects if p["id"] == selected_id)

with database(DB_PATH) as con:
    papers = list_papers(con, selected_id)
    runs = list_search_runs(con, selected_id)
    candidates = list_candidates(con, selected_id)

st.subheader(project["name"])
st.write(project["question"])
counts = Counter(p["decision"] for p in papers)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Papers", len(papers))
m2.metric("Included", counts["include"])
m3.metric("Unscreened", counts["unscreened"])
m4.metric("Reviewed directions", sum(c["status"] == "accepted" for c in candidates))

search_tab, screen_tab, evidence_tab, report_tab = st.tabs(
    ["1. Search", "2. Screen papers", "3. Investigate directions", "4. Report"]
)

with search_tab:
    st.markdown("### Add papers")
    st.caption("Each search or import is recorded with its query, filters, date, and counts. "
               "OpenAlex searches titles and abstracts of articles.")
    with st.form("openalex_search"):
        query = st.text_input("OpenAlex search query", value=project["question"])
        col1, col2, col3 = st.columns(3)
        from_year = col1.number_input("From year", min_value=1900, max_value=2100, value=2020)
        to_year = col2.number_input("To year", min_value=1900, max_value=2100, value=2026)
        limit = col3.number_input("Maximum results", min_value=1, max_value=500, value=100)
        do_search = st.form_submit_button("Run and save search")
    if do_search:
        filters = {"from_year": int(from_year), "to_year": int(to_year),
                   "limit": int(limit), "scope": "article titles and abstracts"}
        try:
            with database(DB_PATH) as con:
                run_id = create_search_run(con, selected_id, "OpenAlex", query.strip(), filters)
            with st.spinner("Searching OpenAlex..."):
                records, details = search_openalex(query, int(from_year), int(to_year), int(limit))
            with database(DB_PATH) as con:
                added = import_papers(con, selected_id, run_id, records)
                con.execute("UPDATE search_runs SET filters_json=? WHERE id=?",
                            (json.dumps(details, sort_keys=True), run_id))
                finish_search_run(con, run_id, "completed", len(records), added)
            st.success(f"Saved {len(records)} results; {added} new papers.")
            st.rerun()
        except Exception as exc:
            if "run_id" in locals():
                with database(DB_PATH) as con:
                    finish_search_run(con, run_id, "failed", error=str(exc))
            st.error(f"Search failed: {exc}")

    st.markdown("### Import a CSV")
    upload = st.file_uploader("CSV with title and abstract columns", type="csv")
    use_sample = st.button("Import bundled AI-in-education sample")
    import_upload = st.button("Import uploaded CSV", disabled=upload is None)
    if use_sample or import_upload:
        try:
            content = SAMPLE_PATH.read_bytes() if use_sample else upload.getvalue()
            source_name = SAMPLE_PATH.name if use_sample else upload.name
            records = read_csv(content)
            with database(DB_PATH) as con:
                run_id = create_search_run(
                    con, selected_id, "CSV import", source_name,
                    {"filename": source_name, "sha256": hashlib.sha256(content).hexdigest(),
                     "scope": "user-supplied abstracts"},
                )
                added = import_papers(con, selected_id, run_id, records)
                finish_search_run(con, run_id, "completed", len(records), added)
            st.success(f"Imported {added} new papers from {source_name}.")
            st.rerun()
        except (ValueError, UnicodeError, OSError) as exc:
            st.error(str(exc))

    st.markdown("### Search history")
    if runs:
        st.dataframe([{
            "date (UTC)": r["executed_at"], "source": r["source"], "query / file": r["query"],
            "status": r["status"], "fetched": r["fetched_count"], "new": r["new_count"],
            "filters": r["filters_json"], "error": r["error"],
        } for r in runs], hide_index=True, width="stretch")
    else:
        st.info("No searches or imports yet.")

with screen_tab:
    st.markdown("### Review relevance")
    st.caption("Screening is a human decision. Exclusions require a reason; unsure papers remain visible.")
    if not papers:
        st.info("Add papers in the Search tab first.")
    else:
        view = st.selectbox("Show", ["unscreened", "unsure", "include", "exclude", "all"])
        visible = papers if view == "all" else [p for p in papers if p["decision"] == view]
        if not visible:
            st.info("No papers in this view.")
        else:
            selected_paper = st.selectbox(
                "Paper", visible, format_func=lambda p: f"{p['year']} · {p['title']}"
            )
            st.markdown(f"**{selected_paper['title']}**")
            st.caption(f"{selected_paper['authors']} · {selected_paper['year']}")
            if selected_paper["url"]:
                st.link_button("Open source record", selected_paper["url"])
            st.write(selected_paper["abstract"])
            with st.form(f"screen_{selected_paper['id']}"):
                choices = ["include", "exclude", "unsure"]
                current = selected_paper["decision"]
                decision = st.radio("Decision", choices,
                                    index=choices.index(current) if current in choices else 2,
                                    horizontal=True)
                reason = st.text_input("Reason / note", value=selected_paper["reason"])
                save = st.form_submit_button("Save decision")
            if save:
                try:
                    with database(DB_PATH) as con:
                        set_screening(con, selected_paper["id"], decision, reason)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

with evidence_tab:
    st.markdown("### Source-stated research directions")
    st.caption("V1 detects explicit limitation or future-work sentences in included abstracts. "
               "Related papers are similarity matches to inspect, not verified counterevidence.")
    if st.button("Find directions in included papers"):
        with database(DB_PATH) as con:
            added = discover_candidates(con, selected_id)
        st.success(f"Found {added} new directions.")
        st.rerun()
    if not candidates:
        st.info("Include papers in the screening tab, then find directions.")
    else:
        candidate = st.selectbox(
            "Direction", candidates,
            format_func=lambda c: f"#{c['id']} [{c['status']}] {c['source_sentence'][:110]}"
        )
        st.markdown("**Exact source statement**")
        st.info(candidate["source_sentence"])
        st.caption(f"{candidate['title']} ({candidate['year']}) · abstract · "
                   f"character offset {candidate['start_offset']}")
        if candidate["url"]:
            st.link_button("Open source record", candidate["url"])
        st.markdown("**Other included papers to inspect**")
        related = related_papers(candidate, [p for p in papers if p["decision"] == "include"])
        if related:
            for paper in related:
                with st.expander(f"{paper['title']} ({paper['year']}) · similarity {paper['similarity']}"):
                    st.write(paper["abstract"])
                    if paper["url"]:
                        st.link_button("Open source record", paper["url"])
        else:
            st.write("No related included abstract found. Run a broader search before accepting.")
        st.warning("Check the wider literature for counterevidence. This workspace cannot "
                   "establish that a topic is unexplored.")
        with st.form(f"review_{candidate['id']}"):
            countercheck = st.checkbox(
                "I reviewed related papers and searched for counterevidence",
                value=candidate["countercheck"] == "checked",
            )
            status = st.selectbox("Review status", ["suggested", "accepted", "rejected"],
                                  index=["suggested", "accepted", "rejected"].index(candidate["status"]))
            note = st.text_area("Reviewer assessment and any contrary studies",
                                value=candidate["reviewer_note"])
            save_review = st.form_submit_button("Save review")
        if save_review:
            try:
                with database(DB_PATH) as con:
                    review_candidate(con, candidate["id"], status,
                                     "checked" if countercheck else "pending", note)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

with report_tab:
    st.markdown("### Review report")
    report = report_markdown(project, papers, runs, candidates)
    st.download_button("Download Markdown report", report, file_name="research_evidence_report.md",
                       mime="text/markdown")
    st.markdown(report)
