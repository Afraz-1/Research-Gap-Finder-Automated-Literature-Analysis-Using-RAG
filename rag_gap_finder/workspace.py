"""Persistent, auditable review workspace for Research Evidence Explorer."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def database(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL, question TEXT NOT NULL,
                inclusion TEXT NOT NULL, exclusion TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS search_runs (
                id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id),
                source TEXT NOT NULL, query TEXT NOT NULL, filters_json TEXT NOT NULL,
                executed_at TEXT NOT NULL, status TEXT NOT NULL,
                fetched_count INTEGER NOT NULL DEFAULT 0, new_count INTEGER NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id),
                identity_key TEXT NOT NULL, title TEXT NOT NULL, abstract TEXT NOT NULL,
                authors TEXT NOT NULL DEFAULT '', year TEXT NOT NULL DEFAULT '',
                doi TEXT NOT NULL DEFAULT '', url TEXT NOT NULL DEFAULT '',
                openalex_id TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
                UNIQUE(project_id, identity_key)
            );
            CREATE TABLE IF NOT EXISTS run_papers (
                run_id INTEGER NOT NULL REFERENCES search_runs(id),
                paper_id INTEGER NOT NULL REFERENCES papers(id),
                PRIMARY KEY (run_id, paper_id)
            );
            CREATE TABLE IF NOT EXISTS screening (
                paper_id INTEGER PRIMARY KEY REFERENCES papers(id),
                decision TEXT NOT NULL CHECK(decision IN ('include','exclude','unsure')),
                reason TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id),
                source_paper_id INTEGER NOT NULL REFERENCES papers(id),
                source_sentence TEXT NOT NULL, start_offset INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'suggested'
                    CHECK(status IN ('suggested','accepted','rejected')),
                countercheck TEXT NOT NULL DEFAULT 'pending'
                    CHECK(countercheck IN ('pending','checked')),
                reviewer_note TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(project_id, source_paper_id, source_sentence)
            );
            CREATE INDEX IF NOT EXISTS idx_papers_project ON papers(project_id);
            CREATE INDEX IF NOT EXISTS idx_candidates_project ON candidates(project_id);
        """)
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def create_project(con: sqlite3.Connection, name: str, question: str,
                   inclusion: str, exclusion: str) -> int:
    if not name.strip() or not question.strip():
        raise ValueError("Project name and research question are required.")
    cur = con.execute(
        "INSERT INTO projects(name,question,inclusion,exclusion,created_at) VALUES(?,?,?,?,?)",
        (name.strip(), question.strip(), inclusion.strip(), exclusion.strip(), now()),
    )
    return int(cur.lastrowid)


def list_projects(con: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in con.execute("SELECT * FROM projects ORDER BY id DESC")]


def create_search_run(con: sqlite3.Connection, project_id: int, source: str,
                      query: str, filters: dict) -> int:
    cur = con.execute(
        "INSERT INTO search_runs(project_id,source,query,filters_json,executed_at,status) "
        "VALUES(?,?,?,?,?,'running')",
        (project_id, source, query, json.dumps(filters, sort_keys=True), now()),
    )
    return int(cur.lastrowid)


def finish_search_run(con: sqlite3.Connection, run_id: int, status: str,
                      fetched: int = 0, added: int = 0, error: str = "") -> None:
    con.execute(
        "UPDATE search_runs SET status=?, fetched_count=?, new_count=?, error=? WHERE id=?",
        (status, fetched, added, error[:500], run_id),
    )


def list_search_runs(con: sqlite3.Connection, project_id: int) -> list[dict]:
    return [dict(r) for r in con.execute(
        "SELECT * FROM search_runs WHERE project_id=? ORDER BY id DESC", (project_id,)
    )]


def _identity(record: dict) -> str:
    doi = str(record.get("doi") or "").lower().strip()
    doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    if doi:
        return "doi:" + doi
    openalex = str(record.get("openalex_id") or "").strip().lower()
    if openalex:
        return "openalex:" + openalex.rsplit("/", 1)[-1]
    title = " ".join(str(record.get("title") or "").lower().split())
    return "title:" + title + ":" + str(record.get("year") or "").strip()


def import_papers(con: sqlite3.Connection, project_id: int, run_id: int,
                  records: list[dict]) -> int:
    added = 0
    for record in records:
        title = str(record.get("title") or "").strip()
        abstract = str(record.get("abstract") or "").strip()
        if not title or not abstract:
            continue
        key = _identity(record)
        cur = con.execute(
            "INSERT OR IGNORE INTO papers(project_id,identity_key,title,abstract,authors,year,doi,url,openalex_id,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (project_id, key, title, abstract, str(record.get("authors") or ""),
             str(record.get("year") or ""), str(record.get("doi") or ""),
             str(record.get("url") or ""), str(record.get("openalex_id") or ""), now()),
        )
        added += cur.rowcount
        paper_id = con.execute(
            "SELECT id FROM papers WHERE project_id=? AND identity_key=?", (project_id, key)
        ).fetchone()["id"]
        con.execute("INSERT OR IGNORE INTO run_papers(run_id,paper_id) VALUES(?,?)", (run_id, paper_id))
    return added


def list_papers(con: sqlite3.Connection, project_id: int) -> list[dict]:
    return [dict(r) for r in con.execute(
        "SELECT p.*, COALESCE(s.decision,'unscreened') AS decision, "
        "COALESCE(s.reason,'') AS reason FROM papers p LEFT JOIN screening s ON s.paper_id=p.id "
        "WHERE p.project_id=? ORDER BY p.year DESC,p.id DESC", (project_id,)
    )]


def set_screening(con: sqlite3.Connection, paper_id: int, decision: str,
                  reason: str = "") -> None:
    if decision not in {"include", "exclude", "unsure"}:
        raise ValueError("Invalid screening decision.")
    if decision == "exclude" and not reason.strip():
        raise ValueError("Give a reason when excluding a paper.")
    con.execute(
        "INSERT INTO screening(paper_id,decision,reason,updated_at) VALUES(?,?,?,?) "
        "ON CONFLICT(paper_id) DO UPDATE SET decision=excluded.decision,"
        "reason=excluded.reason,updated_at=excluded.updated_at",
        (paper_id, decision, reason.strip(), now()),
    )


def add_candidate(con: sqlite3.Connection, project_id: int, paper_id: int,
                  sentence: str, start_offset: int) -> None:
    stamp = now()
    con.execute(
        "INSERT OR IGNORE INTO candidates(project_id,source_paper_id,source_sentence,"
        "start_offset,created_at,updated_at) VALUES(?,?,?,?,?,?)",
        (project_id, paper_id, sentence, start_offset, stamp, stamp),
    )


def list_candidates(con: sqlite3.Connection, project_id: int) -> list[dict]:
    return [dict(r) for r in con.execute(
        "SELECT c.*,p.title,p.year,p.doi,p.url,p.abstract FROM candidates c "
        "JOIN papers p ON p.id=c.source_paper_id "
        "JOIN screening s ON s.paper_id=p.id AND s.decision='include' "
        "WHERE c.project_id=? ORDER BY c.id DESC", (project_id,)
    )]


def review_candidate(con: sqlite3.Connection, candidate_id: int, status: str,
                     countercheck: str, note: str) -> None:
    if status not in {"suggested", "accepted", "rejected"}:
        raise ValueError("Invalid candidate status.")
    if countercheck not in {"pending", "checked"}:
        raise ValueError("Invalid counterevidence check status.")
    if status == "accepted" and (countercheck != "checked" or not note.strip()):
        raise ValueError("Acceptance requires a counterevidence check and reviewer note.")
    con.execute(
        "UPDATE candidates SET status=?,countercheck=?,reviewer_note=?,updated_at=? WHERE id=?",
        (status, countercheck, note.strip(), now(), candidate_id),
    )
