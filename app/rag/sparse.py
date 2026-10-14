"""BM25 sparse retrieval over marriage-record text (SQLite FTS5).

Dense embeddings miss exact identifiers such as license numbers. This index
keeps the same document text as the vector store and ranks lexical matches
with BM25 so a query like "license number 34178" hits the record that
contains that number.
"""

from __future__ import annotations

import re
import sqlite3

from langchain_core.documents import Document

from app.config import corpus_preferred_engine, normalize_corpus
from app.db.repository import RecordRepository, StoredRecord
from app.rag.documents import record_to_document
from app.schema import MarriageRecord

_FTS_TABLE = "marriage_records_fts"

# Query words and form labels that appear on essentially every record.
# Dropping them leaves the distinctive tokens (a license number, a name).
_QUERY_STOP = frozenset(
    {
        "a",
        "about",
        "age",
        "an",
        "and",
        "any",
        "application",
        "are",
        "as",
        "at",
        "bride",
        "by",
        "color",
        "do",
        "does",
        "find",
        "for",
        "from",
        "groom",
        "her",
        "his",
        "in",
        "is",
        "it",
        "know",
        "license",
        "list",
        "marriage",
        "marriages",
        "married",
        "me",
        "name",
        "named",
        "nativity",
        "number",
        "numbers",
        "occupation",
        "of",
        "officiant",
        "on",
        "or",
        "page",
        "record",
        "records",
        "residence",
        "show",
        "source",
        "status",
        "the",
        "their",
        "this",
        "to",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)


def fts_match_query(query: str) -> str | None:
    """Turn a natural-language question into an FTS5 MATCH expression."""
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9]+", query):
        if raw.isdigit():
            if len(raw) >= 3:
                tokens.append(raw)
            continue
        if len(raw) < 3 or raw.lower() in _QUERY_STOP:
            continue
        tokens.append(raw)

    unique = list(dict.fromkeys(tokens))
    if not unique:
        return None
    return " OR ".join(f'"{token}"' for token in unique)


def _table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (_FTS_TABLE,),
    ).fetchone()
    return row is not None


def _create_table(conn: sqlite3.Connection) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {_FTS_TABLE}")
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE {_FTS_TABLE} USING fts5(
            record_id UNINDEXED,
            body,
            tokenize = 'unicode61 remove_diacritics 2'
        )
        """
    )


def rebuild_sparse_index(
    repository: RecordRepository,
    *,
    corpus: str = "docling",
    records: list[StoredRecord] | None = None,
) -> int:
    """Replace the FTS5 index with the corpus's canonical records."""
    name = normalize_corpus(corpus)
    if records is None:
        records = repository.iter_canonical_records(
            preferred_engine=corpus_preferred_engine(name),
        )

    with repository._connect() as conn:
        _create_table(conn)
        conn.executemany(
            f"INSERT INTO {_FTS_TABLE}(record_id, body) VALUES (?, ?)",
            [(str(item.id), record_to_document(item).page_content) for item in records],
        )
    return len(records)


def ensure_sparse_index(repository: RecordRepository, *, corpus: str = "docling") -> int:
    """Build the sparse index when it is missing or out of date."""
    name = normalize_corpus(corpus)
    records = repository.iter_canonical_records(
        preferred_engine=corpus_preferred_engine(name),
    )
    with repository._connect() as conn:
        current = 0
        if _table_exists(conn):
            current = int(
                conn.execute(f"SELECT COUNT(*) AS n FROM {_FTS_TABLE}").fetchone()["n"]
            )
        if current == len(records) and _table_exists(conn):
            return current
    return rebuild_sparse_index(repository, corpus=name, records=records)


def sync_sparse_row(
    conn: sqlite3.Connection,
    record_id: int,
    record: MarriageRecord,
) -> None:
    """Keep one FTS row aligned with a record write, when the index exists."""
    if not _table_exists(conn):
        return
    body = record_to_document(
        StoredRecord(
            id=record_id,
            upload_id=0,
            filename="",
            record=record,
            uploaded_at="",
            status="",
        )
    ).page_content
    conn.execute(
        f"DELETE FROM {_FTS_TABLE} WHERE record_id = ?",
        (str(record_id),),
    )
    conn.execute(
        f"INSERT INTO {_FTS_TABLE}(record_id, body) VALUES (?, ?)",
        (str(record_id), body),
    )


def sparse_search(
    repository: RecordRepository,
    query: str,
    *,
    corpus: str = "docling",
    k: int = 5,
) -> list[Document]:
    """Return the top BM25 matches for ``query``."""
    match = fts_match_query(query)
    if match is None or k <= 0:
        return []

    ensure_sparse_index(repository, corpus=corpus)
    with repository._connect() as conn:
        if not _table_exists(conn):
            return []
        rows = conn.execute(
            f"""
            SELECT record_id, bm25({_FTS_TABLE}) AS rank
            FROM {_FTS_TABLE}
            WHERE {_FTS_TABLE} MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (match, k),
        ).fetchall()

    documents: list[Document] = []
    for row in rows:
        item = repository.get_record(int(row["record_id"]))
        if item is None:
            continue
        documents.append(record_to_document(item))
    return documents
