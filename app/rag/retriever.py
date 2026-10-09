"""Retrieve marriage records for RAG."""

from __future__ import annotations

import re

from langchain_core.documents import Document

from app.config import corpus_db_path, normalize_corpus, settings
from app.db.repository import RecordRepository, StoredRecord
from app.rag.index import get_vectorstore

_OCCUPATION_ALIASES = frozenset(
    {
        "musician",
        "musicians",
        "laborer",
        "laborers",
        "mechanic",
        "mechanics",
        "beautician",
        "beauticians",
        "janitor",
        "janitors",
        "boiler",
        "boilers",
        "presser",
        "pressers",
        "occupation",
    }
)

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "about",
        "any",
        "are",
        "as",
        "at",
        "by",
        "couples",
        "find",
        "for",
        "from",
        "his",
        "her",
        "husband",
        "in",
        "list",
        "marriage",
        "marriages",
        "married",
        "named",
        "of",
        "on",
        "or",
        "person",
        "records",
        "show",
        "the",
        "their",
        "what",
        "when",
        "where",
        "which",
        "who",
        "wife",
        "with",
    }
)


def _year_from_query(query: str) -> int | None:
    match = re.search(r"\b(19[0-4]\d)\b", query)
    return int(match.group(1)) if match else None


def _occupation_from_query(query: str) -> str | None:
    lower = query.lower()
    aliases = {
        "musicians": "MUSICIAN",
        "musician": "MUSICIAN",
        "laborers": "LABORER",
        "laborer": "LABORER",
        "mechanics": "MECHANIC",
        "mechanic": "MECHANIC",
        "beauticians": "BEAUTICIAN",
        "beautician": "BEAUTICIAN",
        "janitors": "JANITOR",
        "janitor": "JANITOR",
        "boilers": "BOILER",
        "boiler": "BOILER",
        "pressers": "PRESSER",
        "presser": "PRESSER",
    }
    for keyword, normalized in aliases.items():
        if keyword in lower:
            return normalized
    return None


def _name_tokens_from_query(query: str) -> list[str]:
    """Extract likely person-name tokens from a free-text query."""
    tokens: list[str] = []
    for match in re.finditer(r"\b([A-Za-z][A-Za-z'.-]*(?: [A-Za-z][A-Za-z'.-]*){0,3})\b", query):
        phrase = match.group(1).strip(" .'")
        words = [w for w in phrase.split() if w.lower() not in _STOP_WORDS]
        if not words:
            continue
        if all(w.lower() in _OCCUPATION_ALIASES for w in words):
            continue
        if len(words) == 1 and (
            len(words[0]) < 3 or words[0].lower() in _OCCUPATION_ALIASES
        ):
            continue
        cleaned = " ".join(words)
        tokens.append(cleaned.upper())
        if len(words) >= 2:
            tokens.append(words[-1].upper())
    return list(dict.fromkeys(tokens))


def _provenance_key(doc: Document) -> tuple[str, int]:
    return (
        str(doc.metadata.get("source_pdf") or ""),
        int(doc.metadata.get("source_page") or 0),
    )


class MarriageRecordRetriever:
    """Hybrid retrieval: structured SQL filters + semantic vector search."""

    def __init__(
        self,
        repository: RecordRepository | None = None,
        *,
        corpus: str = "docling",
    ) -> None:
        self.corpus = normalize_corpus(corpus)
        self.repository = repository or RecordRepository(corpus_db_path(self.corpus))

    def structured_search(self, query: str, *, limit: int = 10) -> list[StoredRecord]:
        year = _year_from_query(query)
        occupation = _occupation_from_query(query)
        name_tokens = _name_tokens_from_query(query)

        if not year and not occupation and not name_tokens:
            return []

        seen: set[int] = set()
        results: list[StoredRecord] = []

        def collect(items: list[StoredRecord]) -> None:
            for item in items:
                if item.id not in seen:
                    results.append(item)
                    seen.add(item.id)

        if year or occupation:
            collect(
                self.repository.search_records(
                    marriage_year=year,
                    occupation_substr=occupation,
                    limit=limit,
                )
            )

        for token in name_tokens:
            collect(
                self.repository.search_records(
                    name_substr=token,
                    marriage_year=year,
                    limit=limit,
                )
            )

        return results[:limit]

    def semantic_search(self, query: str, *, k: int | None = None) -> list[Document]:
        k = k or settings.rag_top_k
        if index_count_safe(self.corpus) == 0:
            rebuild_if_needed(self.repository, self.corpus)
        return get_vectorstore(corpus=self.corpus).similarity_search(query, k=k)

    def hybrid_search(self, query: str, *, k: int | None = None) -> list[Document]:
        k = k or settings.rag_top_k
        structured = self.structured_search(query, limit=k)
        semantic = self.semantic_search(query, k=k)

        docs: list[Document] = []
        seen_ids: set[int] = set()
        seen_pages: set[tuple[str, int]] = set()

        def add_doc(doc: Document) -> None:
            record_id = int(doc.metadata["record_id"])
            page_key = _provenance_key(doc)
            if record_id in seen_ids or page_key in seen_pages:
                return
            seen_ids.add(record_id)
            seen_pages.add(page_key)
            docs.append(doc)

        for item in structured:
            add_doc(_stored_to_doc(item))

        for doc in semantic:
            add_doc(doc)

        return docs[:k]


def _stored_to_doc(item: StoredRecord) -> Document:
    from app.rag.documents import record_to_document

    return record_to_document(item)


def index_count_safe(corpus: str = "docling") -> int:
    try:
        from app.rag.index import index_count

        return index_count(corpus=corpus)
    except Exception:
        return 0


def rebuild_if_needed(repository: RecordRepository, corpus: str = "docling") -> int:
    from app.rag.index import rebuild_index

    if repository.count_records() == 0:
        return 0
    return rebuild_index(repository, corpus=corpus)
