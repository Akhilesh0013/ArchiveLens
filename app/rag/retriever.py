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


_RRF_K = 60


def _identifier_tokens(query: str) -> list[str]:
    """Digit runs long enough to be a license, application number, or year."""
    return re.findall(r"\b\d{4,}\b", query)


def _contains_identifier(text: str, token: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", text) is not None


def fuse_rankings(
    rankings: list[list[Document]],
    query: str,
    k: int,
) -> list[Document]:
    """Reciprocal-rank fusion, with an extra boost for exact identifier hits."""
    scores: dict[int, float] = {}
    chosen: dict[int, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            record_id = int(doc.metadata["record_id"])
            scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            chosen.setdefault(record_id, doc)

    for token in _identifier_tokens(query):
        for record_id, doc in chosen.items():
            if _contains_identifier(doc.page_content, token):
                scores[record_id] += 1.0

    ordered = sorted(scores, key=lambda record_id: scores[record_id], reverse=True)
    docs: list[Document] = []
    seen_ids: set[int] = set()
    seen_pages: set[tuple[str, int]] = set()
    for record_id in ordered:
        doc = chosen[record_id]
        page_key = _provenance_key(doc)
        if record_id in seen_ids or page_key in seen_pages:
            continue
        seen_ids.add(record_id)
        seen_pages.add(page_key)
        docs.append(doc)
        if len(docs) >= k:
            break
    return docs


class MarriageRecordRetriever:
    """Hybrid retrieval: structured filters + BM25 sparse search + dense vectors."""

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

    def sparse_search(self, query: str, *, k: int | None = None) -> list[Document]:
        from app.rag.sparse import sparse_search

        k = k or settings.rag_top_k
        return sparse_search(self.repository, query, corpus=self.corpus, k=k)

    def semantic_search(self, query: str, *, k: int | None = None) -> list[Document]:
        k = k or settings.rag_top_k
        if index_count_safe(self.corpus) == 0:
            rebuild_if_needed(self.repository, self.corpus)
        return get_vectorstore(corpus=self.corpus).similarity_search(query, k=k)

    def hybrid_search(self, query: str, *, k: int | None = None) -> list[Document]:
        k = k or settings.rag_top_k
        structured = [
            _stored_to_doc(item) for item in self.structured_search(query, limit=k)
        ]
        sparse = self.sparse_search(query, k=k)
        semantic = self.semantic_search(query, k=k)
        rankings = [docs for docs in (structured, sparse, semantic) if docs]
        if not rankings:
            return []
        return fuse_rankings(rankings, query, k)


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
