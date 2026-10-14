"""Vector index over SQLite marriage records."""

from __future__ import annotations

from langchain_chroma import Chroma

from app.config import corpus_collection_name, corpus_rag_index_dir, corpus_preferred_engine, normalize_corpus
from app.db.repository import RecordRepository
from app.rag.documents import record_to_document
from app.rag.embeddings import build_embeddings


def get_vectorstore(*, corpus: str = "docling") -> Chroma:
    name = normalize_corpus(corpus)
    index_dir = corpus_rag_index_dir(name)
    index_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=corpus_collection_name(name),
        embedding_function=build_embeddings(),
        persist_directory=str(index_dir),
    )


def rebuild_index(
    repository: RecordRepository | None = None,
    *,
    corpus: str = "docling",
) -> int:
    """Index canonical marriage records from SQLite into Chroma."""
    name = normalize_corpus(corpus)
    repo = repository or RecordRepository()
    repo.backfill_search_columns()
    records = repo.iter_canonical_records(
        preferred_engine=corpus_preferred_engine(name),
    )

    from app.rag.sparse import rebuild_sparse_index

    rebuild_sparse_index(repo, corpus=name, records=records)
    if not records:
        return 0

    store = get_vectorstore(corpus=name)
    existing = store.get(include=[])
    if existing and existing.get("ids"):
        store.delete(ids=existing["ids"])

    documents = [record_to_document(item) for item in records]
    ids = [str(item.id) for item in records]
    store.add_documents(documents, ids=ids)
    return len(documents)


def index_count(*, corpus: str = "docling") -> int:
    store = get_vectorstore(corpus=corpus)
    data = store.get(include=[])
    return len(data.get("ids") or [])
