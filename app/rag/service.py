"""High-level RAG API — independent of OCR/agent pipelines."""

from __future__ import annotations

from app.config import corpus_db_path, normalize_corpus
from app.db.repository import RecordRepository
from app.rag.index import index_count, rebuild_index
from app.rag.qa import MarriageRecordQA, RAGAnswer
from app.rag.retriever import MarriageRecordRetriever


class RAGService:
    """Search and Q&A over ingested marriage records in SQLite."""

    def __init__(
        self,
        repository: RecordRepository | None = None,
        *,
        corpus: str = "docling",
    ) -> None:
        self.corpus = normalize_corpus(corpus)
        self.repository = repository or RecordRepository(corpus_db_path(self.corpus))
        self.retriever = MarriageRecordRetriever(self.repository, corpus=self.corpus)
        self.qa = MarriageRecordQA(self.retriever)

    def rebuild_index(self) -> int:
        return rebuild_index(self.repository, corpus=self.corpus)

    def indexed_count(self) -> int:
        try:
            return index_count(corpus=self.corpus)
        except Exception:
            return 0

    def db_count(self) -> int:
        return self.repository.count_records()

    def ask(self, question: str, *, k: int | None = None) -> RAGAnswer:
        if self.indexed_count() == 0 and self.db_count() > 0:
            self.rebuild_index()
        return self.qa.ask(question, k=k)

    def search(self, query: str, *, k: int | None = None) -> list[dict]:
        if self.indexed_count() == 0 and self.db_count() > 0:
            self.rebuild_index()
        docs = self.retriever.hybrid_search(query, k=k)
        return [
            {
                "record_id": doc.metadata.get("record_id"),
                "groom": doc.metadata.get("groom_name"),
                "bride": doc.metadata.get("bride_name"),
                "year": doc.metadata.get("marriage_year"),
                "source_pdf": doc.metadata.get("source_pdf"),
                "source_page": doc.metadata.get("source_page"),
                "snippet": doc.page_content[:300],
            }
            for doc in docs
        ]
