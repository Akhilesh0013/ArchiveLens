"""Question answering over retrieved marriage records."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import RAG_QA_PROMPT
from app.extraction.extractor import build_llm
from app.rag.retriever import MarriageRecordRetriever


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)


def _format_context(documents: list[Document]) -> str:
    blocks: list[str] = []
    for doc in documents:
        blocks.append(
            f"--- Record ID {doc.metadata.get('record_id')} ---\n{doc.page_content}"
        )
    return "\n\n".join(blocks)


class MarriageRecordQA:
    def __init__(self, retriever: MarriageRecordRetriever | None = None) -> None:
        self.retriever = retriever or MarriageRecordRetriever()
        self.llm = build_llm()

    def ask(self, question: str, *, k: int | None = None) -> RAGAnswer:
        documents = self.retriever.hybrid_search(question, k=k)
        if not documents:
            return RAGAnswer(
                question=question,
                answer="No matching marriage records found in the database.",
                sources=[],
            )

        context = _format_context(documents)
        prompt = RAG_QA_PROMPT.format(context=context, question=question)
        response = self.llm.invoke(
            [
                SystemMessage(content="You answer genealogical questions with citations."),
                HumanMessage(content=prompt),
            ]
        )

        sources = [
            {
                "record_id": doc.metadata.get("record_id"),
                "groom": doc.metadata.get("groom_name"),
                "bride": doc.metadata.get("bride_name"),
                "year": doc.metadata.get("marriage_year"),
                "source_pdf": doc.metadata.get("source_pdf"),
                "source_page": doc.metadata.get("source_page"),
            }
            for doc in documents
        ]

        return RAGAnswer(
            question=question,
            answer=str(response.content).strip(),
            sources=sources,
        )
