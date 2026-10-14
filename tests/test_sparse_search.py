"""BM25 sparse retrieval finds exact identifiers dense search misses."""

import sqlite3
from pathlib import Path

from langchain_core.documents import Document

from app.db.repository import RecordRepository
from app.rag.retriever import MarriageRecordRetriever
from app.rag.sparse import fts_match_query, sparse_search
from app.schema import MarriageRecord, PartyDetails


def _record(**kwargs) -> MarriageRecord:
    defaults = {
        "source_pdf": "1_content.pdf",
        "source_page": 1,
        "ocr_engine": "docling-rapidocr",
        "groom": PartyDetails(full_name="JAMES S CRAWFORD"),
        "bride": PartyDetails(full_name="ALLINE POYAS"),
    }
    defaults.update(kwargs)
    return MarriageRecord(**defaults)


def _seed(tmp_path: Path) -> RecordRepository:
    repo = RecordRepository(tmp_path / "records.db")
    first = repo.create_upload("1.pdf", "1.pdf", status="complete")
    repo.save_record(
        first,
        _record(license_number="No. 34178", application_number="5221"),
    )
    second = repo.create_upload("2.pdf", "2.pdf", status="complete")
    repo.save_record(
        second,
        _record(
            license_number="99999",
            source_pdf="2_content.pdf",
            source_page=2,
            groom=PartyDetails(full_name="JOHN SMITH", occupation="MUSICIAN"),
            bride=PartyDetails(full_name="MARY JONES"),
        ),
    )
    return repo


def test_license_question_keeps_only_the_identifier() -> None:
    assert fts_match_query("What do we know about license number 34178?") == '"34178"'


def test_sparse_search_returns_license_34178(tmp_path: Path) -> None:
    repo = _seed(tmp_path)
    docs = sparse_search(
        repo,
        "What do we know about license number 34178?",
        corpus="docling",
        k=5,
    )
    assert docs
    assert docs[0].metadata["groom_name"] == "JAMES S CRAWFORD"
    assert docs[0].metadata["bride_name"] == "ALLINE POYAS"
    assert "34178" in docs[0].page_content
    assert all("34178" in doc.page_content for doc in docs)


def test_sparse_index_updates_when_a_license_changes(tmp_path: Path) -> None:
    repo = _seed(tmp_path)
    sparse_search(repo, "34178", corpus="docling", k=1)

    upload_id = repo.create_upload("3.pdf", "3.pdf", status="complete")
    record_id = repo.save_record(
        upload_id,
        _record(
            license_number="No. 34178",
            source_pdf="3_content.pdf",
            source_page=3,
            groom=PartyDetails(full_name="UPDATED GROOM"),
            bride=PartyDetails(full_name="UPDATED BRIDE"),
        ),
    )
    repo.save_record(
        upload_id,
        _record(
            license_number="No. 88888",
            source_pdf="3_content.pdf",
            source_page=3,
            groom=PartyDetails(full_name="UPDATED GROOM"),
            bride=PartyDetails(full_name="UPDATED BRIDE"),
        ),
    )

    with sqlite3.connect(repo.db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM marriage_records_fts").fetchone()[0]
        body = conn.execute(
            "SELECT body FROM marriage_records_fts WHERE record_id = ?",
            (str(record_id),),
        ).fetchone()[0]
    assert count == 3
    assert "88888" in body
    assert "34178" not in body

    docs = sparse_search(repo, "license number 88888", corpus="docling", k=3)
    assert any(doc.metadata["record_id"] == record_id for doc in docs)
    assert any(doc.metadata["groom_name"] == "UPDATED GROOM" for doc in docs)


def test_hybrid_ranks_sparse_license_hit_above_dense_neighbors(tmp_path: Path) -> None:
    repo = _seed(tmp_path)
    retriever = MarriageRecordRetriever(repo, corpus="docling")
    decoy = Document(
        page_content="A Baltimore marriage with no matching license number.",
        metadata={
            "record_id": 999,
            "groom_name": "OTHER",
            "bride_name": "PERSON",
            "source_pdf": "other.pdf",
            "source_page": 9,
        },
    )

    def _dense(query: str, *, k: int | None = None) -> list[Document]:
        return [decoy]

    retriever.semantic_search = _dense  # type: ignore[method-assign]
    docs = retriever.hybrid_search(
        "What do we know about license number 34178?",
        k=3,
    )
    assert docs[0].metadata["groom_name"] == "JAMES S CRAWFORD"
    assert "34178" in docs[0].page_content
    assert any(doc.metadata["record_id"] == 999 for doc in docs)
