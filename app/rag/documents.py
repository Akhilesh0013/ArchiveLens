"""Turn stored marriage records into searchable documents."""

from __future__ import annotations

from langchain_core.documents import Document

from app.db.repository import StoredRecord


def _fmt(value: object) -> str:
    if value is None or value == "":
        return "unknown"
    return str(value)


def record_to_document(stored: StoredRecord) -> Document:
    """Build a text chunk + metadata for vector search."""
    record = stored.record
    groom = record.groom
    bride = record.bride

    year = record.marriage_year_text or record.marriage_year
    summary = (
        f"{_fmt(groom.full_name)} married {_fmt(bride.full_name)} "
        f"on {_fmt(record.marriage_month)} {_fmt(record.marriage_day)} {_fmt(year)} "
        f"at {_fmt(record.marriage_location)}. "
        f"Groom: age {_fmt(groom.age)}, residence {_fmt(groom.residence)}, "
        f"occupation {_fmt(groom.occupation)}, nativity {_fmt(groom.nativity)}, "
        f"status {_fmt(groom.marital_status)}. "
        f"Bride: age {_fmt(bride.age)}, residence {_fmt(bride.residence)}, "
        f"occupation {_fmt(bride.occupation)}, nativity {_fmt(bride.nativity)}, "
        f"status {_fmt(bride.marital_status)}. "
        f"License {_fmt(record.license_number)}."
    )

    lines = [
        summary,
        f"Record ID: {stored.id}",
        f"Groom: {_fmt(groom.full_name)}",
        f"Groom age: {_fmt(groom.age)}",
        f"Groom residence: {_fmt(groom.residence)}",
        f"Groom occupation: {_fmt(groom.occupation)}",
        f"Groom nativity: {_fmt(groom.nativity)}",
        f"Groom marital status: {_fmt(groom.marital_status)}",
        f"Bride: {_fmt(bride.full_name)}",
        f"Bride age: {_fmt(bride.age)}",
        f"Bride residence: {_fmt(bride.residence)}",
        f"Bride occupation: {_fmt(bride.occupation)}",
        f"Bride nativity: {_fmt(bride.nativity)}",
        f"Bride marital status: {_fmt(bride.marital_status)}",
        f"Marriage date: {_fmt(record.marriage_month)} {_fmt(record.marriage_day)}, {_fmt(year)}",
        f"Marriage location: {_fmt(record.marriage_location)}",
        f"License number: {_fmt(record.license_number)}",
        f"Application number: {_fmt(record.application_number)}",
        f"Officiant: {_fmt(record.officiant.signature)}",
        f"Relationship: {_fmt(record.relationship_between_parties)}",
        f"Source: {_fmt(record.source_pdf)} page {_fmt(record.source_page)}",
        f"OCR engine: {_fmt(record.ocr_engine)}",
    ]

    return Document(
        page_content="\n".join(lines),
        metadata={
            "record_id": stored.id,
            "groom_name": groom.full_name or "",
            "bride_name": bride.full_name or "",
            "marriage_year": str(record.marriage_year or ""),
            "marriage_month": record.marriage_month or "",
            "groom_occupation": groom.occupation or "",
            "bride_occupation": bride.occupation or "",
            "source_pdf": record.source_pdf or "",
            "source_page": record.source_page or 0,
            "ocr_engine": record.ocr_engine or "",
        },
    )
