"""SQLite persistence for uploaded and imported marriage records."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import DB_PATH
from app.schema import MarriageRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    ocr_engine TEXT,
    status TEXT NOT NULL,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS marriage_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    upload_id INTEGER NOT NULL UNIQUE REFERENCES uploads(id) ON DELETE CASCADE,
    record_json TEXT NOT NULL,
    groom_name TEXT,
    bride_name TEXT,
    marriage_year INTEGER,
    license_number TEXT,
    source_pdf TEXT,
    source_page INTEGER,
    ocr_engine TEXT,
    groom_occupation TEXT,
    bride_occupation TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_groom ON marriage_records(groom_name);
CREATE INDEX IF NOT EXISTS idx_records_bride ON marriage_records(bride_name);
CREATE INDEX IF NOT EXISTS idx_records_year ON marriage_records(marriage_year);
CREATE INDEX IF NOT EXISTS idx_records_provenance ON marriage_records(source_pdf, source_page);
"""

_MIGRATIONS = (
    "ALTER TABLE marriage_records ADD COLUMN groom_occupation TEXT",
    "ALTER TABLE marriage_records ADD COLUMN bride_occupation TEXT",
    "CREATE INDEX IF NOT EXISTS idx_records_groom_occ ON marriage_records(groom_occupation)",
)


@dataclass
class StoredRecord:
    id: int
    upload_id: int
    filename: str
    record: MarriageRecord
    uploaded_at: str
    status: str


class RecordRepository:
    """CRUD for uploads and extracted marriage records."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            for stmt in _MIGRATIONS:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _search_fields(record: MarriageRecord) -> dict[str, Any]:
        return {
            "groom_name": record.groom.full_name,
            "bride_name": record.bride.full_name,
            "marriage_year": record.marriage_year,
            "license_number": record.license_number,
            "source_pdf": record.source_pdf,
            "source_page": record.source_page,
            "ocr_engine": record.ocr_engine,
            "groom_occupation": record.groom.occupation,
            "bride_occupation": record.bride.occupation,
        }

    def create_upload(
        self,
        filename: str,
        stored_path: str,
        *,
        ocr_engine: str | None = None,
        status: str = "pending",
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO uploads (filename, stored_path, uploaded_at, ocr_engine, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (filename, stored_path, self._now(), ocr_engine, status),
            )
            return int(cur.lastrowid)

    def update_upload_status(
        self,
        upload_id: int,
        status: str,
        *,
        error_message: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE uploads
                SET status = ?, error_message = ?
                WHERE id = ?
                """,
                (status, error_message, upload_id),
            )

    def save_record(self, upload_id: int, record: MarriageRecord) -> int:
        fields = self._search_fields(record)
        payload = record.model_dump(mode="json")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO marriage_records (
                    upload_id, record_json, groom_name, bride_name,
                    marriage_year, license_number, source_pdf, source_page,
                    ocr_engine, groom_occupation, bride_occupation, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(upload_id) DO UPDATE SET
                    record_json = excluded.record_json,
                    groom_name = excluded.groom_name,
                    bride_name = excluded.bride_name,
                    marriage_year = excluded.marriage_year,
                    license_number = excluded.license_number,
                    source_pdf = excluded.source_pdf,
                    source_page = excluded.source_page,
                    ocr_engine = excluded.ocr_engine,
                    groom_occupation = excluded.groom_occupation,
                    bride_occupation = excluded.bride_occupation,
                    created_at = excluded.created_at
                """,
                (
                    upload_id,
                    json.dumps(payload),
                    fields["groom_name"],
                    fields["bride_name"],
                    fields["marriage_year"],
                    fields["license_number"],
                    fields["source_pdf"],
                    fields["source_page"],
                    fields["ocr_engine"],
                    fields["groom_occupation"],
                    fields["bride_occupation"],
                    self._now(),
                ),
            )
            row = conn.execute(
                "SELECT id FROM marriage_records WHERE upload_id = ?",
                (upload_id,),
            ).fetchone()
            record_id = int(row["id"])
            from app.rag.sparse import sync_sparse_row

            sync_sparse_row(conn, record_id, record)
            return record_id

    def list_records(self, limit: int = 100) -> list[StoredRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    mr.id,
                    mr.upload_id,
                    u.filename,
                    mr.record_json,
                    u.uploaded_at,
                    u.status
                FROM marriage_records mr
                JOIN uploads u ON u.id = mr.upload_id
                ORDER BY mr.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [self._row_to_stored(row) for row in rows]

    def iter_all_records(self) -> list[StoredRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    mr.id,
                    mr.upload_id,
                    u.filename,
                    mr.record_json,
                    u.uploaded_at,
                    u.status
                FROM marriage_records mr
                JOIN uploads u ON u.id = mr.upload_id
                ORDER BY mr.id ASC
                """
            ).fetchall()

        return [self._row_to_stored(row) for row in rows]

    def iter_canonical_records(
        self,
        preferred_engine: str | None = None,
    ) -> list[StoredRecord]:
        """One record per (source_pdf, source_page), preferring a single OCR engine."""
        from app.config import settings

        preferred = preferred_engine or settings.rag_preferred_ocr_engine
        canonical: dict[tuple[str, int], StoredRecord] = {}

        for item in self.iter_all_records():
            key = (item.record.source_pdf or "", item.record.source_page or 0)
            existing = canonical.get(key)
            if existing is None:
                canonical[key] = item
                continue
            if item.record.ocr_engine == preferred:
                canonical[key] = item

        return sorted(canonical.values(), key=lambda r: r.id)

    def backfill_search_columns(self) -> int:
        """Populate denormalized search columns from record_json for older rows."""
        updated = 0
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, record_json, groom_occupation, bride_occupation
                FROM marriage_records
                WHERE groom_occupation IS NULL OR bride_occupation IS NULL
                """
            ).fetchall()
            for row in rows:
                record = MarriageRecord.model_validate_json(row["record_json"])
                fields = self._search_fields(record)
                conn.execute(
                    """
                    UPDATE marriage_records
                    SET groom_occupation = ?, bride_occupation = ?
                    WHERE id = ?
                    """,
                    (
                        fields["groom_occupation"],
                        fields["bride_occupation"],
                        row["id"],
                    ),
                )
                updated += 1
        return updated

    def search_records(
        self,
        *,
        marriage_year: int | None = None,
        groom_substr: str | None = None,
        bride_substr: str | None = None,
        name_substr: str | None = None,
        occupation_substr: str | None = None,
        limit: int = 10,
    ) -> list[StoredRecord]:
        clauses: list[str] = []
        params: list[Any] = []

        if marriage_year is not None:
            clauses.append("mr.marriage_year = ?")
            params.append(marriage_year)
        if groom_substr:
            clauses.append("UPPER(mr.groom_name) LIKE ?")
            params.append(f"%{groom_substr.upper()}%")
        if bride_substr:
            clauses.append("UPPER(mr.bride_name) LIKE ?")
            params.append(f"%{bride_substr.upper()}%")
        if name_substr:
            token = name_substr.upper()
            clauses.append(
                "(UPPER(mr.groom_name) LIKE ? OR UPPER(mr.bride_name) LIKE ?)"
            )
            params.extend([f"%{token}%", f"%{token}%"])
        if occupation_substr:
            token = occupation_substr.upper()
            clauses.append(
                """(
                    UPPER(IFNULL(mr.groom_occupation, '')) LIKE ?
                    OR UPPER(IFNULL(mr.bride_occupation, '')) LIKE ?
                    OR UPPER(mr.record_json) LIKE ?
                )"""
            )
            params.extend([f"%{token}%", f"%{token}%", f"%{token}%"])

        if not clauses:
            return []

        where = f"WHERE {' AND '.join(clauses)}"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    mr.id,
                    mr.upload_id,
                    u.filename,
                    mr.record_json,
                    u.uploaded_at,
                    u.status
                FROM marriage_records mr
                JOIN uploads u ON u.id = mr.upload_id
                {where}
                ORDER BY mr.id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()

        return [self._row_to_stored(row) for row in rows]

    @staticmethod
    def _row_to_stored(row: sqlite3.Row) -> StoredRecord:
        return StoredRecord(
            id=row["id"],
            upload_id=row["upload_id"],
            filename=row["filename"],
            record=MarriageRecord.model_validate_json(row["record_json"]),
            uploaded_at=row["uploaded_at"],
            status=row["status"],
        )

    def get_record(self, record_id: int) -> StoredRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    mr.id,
                    mr.upload_id,
                    u.filename,
                    mr.record_json,
                    u.uploaded_at,
                    u.status
                FROM marriage_records mr
                JOIN uploads u ON u.id = mr.upload_id
                WHERE mr.id = ?
                """,
                (record_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_stored(row)

    def count_records(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM marriage_records").fetchone()
            return int(row["n"])

    def find_by_provenance(
        self,
        source_pdf: str,
        source_page: int,
        ocr_engine: str | None,
    ) -> StoredRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    mr.id,
                    mr.upload_id,
                    u.filename,
                    mr.record_json,
                    u.uploaded_at,
                    u.status
                FROM marriage_records mr
                JOIN uploads u ON u.id = mr.upload_id
                WHERE mr.source_pdf = ?
                  AND mr.source_page = ?
                  AND IFNULL(mr.ocr_engine, '') = IFNULL(?, '')
                """,
                (source_pdf, source_page, ocr_engine),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_stored(row)
