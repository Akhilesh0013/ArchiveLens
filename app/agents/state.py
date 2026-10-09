from typing import TypedDict

from app.schema import MarriageRecord



class GraphState(TypedDict):

    # --- inputs (set once at start) ---

    ocr_text: str
    source_pdf: str | None
    source_page: int | None
    ocr_engine: str | None

    # --- working data (updated by nodes) ---

    draft: MarriageRecord | None
    validation_errors: list[str]
    judge_verdict: str | None         # "pass" | "fail"
    judge_reason: str | None
    retries: int

    # --- output ---
    final_record: MarriageRecord | None
    status: str                     # "success" | "failed"



