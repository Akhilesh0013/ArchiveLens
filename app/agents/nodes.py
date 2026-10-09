from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.validators import validate_record
from app.config import JUDGE_IGNORE_FIELDS, JUDGE_PROMPT
from app.extraction.extractor import MarriageRecordExtractor, build_llm
from app.schema import MarriageRecord, marriage_record_to_payload, normalize_marriage_record


def _draft_json_for_judge(draft: MarriageRecord) -> str:
    payload = marriage_record_to_payload(draft)
    for key in JUDGE_IGNORE_FIELDS:
        payload.pop(key, None)
    return json.dumps(payload, indent=2)


def extract_node(state: dict) -> dict:
    extractor = MarriageRecordExtractor()

    draft = extractor.extract(
        state["ocr_text"],
        source_pdf=state.get("source_pdf"),
        source_page=state.get("source_page"),
        ocr_engine=state.get("ocr_engine"),
    )

    return {"draft": draft, "retries": state.get("retries", 0)}


def validate_node(state: dict) -> dict:
    draft = state["draft"]
    errors = validate_record(draft)
    return {"draft": draft, "validation_errors": errors}


def judge_node(state: dict) -> dict:
    llm = build_llm()
    draft = state["draft"]
    response = llm.invoke([
        SystemMessage(
            content="You are a lenient extraction judge focused on core party facts, not form headers."
        ),
        HumanMessage(content=JUDGE_PROMPT.format(
            ocr_text=state["ocr_text"],
            draft_json=_draft_json_for_judge(draft),
        )),
    ])
    text = response.content.strip()
    verdict = "fail"
    reason = text
    for line in text.splitlines():
        if line.upper().startswith("VERDICT:"):
            verdict = "pass" if "pass" in line.lower() else "fail"
        if line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return {"judge_verdict": verdict, "judge_reason": reason}


MAX_RETRIES = 2


def finalize_node(state: dict) -> dict:
    return {
        "final_record": normalize_marriage_record(state["draft"]),
        "status": "success",
    }


def retry_node(state: dict) -> dict:
    return {"retries": state.get("retries", 0) + 1}
