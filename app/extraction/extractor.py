"""LLM-based structured extraction from OCR text."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, settings
from app.schema import MarriageRecord, normalize_marriage_record


def build_llm() -> BaseChatModel:
    """Return a LangChain chat model configured from ``settings``."""

    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            api_key=settings.resolve_api_key(),
        )

    if settings.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=settings.temperature,
            google_api_key=settings.resolve_api_key(),
        )

    raise ValueError(f"Unknown provider: {settings.provider!r}")


class MarriageRecordExtractor:
    """Extract ``MarriageRecord`` objects from noisy OCR text."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self._llm = llm or build_llm()
        self._chain = self._llm.with_structured_output(MarriageRecord)

    def extract(
        self,
        ocr_text: str,
        *,
        source_pdf: str | None = None,
        source_page: int | None = None,
        ocr_engine: str | None = None,
    ) -> MarriageRecord:
        """Run structured extraction on one page of OCR text."""
        record = normalize_marriage_record(
            self._chain.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(
                        content=USER_PROMPT_TEMPLATE.format(ocr_text=ocr_text.strip())
                    ),
                ]
            )
        )
        record.source_pdf = source_pdf
        record.source_page = source_page
        record.ocr_engine = ocr_engine

        return normalize_marriage_record(record)
    
    
