"""Central configuration for the document-understanding pipeline."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "baltimore-marriage-certificates"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
AGENT_DIR = ARTIFACTS_DIR / "agent"
DOCLING_DIR = ARTIFACTS_DIR / "docling"
VLM_DIR = ARTIFACTS_DIR / "vlm"
EXTRACTED_DIR = ARTIFACTS_DIR / "extracted"
EVAL_DIR = ARTIFACTS_DIR / "eval"
GOLD_DIR = PROJECT_ROOT / "gold"

UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "records.db"
VLM_DB_PATH = DATA_DIR / "records_vlm.db"
QWEN25_DB_PATH = DATA_DIR / "records_qwen25_zeroshot.db"
RAG_INDEX_DIR = DATA_DIR / "rag_index"
VLM_RAG_INDEX_DIR = DATA_DIR / "rag_index_vlm"
QWEN25_RAG_INDEX_DIR = DATA_DIR / "rag_index_qwen25"

RAW_PDF_GLOB = "*.pdf"
PROCESSED_PDF_GLOB = "*_content.pdf"
DOCLING_JSON_GLOB = "*_content.json"

OCR_ENGINE_TAG = "docling-rapidocr"
VLM_ENGINE_TAG = "qwen2-vl-2b"
QWEN25_ENGINE_TAG = "qwen2.5-vl-7b"
EXTRACTED_FILENAME = "marriage_records_docling.jsonl"
CHECKPOINT_PATH = EXTRACTED_DIR / EXTRACTED_FILENAME

GOLD_OCR_SOURCES = ("docling", "vlm", "qwen25")
RAG_CORPORA = ("docling", "vlm", "qwen25")

# Engine tag written by the zero-shot Qwen2.5-VL scan (see app/extraction/vlm_zeroshot.py).
QWEN25_ZEROSHOT_ENGINE = f"zeroshot+{QWEN25_ENGINE_TAG}"

VLM_CORPUS_PAGES_JSONL = VLM_DIR / "corpus_pages.jsonl"
VLM_GOLD_PAGES_JSONL = VLM_DIR / "pages.jsonl"

# Zero-shot Qwen2.5-VL corpus scan (image → structured record, no agent)
QWEN25_RECORDS_JSONL = VLM_DIR / "qwen25_zeroshot_records.jsonl"
QWEN25_ERRORS_JSONL = VLM_DIR / "qwen25_zeroshot_errors.jsonl"


def normalize_corpus(corpus: str) -> str:
    name = corpus.strip().lower()
    if name not in RAG_CORPORA:
        supported = ", ".join(RAG_CORPORA)
        raise ValueError(f"Unknown corpus {corpus!r}. Supported: {supported}")
    return name


def corpus_db_path(corpus: str) -> Path:
    mapping = {
        "docling": DB_PATH,
        "vlm": VLM_DB_PATH,
        "qwen25": QWEN25_DB_PATH,
    }
    return mapping[normalize_corpus(corpus)]


def corpus_rag_index_dir(corpus: str) -> Path:
    mapping = {
        "docling": RAG_INDEX_DIR,
        "vlm": VLM_RAG_INDEX_DIR,
        "qwen25": QWEN25_RAG_INDEX_DIR,
    }
    return mapping[normalize_corpus(corpus)]


def corpus_collection_name(corpus: str) -> str:
    return f"marriage_records_{normalize_corpus(corpus)}"


def corpus_preferred_engine(corpus: str) -> str:
    name = normalize_corpus(corpus)
    if name == "vlm":
        return f"agent+{VLM_ENGINE_TAG}"
    if name == "qwen25":
        return QWEN25_ZEROSHOT_ENGINE
    return OCR_ENGINE_TAG


def gold_results_dir(ocr_source: str) -> Path:
    """Root folder for one backend's gold eval artifacts."""
    backend = ocr_source.strip().lower()
    mapping = {
        "docling": AGENT_DIR / "docling-gold-results",
        "vlm": AGENT_DIR / "vlm-gold-results",
        "qwen25": AGENT_DIR / "qwen-2.5-vl-7b",
    }
    if backend not in mapping:
        supported = ", ".join(GOLD_OCR_SOURCES)
        raise ValueError(f"Unknown OCR source {ocr_source!r}. Supported: {supported}")
    path = mapping[backend]
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(parents=True, exist_ok=True)
    return path


def gold_predictions_path(ocr_source: str) -> Path:
    return gold_results_dir(ocr_source) / "marriage_records_agent.jsonl"


def gold_plots_root(ocr_source: str) -> Path:
    return gold_results_dir(ocr_source) / "plots"

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a genealogical records analyst. You extract structured data from
OCR transcriptions of 1930s-1940s Baltimore City (Maryland) marriage licenses.

The OCR text is NOISY: characters are misread, field labels bleed into values,
and lines appear out of order. Use the form's known layout to decide WHICH field
each fragment belongs to — but copy the text exactly as it appears in the OCR.

Form layout (for field assignment only):
- Header: "GOOD ONLY IN BALTIMORE CITY", "STATE OF MARYLAND", "BALTIMORE CITY",
  license number ("No. ..."), application number ("APPLICATION # ..." / "APPLI ...").
- Certification: "I Hereby Certify, That on this <day> day of <month>,
  Nineteen hundred and <year words>", married at <location>.
- Groom block, then bride block. Each has: name, residence, age, color
  (WHT/COL), nativity (birth state/country), occupation, marital status
  (single/widower/widow/divorced).
- Consent lines if either party is a minor; relationship line
  (e.g. "NOT RELATED", "NONE"); officiant signature, official character, address.

Rules:
1. Extract values AS THEY APPEAR in the OCR text. Do NOT correct spelling,
   expand abbreviations, guess missing digits, or infer names/years/dates.
   Example: if OCR says "ALLIME POYAS", output "ALLIME POYAS" — not "ALLINE".
2. If OCR merges label and value (e.g. "COLNativity TEN", "OccupationONE"),
   split only when the boundary is unambiguous. Otherwise put the verbatim
   fragment in the best-matching field and leave related fields null.
3. Do NOT invent values. If a field is blank, illegible, or missing in the OCR,
   use null.
4. Preserve capitalization and punctuation exactly as in the OCR source.
5. Occupation "NONE" on the form is a real value — keep it as "NONE".
6. For dates: copy day/month/year fragments verbatim. Put the full year phrase
   in marriage_year_text. Set marriage_year only if all four digits appear
   explicitly in the OCR (do not infer "1939" from "Nineteen hundred and thirty").
"""

USER_PROMPT_TEMPLATE = """\
Extract the marriage record from this OCR transcription.
Copy values verbatim — do not correct OCR errors.

--- OCR TEXT START ---
{ocr_text}
--- OCR TEXT END ---
"""

# ---------------------------------------------------------------------------
# Zero-shot VLM extraction prompt (notebooks/vlm.ipynb — Model 2)
# ---------------------------------------------------------------------------

ZEROSHOT_VLM_PROMPT = """\
Extract every field from this marriage certificate.
Return ONLY valid JSON (no markdown fences, no commentary) in exactly this format.
Use null for any field that is blank, illegible, or missing:

{
  "jurisdiction_note": null,
  "state": null,
  "city": null,
  "license_number": null,
  "application_number": null,
  "marriage_day": null,
  "marriage_month": null,
  "marriage_year": null,
  "marriage_year_text": null,
  "marriage_location": null,
  "groom": {
    "full_name": null,
    "residence": null,
    "age": null,
    "color": null,
    "nativity": null,
    "occupation": null,
    "marital_status": null
  },
  "bride": {
    "full_name": null,
    "residence": null,
    "age": null,
    "color": null,
    "nativity": null,
    "occupation": null,
    "marital_status": null
  },
  "groom_consent_by": null,
  "bride_consent_by": null,
  "parents_consent_filed": null,
  "relationship_between_parties": null,
  "officiant": {
    "signature": null,
    "official_character": null,
    "address": null
  }
}
"""


# ---------------------------------------------------------------------------
# Judge prompts
# ---------------------------------------------------------------------------

# Form boilerplate / provenance — not scored by the judge (often inferred or absent in OCR).
JUDGE_IGNORE_FIELDS: frozenset[str] = frozenset(
    {
        "jurisdiction_note",
        "state",
        "city",
        "source_pdf",
        "source_page",
        "ocr_engine",
    }
)

JUDGE_PROMPT = """\
You are an evaluation judge for genealogical record extraction from noisy OCR.

Given the OCR source text and the extracted JSON draft, decide pass or fail.

IGNORE these fields entirely (do not use them in your verdict):
jurisdiction_note, state, city, source_pdf, source_page, ocr_engine.
These are standard form headers or metadata and may be inferred when absent from OCR.

PASS if the important party and marriage facts in the draft are reasonably supported \
by the OCR text. Be lenient about:
- capitalization and spacing (e.g. "Single" vs "SINGLE", "Baltimore MD" vs "BALTIMORE MD.")
- minor punctuation or abbreviation differences
- license/application numbers when the digit sequence appears in the OCR
- null fields (missing data is fine)

FAIL only when a non-null value is clearly invented or contradicts the OCR — especially \
groom/bride names, ages, residences, nativity, occupation, marriage date parts, or officiant \
signature when those values cannot be found or reasonably matched in the OCR text.

OCR TEXT:
{ocr_text}

DRAFT JSON (header fields omitted):
{draft_json}

Respond with exactly two lines:
VERDICT: pass or fail
REASON: one sentence explaining why
"""


# ---------------------------------------------------------------------------
# RAG prompt
# ---------------------------------------------------------------------------

RAG_QA_PROMPT = """\
You are a genealogical research assistant for Baltimore City marriage records (1930s–1940s).

Answer the user's question using ONLY the marriage records provided below.
If the records do not contain enough information, say so clearly.
Always cite sources as [record_id] matching the Record ID in the context.
Do not invent names, dates, or facts not present in the records.

MARRIAGE RECORDS:
{context}

QUESTION: {question}
"""










# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

EVAL_FIELDS: list[str] = [
    "license_number",
    "application_number",
    "marriage_day",
    "marriage_month",
    "marriage_year",
    "marriage_location",
    "groom.full_name",
    "groom.residence",
    "groom.age",
    "groom.color",
    "groom.nativity",
    "groom.occupation",
    "groom.marital_status",
    "bride.full_name",
    "bride.residence",
    "bride.age",
    "bride.color",
    "bride.nativity",
    "bride.occupation",
    "bride.marital_status",
    "relationship_between_parties",
    "officiant.signature",
]

FUZZY_FIELDS: frozenset[str] = frozenset(
    {
        "officiant.signature",
        "groom.full_name",
        "bride.full_name",
        "groom.residence",
        "bride.residence",
        "groom.nativity",
        "bride.nativity",
        "groom.occupation",
        "bride.occupation",
    }
)

FUZZY_THRESHOLD = 0.85

# Fields where OCR often prepends labels — compare extracted digit sequences.
NUMERIC_ID_FIELDS: frozenset[str] = frozenset(
    {
        "license_number",
        "application_number",
    }
)


# ---------------------------------------------------------------------------
# Runtime settings (overridable via .env)
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys
    openai_api_key: str = ""
    google_api_key: str = ""
    gemini_api_key: str = ""

    # LLM
    provider: str = "openai"  # "openai" or "gemini"
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-2.5-flash"
    temperature: float = 0.0
    max_retries: int = 3

    # OCR backend: "docling" (Docling+RapidOCR) or "vlm" (local Qwen2-VL via mlx-vlm)
    ocr_backend: str = "docling"
    docling_force_full_page_ocr: bool = True
    vlm_model_path: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    vlm_max_tokens: int = 2048

    # Zero-shot structured extraction model (image → JSON, no agent)
    zeroshot_vlm_model_path: str = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    zeroshot_vlm_max_tokens: int = 2048

    # RAG
    rag_top_k: int = 5
    rag_preferred_ocr_engine: str = "docling-rapidocr"
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_embedding_model: str = "models/text-embedding-004"

    def resolve_api_key(self) -> str:
        if self.provider == "openai":
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY not set. Add it to .env and restart."
                )
            return self.openai_api_key
        if self.provider == "gemini":
            key = self.google_api_key or self.gemini_api_key
            if not key:
                raise ValueError(
                    "GOOGLE_API_KEY not set. Add it to .env and restart."
                )
            return key
        raise ValueError(f"Unknown provider: {self.provider!r}")


settings = Settings()
