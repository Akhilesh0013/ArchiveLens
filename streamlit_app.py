"""Streamlit UI — genealogy RAG over Baltimore marriage certificates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from app.config import PROJECT_ROOT, VLM_CORPUS_PAGES_JSONL, corpus_db_path, settings
from app.db.repository import RecordRepository
from app.ingestion.service import IngestionService
from app.ocr.vlm import VLM_PAGES_JSONL
from app.pipeline.ocr_loader import load_vlm_gold_pages, load_vlm_pages
from app.rag.service import RAGService

IMAGES_DIR = PROJECT_ROOT / "images"

# Corpora selectable in the UI. Only OCR-backed corpora support Upload / Gold eval;
# "qwen25" is a read-only zero-shot corpus (search/browse only).
UI_CORPORA = ("docling", "vlm", "qwen25")
OCR_BACKED_CORPORA = ("docling", "vlm")

USE_CASES: dict[str, dict[str, Any]] = {
    "🎵 Musicians": {
        "query": "Which marriage records list musician as an occupation?",
        "mode": "Ask (RAG Q&A)",
        "top_k": 5,
        "description": "Hybrid search over occupation fields",
    },
    "📅 1939 marriages": {
        "query": "Show marriages from 1939 in Baltimore",
        "mode": "Search (retrieve only)",
        "top_k": 8,
        "description": "Structured year filter + semantic search",
    },
    "👰 Bride named Mary": {
        "query": "Find records where the bride's name contains Mary",
        "mode": "Search (retrieve only)",
        "top_k": 6,
        "description": "Name substring retrieval",
    },
    "📜 License lookup": {
        "query": "What do we know about license number 34178?",
        "mode": "Ask (RAG Q&A)",
        "top_k": 3,
        "description": "Point lookup with cited answer",
    },
    "🏠 Balto residents": {
        "query": "List couples where groom or bride residence is BALTO MD",
        "mode": "Ask (RAG Q&A)",
        "top_k": 5,
        "description": "Residence field search",
    },
    "⚖️ Divorced grooms": {
        "query": "Which grooms have marital status divorced?",
        "mode": "Ask (RAG Q&A)",
        "top_k": 5,
        "description": "Marital status filter via RAG",
    },
    "🔍 Officiant Harris": {
        "query": "Find marriages officiated by Harris",
        "mode": "Search (retrieve only)",
        "top_k": 5,
        "description": "Officiant signature search",
    },
    "📊 Corpus overview": {
        "query": "How many marriage records are in the database and what years do they cover?",
        "mode": "Ask (RAG Q&A)",
        "top_k": 10,
        "description": "Aggregate question over indexed corpus",
    },
}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _badge(label: str, color: str = "brown") -> str:
    colors = {
        "brown": ("#f5ebe0", "#5c4033"),
        "green": ("#d1fae5", "#065f46"),
        "blue": ("#dbeafe", "#1e40af"),
        "yellow": ("#fef9c3", "#854d0e"),
        "purple": ("#f3e8ff", "#6b21a8"),
    }
    bg, fg = colors.get(color, colors["brown"])
    return (
        f'<span style="background:{bg};color:{fg};padding:4px 10px;'
        f'border-radius:999px;font-size:0.85rem;font-weight:600;">{label}</span>'
    )


def _show_svg(path: Path, *, caption: str | None = None) -> None:
    """Render local SVG via markdown — Streamlit st.image mishandles SVG files."""
    svg = path.read_text(encoding="utf-8")
    st.markdown(
        f'<div style="width:100%;">{svg}</div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.caption(caption)


@st.cache_resource
def get_service() -> IngestionService:
    return IngestionService()


@st.cache_resource
def get_repo(corpus: str) -> RecordRepository:
    return RecordRepository(corpus_db_path(corpus))


def _ocr_backend_label(backend: str) -> str:
    if backend == "vlm":
        return "Qwen2-VL (mlx · Apple Silicon)"
    return "Docling + RapidOCR"


def _corpus_display_label(corpus: str) -> str:
    labels = {
        "docling": "Docling + RapidOCR",
        "vlm": "Qwen2-VL (agent · mlx)",
        "qwen25": "Qwen2.5-VL 7B (zero-shot · read-only)",
    }
    return labels.get(corpus, corpus)


def _selected_ocr_backend() -> str:
    return st.session_state.get("ocr_backend", settings.ocr_backend)


def _active_corpus() -> str:
    return _selected_ocr_backend()


def _is_ocr_backed(corpus: str) -> bool:
    return corpus in OCR_BACKED_CORPORA


def _corpus_label(corpus: str) -> str:
    labels = {
        "docling": "Docling corpus (records.db)",
        "vlm": "VLM corpus (records_vlm.db)",
        "qwen25": "Qwen2.5-VL zero-shot corpus (records_qwen25_zeroshot.db)",
    }
    return labels.get(corpus, corpus)


def _sidebar_ocr_settings() -> None:
    st.markdown("**OCR backend / search corpus**")
    default = settings.ocr_backend if settings.ocr_backend in UI_CORPORA else "docling"
    if "ocr_backend" not in st.session_state or st.session_state.ocr_backend not in UI_CORPORA:
        st.session_state.ocr_backend = default

    st.session_state.ocr_backend = st.selectbox(
        "Active corpus",
        options=list(UI_CORPORA),
        index=UI_CORPORA.index(st.session_state.ocr_backend),
        format_func=_corpus_display_label,
        help=(
            "Docling: notebook-imported records.db + rag_index. "
            "VLM: agent-extracted records_vlm.db + rag_index_vlm. "
            "Qwen2.5-VL: zero-shot records_qwen25_zeroshot.db + rag_index_qwen25 "
            "(search/browse only — no upload or gold eval)."
        ),
    )
    st.caption(_corpus_label(_active_corpus()))

    if not _is_ocr_backed(st.session_state.ocr_backend):
        st.info(
            "Read-only zero-shot corpus. Upload and Gold eval are disabled; "
            "use Search and Browse.",
            icon="🔍",
        )

    if st.session_state.ocr_backend == "vlm":
        gold_cached = len(load_vlm_gold_pages())
        corpus_cached = len(load_vlm_pages(VLM_CORPUS_PAGES_JSONL))
        st.caption(
            f"Gold OCR: {gold_cached}/14 · `{VLM_PAGES_JSONL.name}` · "
            f"Corpus: {corpus_cached} · `{VLM_CORPUS_PAGES_JSONL.name}`"
        )
        if gold_cached == 0:
            st.warning(
                "Gold eval needs `pages.jsonl`. Run **Build gold VLM cache** "
                "(instant if corpus cache already has the 14 gold pages).",
                icon="⚠️",
            )
            if st.button("Build gold VLM cache", width="stretch", key="build_vlm_gold_cache"):
                from app.ocr.vlm import build_gold_vlm_cache

                with st.spinner("Running Qwen2-VL on 14 gold pages…"):
                    try:
                        count = build_gold_vlm_cache()
                        if count == 0:
                            st.error(
                                "No pages written. Add `gold/images/*.png` or PDFs under "
                                "`data/baltimore-marriage-certificates/`."
                            )
                        else:
                            st.success(f"Cached {count} gold page(s) → `{VLM_PAGES_JSONL}`")
                            st.rerun()
                    except Exception as exc:
                        st.error(str(exc))

        if corpus_cached == 0:
            st.warning(
                "Search corpus is empty until you build the processed-PDF VLM cache.",
                icon="⚠️",
            )
        if st.button("Build corpus VLM cache", width="stretch", key="build_vlm_corpus_cache"):
            service = get_service()
            with st.spinner("Running Qwen2-VL on processed PDFs…"):
                try:
                    count = service.build_vlm_corpus_cache()
                    st.success(f"Cached {count} new page(s) → `{VLM_CORPUS_PAGES_JSONL}`")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        if corpus_cached > 0 and st.button(
            "Ingest VLM corpus to DB",
            width="stretch",
            key="ingest_vlm_corpus",
        ):
            service = get_service()
            with st.spinner("Running agent on cached VLM OCR…"):
                try:
                    result = service.ingest_vlm_corpus(skip_existing=True)
                    st.success(
                        f"Ingested {result.processed} page(s), skipped {result.skipped}"
                    )
                    if result.errors:
                        st.warning(f"{len(result.errors)} page(s) failed — see logs.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


@st.cache_resource
def get_rag(corpus: str) -> RAGService:
    return RAGService(corpus=corpus)


def _sidebar_brand() -> None:
    brand = IMAGES_DIR / "sidebar_brand.svg"
    if brand.exists():
        _show_svg(brand)
    else:
        st.markdown("### 📜 Genealogy RAG")
        st.caption("Baltimore Marriage Records")


def _sidebar(service: IngestionService, rag: RAGService) -> None:
    corpus = _active_corpus()
    repo = get_repo(corpus)
    with st.sidebar:
        _sidebar_brand()
        st.markdown("---")

        illustration = IMAGES_DIR / "generic_form.svg"
        if illustration.exists():
            _show_svg(
                illustration,
                caption="Vital record form (illustration only)",
            )

        st.markdown("---")
        _sidebar_ocr_settings()
        st.markdown("---")
        st.markdown(f"**Corpus status · {corpus}**")
        total = repo.count_records()
        canonical = len(repo.iter_canonical_records())
        indexed = rag.indexed_count()
        st.metric("Records in DB", total)
        st.metric("Canonical pages", canonical)
        st.metric("Indexed vectors", indexed)

        if st.button("🔄 Rebuild search index", width="stretch"):
            with st.spinner(f"Embedding {corpus} records…"):
                try:
                    count = rag.rebuild_index()
                    st.success(f"Indexed {count} record(s) in {corpus} corpus")
                except Exception as exc:
                    st.error(str(exc))

        st.markdown("---")
        pipeline = IMAGES_DIR / "pipeline.svg"
        if pipeline.exists():
            _show_svg(pipeline, caption="Pipeline overview")

        st.caption(
            f"Provider: `{settings.provider}` · "
            f"Embeddings: `{settings.openai_embedding_model if settings.provider == 'openai' else settings.gemini_embedding_model}`"
        )
        st.markdown("---")
        st.caption("Recent queries")
        history = st.session_state.get("query_history", [])
        if not history:
            st.caption("No queries yet this session.")
        else:
            for item in reversed(history[-5:]):
                st.markdown(f"· {item['query'][:60]}…")


def _render_answer_card(result: Any, *, retrieve_only: bool = False) -> None:
    with st.container(border=True):
        cols = st.columns([3, 1, 1])
        with cols[0]:
            st.markdown(
                _badge("RAG Answer" if not retrieve_only else "Retrieved", "green"),
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.metric("Sources", len(result.sources if hasattr(result, "sources") else result))
        with cols[2]:
            st.metric("Mode", "Q&A" if not retrieve_only else "Search")

        st.divider()
        if retrieve_only:
            st.markdown("**Matching records**")
            if not result:
                st.info("No matches.")
                return
            st.dataframe(result, width="stretch", hide_index=True)
            return

        st.markdown("**Answer**")
        st.markdown(result.answer if result.answer else "_No answer returned._")
        if result.sources:
            with st.expander(f"📚 Sources ({len(result.sources)})", expanded=True):
                for src in result.sources:
                    st.markdown(
                        f"**[{src['record_id']}]** {src.get('groom') or '?'} / "
                        f"{src.get('bride') or '?'} ({src.get('year') or '?'}) — "
                        f"`{src.get('source_pdf') or '?'} p{src.get('source_page') or '?'}`"
                    )


def _search_section(rag: RAGService) -> None:
    st.header("💬 Search marriage records")

    last = st.session_state.get("last_search_result")
    if last:
        with st.container(border=True):
            st.markdown("**📌 Last result**")
            st.caption(f"Query: `{last.get('query', '—')}`")
            if last.get("retrieve_only"):
                _render_answer_card(last.get("hits", []), retrieve_only=True)
            else:
                _render_answer_card(last["result"], retrieve_only=False)
            if st.button("🗑️ Dismiss result", key="dismiss_search"):
                st.session_state.pop("last_search_result", None)
                st.rerun()
        st.divider()

    st.markdown("**🎯 Example queries** — click to auto-fill:")
    preset_items = list(USE_CASES.items())
    for row_start in range(0, len(preset_items), 4):
        row = preset_items[row_start : row_start + 4]
        cols = st.columns(len(row))
        for idx, (label, cfg) in enumerate(row):
            with cols[idx]:
                if st.button(
                    label,
                    width="stretch",
                    key=f"preset_{row_start + idx}",
                    help=cfg.get("description", ""),
                ):
                    st.session_state["q_query"] = cfg["query"]
                    st.session_state["q_mode"] = cfg["mode"]
                    st.session_state["q_top_k"] = cfg["top_k"]
                    st.rerun()

    st.divider()

    if "q_query" not in st.session_state:
        st.session_state["q_query"] = "Which marriage records list musician as an occupation?"
    if "q_top_k" not in st.session_state:
        st.session_state["q_top_k"] = 5

    query = st.text_area(
        "Your question",
        height=90,
        key="q_query",
        placeholder="e.g. musicians married in 1939 · bride named Mary · license 34178",
    )

    with st.expander("⚙️ Search settings", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            mode = st.radio(
                "Mode",
                ["Ask (RAG Q&A)", "Search (retrieve only)"],
                horizontal=True,
                key="q_mode",
            )
        with c2:
            top_k = st.slider(
                "top_k", min_value=1, max_value=15, value=5, key="q_top_k"
            )

        st.caption(
            f"Active: **{mode.split('(')[0].strip()}** · top_k: **{top_k}** · "
            f"Searching **{_corpus_label(_active_corpus())}** — no OCR/agent re-run."
        )

    if st.button("🚀 Run search", width="stretch", type="primary"):
        if not query.strip():
            st.warning("Enter a question first.")
            return
        with st.spinner("Searching…"):
            try:
                retrieve_only = mode.startswith("Search")
                if retrieve_only:
                    hits = rag.search(query, k=top_k)
                    _render_answer_card(hits, retrieve_only=True)
                    st.session_state["last_search_result"] = {
                        "query": query,
                        "retrieve_only": True,
                        "hits": hits,
                    }
                else:
                    result = rag.ask(query, k=top_k)
                    _render_answer_card(result, retrieve_only=False)
                    st.session_state["last_search_result"] = {
                        "query": query,
                        "retrieve_only": False,
                        "result": result,
                    }

                history = st.session_state.get("query_history", [])
                history.append({"query": query, "mode": mode})
                st.session_state["query_history"] = history[-20:]
            except Exception as exc:
                st.error(f"Search failed: {exc}")


def _upload_section(service: IngestionService) -> None:
    backend = _selected_ocr_backend()
    st.header("📤 Upload certificate")
    st.info(
        f"Pipeline: {_ocr_backend_label(backend)} → extract → validate → DB. "
        "Requires `OPENAI_API_KEY` or `GOOGLE_API_KEY` in `.env`.",
        icon="⏳",
    )

    uploaded = st.file_uploader(
        "Choose a PDF or image",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "webp"],
        key="cert_uploader",
    )

    if uploaded is None:
        illustration = IMAGES_DIR / "generic_form.svg"
        if illustration.exists():
            st.caption("Expected form layout (illustration - no real record data):")
            _show_svg(illustration)
        return

    size_kb = len(uploaded.getvalue()) / 1024
    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.markdown(f"**📄 {uploaded.name}**")
        with c2:
            st.markdown(f"`{size_kb:.1f} KB`")
        with c3:
            st.markdown(f"`{uploaded.type or 'file'}`")

    if st.button("🚀 Process and save to database", width="stretch", type="primary"):
        with st.status("Running OCR and agent pipeline…", expanded=True) as status:
            try:
                st.write(f"📡 Step 1/3 — {_ocr_backend_label(backend)}…")
                result = service.ingest_file(
                    uploaded.getvalue(),
                    uploaded.name,
                    ocr_backend=backend,
                )
                st.write(f"✅ Step 2/3 — Agent extracted {len(result.pages)} page(s)")
                st.write("✅ Step 3/3 — Saved to SQLite")
                status.update(
                    label=f"Saved {len(result.pages)} page(s) from upload #{result.upload_id}",
                    state="complete",
                )
                for page in result.pages:
                    with st.expander(
                        f"Page {page.page_no}: "
                        f"{page.record.groom.full_name or '?'} / "
                        f"{page.record.bride.full_name or '?'} "
                    ):
                        if page.validation_errors:
                            st.warning(", ".join(page.validation_errors))
                        st.json(page.record.model_dump(mode="json"))
            except Exception as exc:
                status.update(label="Processing failed", state="error")
                st.error(str(exc))


def _gold_section(service: IngestionService) -> None:
    backend = _selected_ocr_backend()
    st.header("📊 Gold evaluation")
    st.write(
        f"Runs the LangGraph agent on 14 hand-labeled pages using cached "
        f"**{_ocr_backend_label(backend)}** text, then writes eval CSVs and plots."
    )
    if backend == "vlm" and len(load_vlm_gold_pages()) < 14:
        st.warning(
            "Build the gold VLM cache (`pages.jsonl`, 14 pages) before running gold eval.",
            icon="⚠️",
        )

    if st.button("Run gold eval", width="stretch", type="primary"):
        with st.spinner("Running agent on gold set…"):
            try:
                result = service.run_gold_eval(persist=False, ocr_source=backend)
                metrics = result.metrics
                st.success(f"Evaluated {metrics['n_pages']} gold pages")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Field accuracy", f"{metrics['field_accuracy']:.1%}")
                c2.metric("Correct", int(metrics["correct"]))
                c3.metric("Wrong", int(metrics["wrong"]))
                c4.metric("Missed", int(metrics["missed"]))
                st.caption(f"Predictions: `{result.output_path}`")
                if result.run_dir:
                    st.caption(f"Reports & plots: `{result.run_dir}/`")
            except Exception as exc:
                st.error(str(exc))


def _browse_section(service: IngestionService, total: int) -> None:
    corpus = _active_corpus()
    repo = get_repo(corpus)
    total = repo.count_records()
    st.header(f"🗂️ Browse database ({total}) · {corpus}")
    records = repo.list_records(limit=200)

    if not records:
        st.warning("No records yet. Upload a certificate or import your notebook batch.")
        return

    rows = []
    for item in records:
        rec = item.record
        rows.append(
            {
                "id": item.id,
                "groom": rec.groom.full_name,
                "bride": rec.bride.full_name,
                "year": rec.marriage_year,
                "month": rec.marriage_month,
                "location": rec.marriage_location,
                "license": rec.license_number,
                "source": rec.source_pdf,
                "page": rec.source_page,
                "engine": rec.ocr_engine,
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True, height=420)

    record_id = st.number_input(
        "View record JSON by id",
        min_value=1,
        max_value=max(r.id for r in records),
        value=records[0].id,
    )
    item = repo.get_record(int(record_id))
    if item:
        with st.container(border=True):
            st.json(item.record.model_dump(mode="json"))


def _import_section(service: IngestionService) -> None:
    st.header("📥 Import notebook batch")
    st.write(
        "Loads `artifacts/extracted/marriage_records_*.jsonl` into SQLite — "
        "no LLM or OCR re-run."
    )

    if st.button("Import all batch JSONL files", width="stretch"):
        with st.spinner("Importing…"):
            counts = service.import_default_batch(skip_existing=True)
        if counts:
            st.success("Import complete.")
            for name, count in counts.items():
                st.write(f"- **{name}**: {count} new record(s)")
            st.write(f"**Total in database:** {service.repository.count_records()}")
        else:
            st.warning("No JSONL files found in artifacts/extracted/")


def main() -> None:
    st.set_page_config(
        page_title="Genealogy RAG — Baltimore Marriage Records",
        page_icon="📜",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    service = get_service()
    corpus = _active_corpus()
    rag = get_rag(corpus)
    repo = get_repo(corpus)
    total = repo.count_records()
    indexed = rag.indexed_count()

    _sidebar(service, rag)

    c_title, c_status = st.columns([3, 1])
    with c_title:
        st.title("📜 Genealogy RAG — Baltimore Marriage Records")
        st.caption(
            "1930s–1940s vital records · swappable OCR corpus (sidebar) · "
            "LangGraph extraction · hybrid RAG search over SQLite + Chroma"
        )
    with c_status:
        if total > 0:
            st.markdown(_badge(f"{total} records", "green"), unsafe_allow_html=True)
        else:
            st.markdown(_badge("Empty corpus", "yellow"), unsafe_allow_html=True)
        if indexed > 0:
            st.markdown(_badge(f"{indexed} indexed", "blue"), unsafe_allow_html=True)

    if total == 0:
        hints = {
            "vlm": "Build + ingest the **VLM corpus** from the sidebar.",
            "qwen25": "Run `make scan-qwen25-corpus` to populate the zero-shot corpus.",
            "docling": "Use **Import notebook batch** or **Upload** to seed the corpus.",
        }
        st.warning(
            f"No records in the {corpus} database yet. {hints.get(corpus, '')}",
            icon="📥",
        )
    else:
        st.success(
            f"**{corpus}** corpus ready — **{total}** records in SQLite, "
            f"**{indexed}** vectors in Chroma. Search does not re-run OCR or agents.",
            icon="✅",
        )

    st.divider()

    if _is_ocr_backed(corpus):
        tab_search, tab_upload, tab_gold, tab_browse, tab_import = st.tabs(
            [
                "💬 Search",
                "📤 Upload",
                "📊 Gold eval",
                "🗂️ Browse",
                "📥 Import",
            ]
        )
        with tab_search:
            _search_section(rag)
        with tab_upload:
            _upload_section(service)
        with tab_gold:
            _gold_section(service)
        with tab_browse:
            _browse_section(service, total)
        with tab_import:
            _import_section(service)
    else:
        # Read-only zero-shot corpus (qwen25): search + browse only.
        tab_search, tab_browse = st.tabs(["💬 Search", "🗂️ Browse"])
        with tab_search:
            _search_section(rag)
        with tab_browse:
            _browse_section(service, total)


if __name__ == "__main__":
    main()
