.PHONY: help install sync ui import-batch list-records run-gold-docling run-gold-vlm run-gold-qwen25 upload rag-index rag-index-vlm rag-index-qwen25 rag-ask rag-ask-qwen25 rag-search rag-search-qwen25 gold-plots-docling gold-plots-vlm build-vlm-cache build-vlm-corpus-cache ingest-vlm-corpus scan-qwen25-corpus scan-qwen25-status scan-qwen25-watch


help:
	@echo "Document Understanding — Available commands"
	@echo "  make install       — create venv & install all deps (one-time)"
	@echo "  make sync          — sync deps with pyproject.toml"
	@echo "  make run-ui        — launch Streamlit upload app"
	@echo "  make import-batch  — import notebook JSONL into SQLite (no LLM re-run)"
	@echo "  make run-gold-docling — gold eval with Docling OCR → docling-gold-results/plots/"
	@echo "  make run-gold-vlm     — gold eval with VLM OCR (auto-builds pages.jsonl) → vlm-gold-results/plots/"
	@echo "  make build-vlm-cache  — manually rebuild gold VLM OCR cache (14 pages → artifacts/vlm/pages.jsonl)"
	@echo "  make build-vlm-corpus-cache — VLM OCR all processed PDFs → artifacts/vlm/corpus_pages.jsonl"
	@echo "  make ingest-vlm-corpus — agent-extract VLM OCR into records_vlm.db"
	@echo "  make run-gold-qwen25   — zero-shot Qwen2.5-VL-7B gold eval (14 pages) → artifacts/agent/qwen-2.5-vl-7b/"
	@echo "  make scan-qwen25-corpus — zero-shot Qwen2.5-VL-7B scan of processed PDFs → records_qwen25_zeroshot.db (no agent)"
	@echo "  make scan-qwen25-status — one-off progress snapshot for the zero-shot scan"
	@echo "  make scan-qwen25-watch  — live progress for the zero-shot scan (refreshes every 5s)"
	@echo "  make rag-index-vlm    — build Chroma index from records_vlm.db"
	@echo "  make rag-index-qwen25 — build Chroma index from records_qwen25_zeroshot.db"
	@echo "  make rag-ask-qwen25 Q=\"...\"    — ask a question over the qwen25 zero-shot corpus"
	@echo "  make rag-search-qwen25 Q=\"...\" — hybrid search over the qwen25 zero-shot corpus"
	@echo "  make list-records  — list records in the database"
	@echo "  make upload FILE=path/to/cert.pdf — upload through agent pipeline"
	@echo "  make rag-index     — build vector index from ingested DB records"
	@echo "  make rag-ask Q=\"...\" — ask a question over indexed records"
	@echo "  make rag-search Q=\"...\" — hybrid search over indexed records"
	@echo "  make gold-plots-docling — replot from docling-gold-results (no LLM)"
	@echo "  make gold-plots-vlm     — replot from vlm-gold-results (no LLM)"


install:
	uv python pin 3.11
	uv venv --python 3.11
	uv sync --extra dev

sync:
	uv sync --extra dev

ui:
	uv run streamlit run streamlit_app.py

import-batch:
	uv run python -m app.cli import-batch

run-gold-docling:
	uv run --extra dev python -m app.cli run-gold --ocr docling

run-gold-vlm:
	uv run --extra dev python -m app.cli run-gold --ocr vlm

run-gold-qwen25:
	uv run --extra dev python -m app.cli run-gold-qwen25

list-records:
	uv run python -m app.cli list --limit 20

upload:
	uv run python -m app.cli upload $(FILE) $(if $(OCR),--ocr $(OCR),)

build-vlm-cache:
	uv run python -m app.cli build-vlm-cache

build-vlm-corpus-cache:
	uv run python -m app.cli build-vlm-corpus-cache

ingest-vlm-corpus:
	uv run --extra dev python -m app.cli ingest-vlm-corpus

scan-qwen25-corpus:
	uv run python -m app.cli scan-qwen25-corpus

RECORDS_JSONL = artifacts/vlm/qwen25_zeroshot_records.jsonl
ERRORS_JSONL  = artifacts/vlm/qwen25_zeroshot_errors.jsonl

scan-qwen25-status:
	@printf "done:   %s/225\n" "$$(wc -l < $(RECORDS_JSONL) 2>/dev/null | tr -d ' ' || echo 0)"
	@printf "errors: %s\n" "$$([ -f $(ERRORS_JSONL) ] && wc -l < $(ERRORS_JSONL) | tr -d ' ' || echo 0)"

scan-qwen25-watch:
	@while true; do \
		printf "\033[2J\033[H"; \
		printf "Zero-shot Qwen2.5-VL scan — %s\n" "$$(date '+%H:%M:%S')"; \
		printf "done:   %s/225\n" "$$(wc -l < $(RECORDS_JSONL) 2>/dev/null | tr -d ' ' || echo 0)"; \
		printf "errors: %s\n\n" "$$([ -f $(ERRORS_JSONL) ] && wc -l < $(ERRORS_JSONL) | tr -d ' ' || echo 0)"; \
		printf "latest:\n"; \
		tail -n 3 $(RECORDS_JSONL) 2>/dev/null | python3 -c 'import sys,json;[print("  ", (r.get("source_pdf")), "p"+str(r.get("source_page")), "—", (r.get("groom") or {}).get("full_name","?"), "/", (r.get("bride") or {}).get("full_name","?")) for r in map(json.loads, sys.stdin)]' 2>/dev/null; \
		sleep 5; \
	done

rag-index-docling:
	uv run python -m app.cli rag-index --corpus docling

rag-index-vlm:
	uv run python -m app.cli rag-index --corpus vlm

rag-index-qwen25:
	uv run python -m app.cli rag-index --corpus qwen25

rag-ask:
	uv run python -m app.cli rag-ask "$(Q)"

rag-ask-qwen25:
	uv run python -m app.cli rag-ask "$(Q)" --corpus qwen25

rag-search:
	uv run python -m app.cli rag-search "$(Q)"

rag-search-qwen25:
	uv run python -m app.cli rag-search "$(Q)" --corpus qwen25

gold-plots-docling:
	uv run --extra dev python scripts/generate_gold_eval_plots.py --ocr docling

gold-plots-vlm:
	uv run --extra dev python scripts/generate_gold_eval_plots.py --ocr vlm
