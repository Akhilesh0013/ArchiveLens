# Gold labels — human ground truth for evaluation

These are hand-labeled "correct answers" used to measure extraction accuracy
(Docling→LLM vs TrOCR→LLM) and to ground the LLM-as-a-Judge.

## How to label

1. Open a page image in `gold/images/`, e.g. `1_content_p001.png`.
2. Copy `_template.json` to a file named after the image:
   `gold/1_content_p001.json` (same stem, `.json` extension).
3. Read the certificate and fill in each field with what you SEE.

## Labeling rules (match the extraction convention)

- **Transcribe what is legibly written**, correcting nothing about your own
  reading. If the printed/typed name is "ALLINE POYAS", label it "ALLINE POYAS"
  even though OCR misread it as "ALLIME". Gold = the truth on the document.
- If a field is **blank or illegible**, use `null`.
- **Occupation "NONE"** printed on the form is a real value — label `"NONE"`.
- **Dates:** put the day number in `marriage_day`, the month in `marriage_month`
  (as written, e.g. "Aug"), and the full year in `marriage_year` ONLY if you can
  read all four digits; otherwise `null`. Put any year phrase in
  `marriage_year_text`.
- `marital_status`: single / widow / widower / divorced (as written).
- Keep `source_pdf` and `source_page` matching the image filename.

## Files

- `images/` — page images to label (14 pages across both PDFs)
- `_template.json` — blank schema to copy
- `<stem>.json` — your completed labels (one per image)
