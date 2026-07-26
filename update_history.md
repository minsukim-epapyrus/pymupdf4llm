# ReStyle — Text-Style Recovery for pymupdf4llm

Source changes on top of **pymupdf4llm 1.28.0** (`git diff 1.28.0`). The goal is
to improve the *fidelity of inline text styling and headings* in the Markdown
output — i.e. to recover styles that the document actually renders but the
current serializer drops. Evaluated on the ParseBench **text_formatting**
dimension (476 documents).

All changes are **source edits** intended for review/PR; there is no output
post-processing. Two files are touched:

- `src/helpers/document_layout.py` (the active Layout-model emission path)
- `src/helpers/get_text_lines.py` (shared line/span assembly)

## Method

Every patch was gated the same way, to avoid benchmark-gaming:

1. **Verify a real defect first.** A candidate is implemented only after
   confirming the signal physically exists in the PDF (font/geometry), not merely
   that a score would move. (Example: a "bold weight" idea was *rejected* because
   the target spans are Regular/Light fonts — see Rejected below.)
2. **Two-stage A/B, both directions.** Stage 1: a small fixture set. Stage 2: the
   full 476-doc run vs an unpatched 1.28.0 baseline, reporting **improvements and
   regressions**. A positive aggregate delta alone was not treated as sufficient
   for adoption.
3. **Overfit self-audit.** Any change shaped to the evaluator's matching format
   rather than to output fidelity was rejected (see D7-ext2 below).

Baseline (unpatched 1.28.0, `use_ocr=False`): semantic_formatting = **0.5231**.

## Adopted changes

| # | Change | File / function | Effect (vs baseline) |
|---|---|---|---|
| **D4** | A body "text" box that is entirely bold and short is really a heading — promote it to a section-header instead of emitting it inline. | `document_layout.py`: `is_bold_title` + dispatch | is_title ▲ |
| **D4-ext** | A box that *starts* with bold title line(s) and continues with body text (`**DRAFT RED HERRING PROSPECTUS**` then `Dated April 24, 2025 …`) is split: leading bold lines → heading, remainder → body. | `document_layout.py`: `_leading_bold_title_lines` + dispatch | is_title ▲, **title_hierarchy +0.0102** |
| **D7** | Geometric superscript detection. MuPDF's `TEXT_FONT_SUPERSCRIPT` flag misses raised citation/footnote markers (`disease[22]`). Per line, take the largest span as the body size and the full-size baseline; a smaller span sitting above the baseline is a superscript → `<sup>`. Body text only (disabled in heading renderers); guarded to short markers (≤10 chars) that contain an alphanumeric character (so a raised closing quote/period is **not** marked). | `document_layout.py`: `get_styled_text` / `_geom_sup` | **is_sup 0.5484 → 0.5932** |
| **D7-ext** | Keep a small, baseline-shifted span from being merged into its neighbour during line assembly — merging erased the size/baseline that identifies an inline citation as a superscript. The guard is sup/sub-specific (small **and** baseline-shifted), so same-size word fragments and larger drop-caps still join normally (titles like "OVERVIEW" are not fragmented). | `get_text_lines.py`: `sanitize_spans` | enables D7 on inline citations |
| **D7-sub** | Geometric **subscript** detection — the mirror of D7. MuPDF exposes *no* subscript flag at all, so lowered small spans (chemical formulae like H₂O, math subscripts) were dropped entirely. Same short-marker/alnum/size guards; the baseline test is *below* instead of above → `<sub>`. | `document_layout.py`: `get_styled_text` / `_geom_sub` | **is_sub 0.1667 → 0.6667** |
| **D8** | **Vector strikethrough / underline** recovery. When a strike/underline is drawn as a thin vector line or rectangle (not a font `char_flag`), align its x-range to the RAWDICT character boxes and set the strikeout/underline flag on the covered characters only — splitting the span at whole-word boundaries (never on empty/whitespace pieces). A line through the character middle → `~~`; along the baseline → `<u>`. Guards drop over-wide rules, dashed annotation connectors and form lines; table boxes are excluded. `get_styled_text` additionally keeps a common outer **bold/italic** run continuous across an inserted `<u>`/`~~` (`**a <u>b</u> c**`). | `document_layout.py`: `_thin_hlines`, `_apply_decorators`, `_coalesce_outer_style_runs` | **is_strikeout 0.1522 → 0.2117**, is_underline 0.1907 → 0.3443 |
| **D9** | **Raster straight-underline** recovery for OCR/scanned pages. There the underline is drawn *pixels* (no vector, no `char_flag`; OCR emits no underline attribute). Render a 300 dpi grayscale pixmap, detect solid horizontal dark runs, convert to PDF coordinates, and feed them through the same D8 char-range marking (underline only). Gated to OCR pages; guards drop dashed/box borders (redaction boxes), footer / full-width decorative rules, dense repeated grid separators (newspaper-ad grids), and runs not aligned to an OCR word baseline. Straight underline only — raster strikeout and curved/handwritten underlines are out of scope. | `document_layout.py`: `_pixel_hlines`, `_is_ocr_page`, `_apply_decorators(underline_only)` | is_underline 0.3443 → 0.3831 (12 rules; **0 false positives** after guard tightening) |

### Result (full 476-doc A/B vs unpatched 1.28.0)

| metric | baseline | ReStyle | Δ |
|---|---|---|---|
| **semantic_formatting** | 0.5231 | **0.5332** | **+0.0101** |
| is_sup | 0.5484 | 0.5932 | +0.0448 |
| is_sub | 0.1667 | 0.6667 | +0.5000 |
| is_strikeout | 0.1522 | 0.2117 | +0.0594 |
| is_underline | 0.1907 | 0.3831 | +0.1924 |
| is_title | 0.5099 | 0.5264 | +0.0165 |
| title_hierarchy_percent | 0.4538 | 0.4640 | +0.0102 |
| rule pass_rate | 0.4769 | 0.5009 | +0.0240 |

**39 documents improved, 1 regressed** (a single Oromo-language document,
−0.0129 — see Known limitation). The geometric guards produced **zero** false
superscripts/subscripts across the corpus.

**D8 boundary trade-off (accepted):** 7 underline rules across 4 documents change
PASS→FAIL because the recovered underline follows the *actual* drawn line, which
stops before a trailing colon or period, whereas the benchmark's exact
`<u>query</u>` matcher expects the looser extent (e.g. `<u>Rounding</u>:` vs the
query `Rounding:`). Inspection of the rendered PDFs confirms the new output is the
more faithful one; underline is not part of the `semantic_formatting` metric, so
the net is still a large underline gain and no `semantic_formatting` regression.

## Rejected experiments (recorded so they are not re-tried)

- **Bold via ink density (D2)** — *not a real defect.* The bold spans that go
  unmarked are Regular/Light font files (`pymupdf.Font.is_bold == 0`); pixel
  ink-density was misleading. Marking them bold would be benchmark-gaming.
- **Underline-mask removal (D1a)** — net-negative. MuPDF's underline char-flag is
  noisy on some documents and fragmented words (`L<u>ette</u>r …`). Reverted.
- **Adjacent-citation split (D7-ext2)** — *ParseBench overfit.* Splitting a
  visually contiguous superscript run (`[42-44][44-54]`) into two `<sup>` tags was
  motivated only by matching the evaluator's per-citation rules; it fabricates an
  artificial tag boundary and is **less** faithful to the source (the markers
  render as one run). Reverted.
- **"Titles aren't sentences" guard** — skipping a leading bold line that ends in
  `. , : ;`. Net-negative: it fixed one document but broke legitimate
  colon-terminated headings ("NOTE:", section labels). Reverted.

## Known limitation

One document (Oromo-language, `reverRo`) regresses −0.0129: D4-ext promotes bold
instruction lines (6–8 word phrases with no terminal punctuation) to headings.
They are indistinguishable from bold titles without language knowledge; the only
clean textual signal (terminal punctuation) was net-negative overall (see the
rejected guard above), so this is left as-is.

## Base

Branched from tag **1.28.0**. `git diff 1.28.0` is the complete change set.
