import base64
import io
import json
import os
import math
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union
import textwrap

import pymupdf
import numpy as np
import tabulate
from pymupdf import mupdf
from pymupdf4llm.helpers import utils
from pymupdf4llm.helpers.get_text_lines import get_raw_lines
from pymupdf4llm.ocr import OCRMode

try:
    from tqdm import tqdm as ProgressBar
except ImportError:
    from pymupdf4llm.helpers.progress import ProgressBar

from dataclasses import dataclass

pymupdf.TOOLS.unset_quad_corrections(True)

INFO_MESSAGES = io.StringIO()
GRAPHICS_TEXT = "\n![](%s)\n"

FLAGS = (
    0
    | pymupdf.TEXT_COLLECT_STYLES
    | pymupdf.TEXT_COLLECT_VECTORS
    | pymupdf.TEXT_PRESERVE_IMAGES
    | pymupdf.TEXT_ACCURATE_BBOXES
    | pymupdf.TEXT_MEDIABOX_CLIP
    | pymupdf.TEXT_IGNORE_ACTUALTEXT
)
BULLETS = tuple(utils.BULLETS)


def get_table_details(tab_dict, table_blocks):
    """Create a TableDetails object.

    The table dictionary is as returned by the Layout module with option
    "return_raw=True".
    """
    tab_det = TableDetails()
    tab_det.bbox = tab_dict["group_bbox"]  # bounding box
    x0, y0, x1, y1 = tab_det.bbox
    grid = tab_dict.get("table_grid")  # Layout's GridPrediction object
    cells = []  # cell bounding boxes
    extract = []  # cell text content
    md_cells = []  # cell markdown content
    h_lines = [y0] + [h + y0 for h in grid.h_lines] + [y1]
    v_lines = [x0] + [v + x0 for v in grid.v_lines] + [x1]
    tab_det.row_count = len(h_lines) - 1
    tab_det.col_count = len(v_lines) - 1
    for i in range(tab_det.row_count):
        row = []
        text_row = []
        md_row = []
        for j in range(tab_det.col_count):
            cell_bbox = (v_lines[j], h_lines[i], v_lines[j + 1], h_lines[i + 1])
            row.append(cell_bbox)
            text = utils.extract_cells(
                table_blocks, cell_bbox, markdown=False, ocrpage=False
            )
            text_row.append(text)
            md_text = utils.extract_cells(
                table_blocks, cell_bbox, markdown=True, ocrpage=False
            )
            md_row.append(md_text)
        cells.append(row)
        extract.append(text_row)
        md_cells.append(md_row)
    tab_det.cells = cells
    tab_det.extract = extract
    tab_det.markdown = utils.table_to_markdown(md_cells)
    return tab_det


def wrap_table_for_tabulate(table, max_width=100, min_col_width=10):
    """
    Pre-wraps a table (List[List[str]]) so that tabulate cannot produce
    absurdly wide tables. Each column gets a width budget based on max_width.
    """
    if not table:
        return table

    # Number of columns
    num_cols = max(len(row) for row in table)

    # Distribute width evenly
    base_width = max(min_col_width, max_width // num_cols)
    col_widths = [base_width] * num_cols

    wrapped_table = []

    for row in table:
        new_row = []
        for col_idx, cell in enumerate(row):
            cell = cell or ""
            width = col_widths[col_idx]

            # Wrap the cell text
            lines = textwrap.wrap(cell, width=width) or [""]
            new_row.append("\n".join(lines))

        wrapped_table.append(new_row)

    return wrapped_table


def make_page_chunk(doc, page, text, string_lengths) -> Dict:
    """Create a page chunk dictionary for output.

    Args:
        doc: the ParsedDocument object
        page: the PageLayout object
        text: the page text string

    Returns:
        dict: page chunk dictionary
    """
    assert len(page.boxes) == len(string_lengths)
    chunk = defaultdict(lambda: None)
    page_tocs = [t for t in doc.toc if t[-1] == page.page_number]
    chunk["metadata"] = doc.metadata | {
        "file_path": doc.filename,
        "page_count": doc.page_count,
        "page_number": page.page_number,
    }

    chunk["toc_items"] = page_tocs
    page_boxes = []
    for i in range(len(page.boxes)):
        b = page.boxes[i]
        start = string_lengths[i - 1] if i > 0 else 0
        stop = string_lengths[i]
        page_boxes.append(
            {
                "index": i,
                "class": b.boxclass,
                "bbox": tuple(pymupdf.IRect(b.x0, b.y0, b.x1, b.y1)),
                "pos": (start, stop),
            }
        )
    chunk["page_boxes"] = page_boxes
    chunk["text"] = text
    return chunk


def omit_if_pua_char(text):
    """Check if character is in the Private Use Area (PUA) of Unicode."""
    if len(text) != 1:  # only single characters are checked
        return text
    o = ord(text)
    if (
        (0xE000 <= o <= 0xF8FF)
        or (0xF0000 <= o <= 0xFFFFD)
        or (0x100000 <= o <= 0x10FFFD)
    ):
        return ""
    return text


def create_list_item_levels(layout_info):
    """Map the layout box number of each list-item to its hierarchy level.

    Args:
        layout_info (list): the bbox list "page.layout_information"

    Returns:
        dict: {bbox sequence number: level} where level is 1 for top-level.
    """
    segments = []  # list of item segments
    segment = []  # current segment

    # Create segments of contiguous list items. Each non-list-item finishes
    # the current segment. Also, two list-items in a row belonging to different
    # page text columns end the segment after the first item.
    for i, item in enumerate(layout_info):
        if item.boxclass != "list-item":  # bbox class is no list-item
            if segment:  # end and save the current segment
                segments.append(segment)
                segment = []
            continue
        if segment:  # check if we need to end the current segment
            _, prev_item = segment[-1]
            if item.x0 > prev_item.x1 or item.y1 < prev_item.y0:
                # end and save the current segment
                segments.append(segment)
                segment = []
        segment.append((i, item))  # append item to segment
    if segment:
        segments.append(segment)  # append last segment

    item_dict = {}  # dictionary of item index -> (level
    if not segments:  # no list items found
        return item_dict

    # walk through segments and assign levels
    for i, s in enumerate(segments):
        if not s:  # skip empty segments
            continue
        s.sort(key=lambda x: x[1].x0)  # sort by x0 coordinate of the bbox

        # list of leveled items in the segment: (idx, bbox, level)
        # first item has level 1
        leveled_items = [(s[0][0], s[0][1], 1)]
        for idx, bbox in s[1:]:
            prev_idx, prev_bbox, prev_lvl = leveled_items[-1]
            # x0 coordinate increased by more than 10 points: increase level
            if bbox.x0 > prev_bbox.x0 + 10:
                curr_lvl = prev_lvl + 1
                leveled_items.append((idx, bbox, curr_lvl))
            else:
                leveled_items.append((idx, bbox, prev_lvl))
        for idx, bbox, lvl in leveled_items:
            item_dict[idx] = lvl
    return item_dict


def is_monospaced(textlines):
    """Detect text bboxes with all mono-spaced lines.

    Returns True if all lines are mono-spaced.
    Used to output code blocks.
    """
    line_count = len(textlines)
    mono = 0

    for l in textlines:
        all_mono = all(
            bool(
                s["flags"] & pymupdf.TEXT_FONT_MONOSPACED
                and not utils.is_ocr_text(s)
            )
            for s in l["spans"]
            if not s["text"].isspace()
        )
        if all_mono:
            mono += 1
    return mono == line_count


def is_superscripted(line):
    spans = line["spans"]
    line_bbox = line["bbox"]
    if not spans:
        return False
    span0 = spans[0]
    if span0["flags"] & 1:  # check for superscript flag
        return True
    if len(spans) < 2:  # single span line: skip
        return False
    if span0["origin"][1] < spans[1]["origin"][1] and span0["size"] < spans[1]["size"]:
        return True
    return False


def get_plain_text(spans):
    """Output text without any markdown or other styling.
    Parameter is a list of span dictionaries. The spans may come from
    one or more original "textlines" items.
    Returns the text string of the boundary box.
    """
    output = ""
    for i, s in enumerate(spans):
        superscript = s["flags"] & 1
        span_text = s["text"].strip()  # remove leading/trailing spaces
        if superscript:
            # enclose superscripted text in brackets if first span
            if i == 0:
                span_text = f"[{span_text}] "
            elif output.endswith(" "):
                output = output[:-1]
        # resolve hyphenation
        if output.endswith("- ") and len(output.split()[-1]) > 2:
            output = output[:-2]
        output += span_text + " "
    return output


def list_item_to_text(textlines, level) -> str:
    """
    Convert "list-item" bboxes to text.
    """
    if not textlines:
        return ""
    indent = "   " * (level - 1)  # indentation based on level
    output = indent
    line = textlines[0]
    x0 = line["bbox"][0]  # left of first line
    spans = line["spans"]
    span0 = line["spans"][0]
    span0_text = span0["text"].strip()

    if not omit_if_pua_char(span0_text):
        spans.pop(0)
        if spans:
            x0 = spans[0]["bbox"][0]

    for line in textlines[1:]:
        this_x0 = line["bbox"][0]
        if this_x0 < x0 - 2:
            line_output = get_plain_text(spans)
            output += line_output
            output = output.rstrip() + f"\n\n{indent}"
            spans = line["spans"]
            if not omit_if_pua_char(spans[0]["text"].strip()):
                spans.pop(0)
        else:
            spans.extend(line["spans"])
        x0 = this_x0  # store this left coordinate
    line_output = get_plain_text(spans)
    output += line_output

    return output.rstrip() + "\n\n"


def footnote_to_text(textlines) -> str:
    """
    Convert "footnote" bboxes to text.
    """
    if not textlines:
        return ""
    # we render footnotes as blockquotes
    output = "> "
    line = textlines[0]
    spans = line["spans"]

    for line in textlines[1:]:
        # superscripted line starts a new footnote line
        if is_superscripted(line):
            line_output = get_plain_text(spans)
            output += line_output
            output = output.rstrip() + "\n\n> "
            spans = line["spans"]
        else:
            spans.extend(line["spans"])
    line_output = get_plain_text(spans)
    output += line_output

    return output.rstrip() + "\n\n"


def code_block_to_text(textlines):
    """Output a code block in plain text format.

    Basic difference is that lines are separated by line breaks.
    """
    output = ""
    for line in textlines:
        line_text = ""
        for s in line["spans"]:
            span_text = s["text"]
            line_text += span_text
        output += line_text.rstrip() + "\n"
    output += "\n\n"
    return output


def text_to_text(textlines, ignore_code: bool = False):
    """
    Convert "text" bboxes to plain text, as well as boxclasses
    not specifically handled elsewhere.
    The text of all spans of all lines is written without line breaks.
    At the end, two newlines are added to separate from the next block.
    """
    if not textlines:
        return ""
    if is_superscripted(textlines[0]):  # check for superscript
        # handle mis-classified text boundary box
        return footnote_to_text(textlines)
    # handle completely mnonospaced textlines as code block
    if not ignore_code and is_monospaced(textlines):
        return code_block_to_text(textlines)

    spans = []
    for l in textlines:
        for s in l["spans"]:
            assert isinstance(s, dict)
            spans.append(s)
    output = get_plain_text(spans)
    return output + "\n\n"


def picture_text_to_text(textlines, ignore_code: bool = False, clip=None):
    """Convert text extracted from images to plain text format.

    In case text has been written inside a picture bbxox, we want to output it
    in some form. Because we cannot be sure about the formatting we simply
    write it line by line wrapped by markers.
    """
    if not textlines:
        return "\n"
    output = "----- Start of picture text -----\n"
    for tl in textlines:
        line_text = " ".join([s["text"] for s in tl["spans"]])
        output += line_text.rstrip() + "\n"
    output += "----- End of picture text -----\n"
    return output + "\n"


def fallback_text_to_text(textlines, ignore_code: bool = False, clip=None):
    """Convert text extracted from unrecognized tables.

    We hope for some sort of table structure being present in the text spans:
    The maximum span count in the lines is assumed to equal column count.
    """
    span_count = max(len(tl["spans"]) for tl in textlines)
    lines = []
    output = ""
    for tl in textlines:
        spans = tl["spans"]
        # prepare a row with empty strings in each cell
        line = [""] * span_count
        if len(spans) < span_count and spans[0]["bbox"][0] > clip[0] + 10:
            i = 1
        else:
            i = 0
        for j, s in enumerate(spans, start=i):
            line[j] = f'{s["text"].strip()} '
        lines.append(line)
    tab_text = tabulate.tabulate(
        lines,
        tablefmt="grid",
        disable_numparse=True,
        maxcolwidths=int(100 / span_count),
    )
    output += tab_text + "\n"
    return output + "\n"


def _get_styled_text_legacy(spans, geom_sup=True):
    """Output text with markdown style codes based on font properties.
    Parameter is a list of span dictionaries. The spans may come from
    one or more original "textlines" items.
    Returns the text string and the suffix for continuing styles.
    The text string always ends with the suffix and a space
    """
    output = ""
    prefix = ""
    suffix = ""
    old_line = 0
    old_block = 0

    # ReStyle D7 (TextStyle Recovery): geometric superscript detection. MuPDF's
    # TEXT_FONT_SUPERSCRIPT flag misses raised citation markers ([22], footnote
    # numbers) that are smaller than the line and sit above its baseline. Per
    # line, take the largest span as the body size and the baseline of full-size
    # spans; a smaller, raised span is a superscript.
    _line_norm = {}
    for _s in spans:
        _line_norm.setdefault(_s.get("line", 0), []).append(_s)
    _norm = {}
    for _ln, _ss in _line_norm.items():
        _nsz = max(x["size"] for x in _ss)
        _base = max(
            (x["origin"][1] for x in _ss if x["size"] >= 0.95 * _nsz),
            default=None,
        )
        _norm[_ln] = (_nsz, _base)

    def _geom_sup(sp):
        # a superscript marker is short (a citation/footnote number); this guard
        # also prevents a smaller-font neighbouring column (merged onto the same
        # line in interleaved multi-column layouts) from being flagged wholesale.
        _t = sp["text"].strip()
        if len(_t) > 10 or not any(c.isalnum() for c in _t):
            return False  # too long, or pure punctuation (a raised quote/period)
        _nsz, _base = _norm.get(sp.get("line", 0), (0, None))
        if not _nsz or _base is None:
            return False
        if sp["size"] >= 0.85 * _nsz:  # not smaller than body
            return False
        return sp["origin"][1] < _base - 0.1 * _nsz  # raised above the baseline

    def _geom_sub(sp):
        # ReStyle D7-sub: geometric subscript — the mirror of _geom_sup. MuPDF has
        # no subscript flag at all (only TEXT_FONT_SUPERSCRIPT), so a lowered small
        # span (chemical formula H2O, math subscript) can only be recovered
        # geometrically. Same short-marker/alnum/size guards; baseline is *below*.
        _t = sp["text"].strip()
        if len(_t) > 10 or not any(c.isalnum() for c in _t):
            return False
        _nsz, _base = _norm.get(sp.get("line", 0), (0, None))
        if not _nsz or _base is None:
            return False
        if sp["size"] >= 0.85 * _nsz:  # not smaller than body
            return False
        return sp["origin"][1] > _base + 0.1 * _nsz  # dropped below the baseline

    for i, s in enumerate(spans):
        # decode font flags and char_flags properties
        superscript = (s["flags"] & pymupdf.TEXT_FONT_SUPERSCRIPT) or (
            geom_sup and _geom_sup(s)
        )
        subscript = geom_sup and not superscript and _geom_sub(s)
        mono = s["flags"] & pymupdf.TEXT_FONT_MONOSPACED and not utils.is_ocr_text(s)
        bold = (
            s["flags"] & pymupdf.TEXT_FONT_BOLD
            or s["char_flags"] & pymupdf.mupdf.FZ_STEXT_BOLD
        )
        italic = s["flags"] & pymupdf.TEXT_FONT_ITALIC
        strikeout = s["char_flags"] & pymupdf.mupdf.FZ_STEXT_STRIKEOUT
        underline = s["char_flags"] & pymupdf.mupdf.FZ_STEXT_UNDERLINE
        highlight = s["char_flags"] & pymupdf.mupdf.FZ_STEXT_HIGHLIGHT

        # compute styling prefix and suffix
        prefix = []
        suffix = []

        if superscript:
            prefix.append("<sup>")
            suffix.append("</sup>")

        if subscript:
            prefix.append("<sub>")
            suffix.append("</sub>")

        if bold:
            prefix.append("**")
            suffix.append("**")

        if italic:
            prefix.append("_")
            suffix.append("_")

        if strikeout:
            prefix.append("~~")
            suffix.append("~~")

        if underline:
            prefix.append("<u>")
            suffix.append("</u>")

        if highlight:
            prefix.append("<mark>")
            suffix.append("</mark>")

        if mono:
            prefix.append("`")
            suffix.append("`")

        prefix = "".join(prefix)
        suffix = "".join(reversed(suffix))

        span_text = s["text"].strip()  # remove leading/trailing spaces
        # convert intersecting link to markdown syntax
        # ltext = resolve_links(parms.links, s)
        # ltext = ""  # TODO: implement link resolution
        # if ltext:
        #     text = f"{hdr_string}{prefix}{ltext}{suffix} "
        # else:
        #     text = f"{prefix}{span_text}{suffix} "
        text = f"{prefix}{span_text}{suffix} "
        # Extend output string taking care of styles staying the same.
        if output.endswith(f"{suffix} "):
            output = output[: -len(suffix) - 1]
            # resolve hyphenation if old_block and old_line are not the same
            if (
                1
                and (old_block, old_line) != (s["block"], s["line"])
                and output.endswith("-")
                and len(output.split()[-1]) > 2
            ):
                output = output[:-1]
                text = span_text + suffix + " "
            elif superscript:
                text = span_text + suffix + " "
            else:
                text = " " + span_text + suffix + " "

        old_line = s["line"]
        old_block = s["block"]
        if superscript or subscript:
            output = output.rstrip(" ")  # attach the marker to the preceding word
        if s.get("_restyle_join_left"):
            output = output.rstrip(" ")
            text = text.lstrip(" ")
        output += text
    return output, suffix


def _outer_style_signature(span):
    """Return shared outer Markdown styles and eligibility for coalescing."""
    bold = bool(
        span["flags"] & pymupdf.TEXT_FONT_BOLD
        or span["char_flags"] & pymupdf.mupdf.FZ_STEXT_BOLD
    )
    italic = bool(span["flags"] & pymupdf.TEXT_FONT_ITALIC)
    # Sup/sub and monospace have attachment / delimiter semantics of their own.
    # Keep those paths on the established serializer.
    eligible = not (
        span["flags"]
        & (pymupdf.TEXT_FONT_SUPERSCRIPT | pymupdf.TEXT_FONT_MONOSPACED)
    )
    return (bold, italic), eligible


def _inner_decorator_signature(span):
    mask = (
        pymupdf.mupdf.FZ_STEXT_STRIKEOUT
        | pymupdf.mupdf.FZ_STEXT_UNDERLINE
        | pymupdf.mupdf.FZ_STEXT_HIGHLIGHT
    )
    return span["char_flags"] & mask


def _coalesce_outer_style_runs(spans, geom_sup):
    """Keep shared bold/italic outside D8 decorator transitions.

    A D8 character-range split can introduce an inner underline/strike boundary
    in a sentence that remains bold or italic throughout. Render the varying
    decorators first, then expose the whole run to the established serializer as
    one span carrying only the common outer styles.
    """
    result = []
    pos = 0
    while pos < len(spans):
        outer, eligible = _outer_style_signature(spans[pos])
        if not any(outer) or not eligible:
            result.append(spans[pos])
            pos += 1
            continue

        size = spans[pos]["size"]
        stop = pos + 1
        while stop < len(spans):
            this_outer, this_eligible = _outer_style_signature(spans[stop])
            if (
                this_outer != outer
                or not this_eligible
                or abs(spans[stop]["size"] - size) > 0.05 * max(size, 1)
            ):
                break
            stop += 1

        run = spans[pos:stop]
        decorators = {_inner_decorator_signature(span) for span in run}
        if (
            len(run) < 2
            or len(decorators) < 2
            or not any(span.get("_restyle_split") for span in run)
        ):
            result.extend(run)
            pos = stop
            continue

        inner_spans = []
        for span in run:
            inner = dict(span)
            inner["flags"] &= ~(
                pymupdf.TEXT_FONT_BOLD | pymupdf.TEXT_FONT_ITALIC
            )
            inner["char_flags"] &= ~pymupdf.mupdf.FZ_STEXT_BOLD
            inner_spans.append(inner)
        inner_text, _ = _get_styled_text_legacy(inner_spans, geom_sup=geom_sup)

        combined = dict(run[0])
        combined["text"] = inner_text.rstrip()
        combined["char_flags"] &= ~(
            pymupdf.mupdf.FZ_STEXT_STRIKEOUT
            | pymupdf.mupdf.FZ_STEXT_UNDERLINE
            | pymupdf.mupdf.FZ_STEXT_HIGHLIGHT
        )
        combined["bbox"] = pymupdf.Rect(run[0]["bbox"])
        for span in run[1:]:
            combined["bbox"] |= pymupdf.Rect(span["bbox"])
        result.append(combined)
        pos = stop
    return result


def get_styled_text(spans, geom_sup=True):
    """Output styled text while preserving common outer bold/italic runs."""
    spans = _coalesce_outer_style_runs(spans, geom_sup)
    return _get_styled_text_legacy(spans, geom_sup=geom_sup)


def list_item_to_md(textlines, level):
    """
    Convert "list-item" bboxes to markdown.
    The first line is prefixed with "- ". Subsequent lines are appended
    without line break if their rectangle does not start to the left
    of the previous line.
    Otherwise, a linebreak and "- " are added to the output string.
    2 units of tolerance is used to avoid spurious line breaks.

    This post-layout heuristics helps cover cases where more than
    one list item is contained in a single bbox.
    """

    if not textlines:
        return ""
    indent = "   " * (level - 1)  # indentation based on level
    line = textlines[0]
    x0 = line["bbox"][0]  # left of first line
    spans = line["spans"]
    span0 = line["spans"][0]
    span0_text = span0["text"].strip()

    starter = "- "
    if utils.startswith_bullet(span0_text):
        span0_text = span0_text[1:].strip()
        line["spans"][0]["text"] = span0_text
    elif span0_text.endswith(".") and span0_text[:-1].isdigit():
        starter = ""
    elif " " in span0_text:
        first_word = span0_text.split(" ")[0]
        if first_word.endswith(".") and first_word[:-1].isdigit():
            starter = ""

    if not omit_if_pua_char(span0["text"].strip()):
        # bullet was a PUA char: remove it
        spans.pop(0)
        if spans:
            x0 = spans[0]["bbox"][0]

    output = indent + starter
    for line in textlines[1:]:
        this_x0 = line["bbox"][0]
        if this_x0 < x0 - 2:
            line_output, suffix = get_styled_text(spans)
            output += line_output + f"\n\n{indent}{starter}"
            spans = line["spans"]
            if not omit_if_pua_char(spans[0]["text"].strip()):
                spans.pop(0)
        else:
            spans.extend(line["spans"])
        x0 = this_x0  # store this left coordinate
    line_output, suffix = get_styled_text(spans)
    output += line_output

    return output + "\n\n"


def footnote_to_md(textlines):
    """
    Convert "footnote" bboxes to markdown.
    The first line is prefixed with "> ". Subsequent lines are appended
    without line break if they do not start with a superscript.
    Otherwise, a linebreak and "> " are added to the output string.

    This post-layout heuristics helps cover cases where more than
    one list item is contained in a single bbox.
    """
    if not textlines:
        return ""
    line = textlines[0]
    spans = line["spans"]
    output = "> "
    for line in textlines[1:]:
        if is_superscripted(line):
            line_output, suffix = get_styled_text(spans)
            output += line_output + "\n\n> "
            spans = line["spans"]
        else:
            spans.extend(line["spans"])
    line_output, suffix = get_styled_text(spans)
    output += line_output

    return output + "\n\n"


def _thin_hlines(page):
    """Return plausible vector text decorators on *page*.

    ReStyle D8 (TextStyle Recovery) deliberately accepts only isolated, thin,
    horizontal, solid paths.  In particular, dashed review connectors and the
    horizontal sides of annotation boxes must not become text underlines.
    Whether a candidate actually is a decorator is decided later from its
    alignment with character geometry.
    """
    hlines = []
    for path in page.get_drawings():
        items = path.get("items", ())
        if len(items) != 1:
            continue  # boxes, callouts and other compound graphics
        dashes = (path.get("dashes") or "").replace(" ", "")
        if dashes not in ("", "[]0"):
            continue

        item = items[0]
        if item[0] == "l":
            p0, p1 = item[1:3]
            if abs(p0.y - p1.y) > 0.2:
                continue
            x0, x1 = sorted((p0.x, p1.x))
            y = (p0.y + p1.y) / 2
            width = path.get("width") or 0
            color = path.get("color")
        elif item[0] == "re":
            rect = pymupdf.Rect(item[1])
            if rect.height > 1.5:
                continue
            x0, x1 = rect.x0, rect.x1
            y = (rect.y0 + rect.y1) / 2
            width = max(path.get("width") or 0, rect.height)
            color = path.get("fill") or path.get("color")
        else:
            continue

        if x1 - x0 < 3 or width > 1.5:
            continue
        if (path.get("stroke_opacity") or 1) < 0.5 or (
            path.get("fill_opacity") or 1
        ) < 0.5:
            continue
        if color is not None and min(color) > 0.95:
            continue  # white knockout / annotation-box fill
        hlines.append((x0, y, x1, width))
    return hlines


def _is_ocr_page(blocks):
    """Return whether *blocks* predominantly contain invisible OCR text."""
    spans = [
        span
        for block in blocks
        if block.get("type") == 0
        for line in block.get("lines", ())
        for span in line.get("spans", ())
        if span.get("text", "").strip()
    ]
    if not spans:
        return False
    ocr_spans = sum(utils.is_ocr_text(span) for span in spans)
    return ocr_spans / len(spans) >= 0.8


def _pixel_hlines(page, dpi):
    """Return straight raster underline candidates in PDF coordinates.

    ReStyle D9 only examines pages whose text is predominantly an invisible OCR
    layer. At OCR resolution, dark horizontal runs are gap-closed per row and
    retained only when a similarly long run is stable across adjacent rows.
    The returned tuples match ``_thin_hlines``: ``(x0, y, x1, width)``.
    Character alignment and form-rule rejection remain the responsibility of
    ``_apply_decorators``.
    """
    raw = page.get_text("dict", flags=FLAGS)
    if not _is_ocr_page(raw.get("blocks", ())):
        return []

    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    scale = dpi / 72
    samples = np.frombuffer(pix.samples_mv, dtype=np.uint8)
    gray = samples.reshape(pix.height, pix.stride)[:, : pix.width]
    dark = gray < 200

    # Restrict the expensive run search to the underline band of OCR baselines.
    active_rows = np.zeros(pix.height, dtype=bool)
    for block in raw.get("blocks", ()):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", ()):
            if abs(line.get("dir", (1, 0))[0] - 1) > 1e-3:
                continue
            for span in line.get("spans", ()):
                if not span.get("text", "").strip() or not utils.is_ocr_text(span):
                    continue
                baseline = span["origin"][1]
                size = span["size"]
                y0 = round((baseline - 0.08 * size) * scale) - pix.y
                y1 = round((baseline + 0.42 * size) * scale) - pix.y
                y0 = max(0, y0)
                y1 = min(pix.height - 1, y1)
                if y0 <= y1:
                    active_rows[y0 : y1 + 1] = True

    gap = max(1, round(dpi / 75))  # 4 pixels at the default 300 dpi
    min_length = max(3, round(30 * scale))
    max_length = round(0.72 * pix.width)
    row_runs = []
    for y in np.flatnonzero(active_rows):
        xs = np.flatnonzero(dark[y])
        if not len(xs):
            continue
        starts = np.r_[0, np.flatnonzero(np.diff(xs) > gap + 1) + 1]
        stops = np.r_[starts[1:], len(xs)]
        for start, stop in zip(starts, stops):
            x0 = int(xs[start])
            x1 = int(xs[stop - 1]) + 1
            length = x1 - x0
            if (
                min_length <= length <= max_length
                and (stop - start) / length >= 0.70
            ):
                row_runs.append((int(y), x0, x1))

    if not row_runs:
        return []

    # Join stable runs in neighboring rows. Normal glyph strokes tend to be
    # shorter or change extent rapidly; a straight underline repeats its
    # horizontal extent over multiple scan rows.
    parents = list(range(len(row_runs)))

    def root(item):
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(left, right):
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    rows_to_runs = defaultdict(list)
    for index, (y, x0, x1) in enumerate(row_runs):
        for other in rows_to_runs[y - 1]:
            _, other_x0, other_x1 = row_runs[other]
            overlap = min(x1, other_x1) - max(x0, other_x0)
            shorter = min(x1 - x0, other_x1 - other_x0)
            if overlap >= 0.10 * shorter:
                union(index, other)
        rows_to_runs[y].append(index)

    grouped = defaultdict(list)
    for index, run in enumerate(row_runs):
        grouped[root(index)].append(run)
    components = grouped.values()

    hlines = []
    # A straight scan line can be slightly skewed, so its component may span
    # several rows vertically even though its effective stroke is still thin.
    max_vertical_span = max(2, round(4 * scale))
    max_thickness = max(2, round(2 * scale))
    for component in components:
        rows = sorted({item[0] for item in component})
        if len(rows) < 2 or rows[-1] - rows[0] + 1 > max_vertical_span:
            continue
        x0 = min(item[1] for item in component)
        x1 = max(item[2] for item in component)
        thickness = sum(item[2] - item[1] for item in component) / (x1 - x0)
        if thickness > max_thickness:
            continue
        # Gap closing tolerates antialiasing and scan breaks, but a semantic
        # underline still contains a long, genuinely solid stroke. Dashed box
        # borders can have high gap-closed fill while no row has a long
        # uninterrupted run.
        solid_ratio = 0
        for row in rows:
            segment = dark[row, x0:x1]
            indexes = np.flatnonzero(segment)
            if not len(indexes):
                continue
            starts = np.r_[0, np.flatnonzero(np.diff(indexes) > 1) + 1]
            stops = np.r_[starts[1:], len(indexes)]
            longest = max(
                indexes[stop - 1] - indexes[start] + 1
                for start, stop in zip(starts, stops)
            )
            solid_ratio = max(solid_ratio, longest / (x1 - x0))
        if solid_ratio < 0.50:
            continue
        y = sum(rows) / len(rows)
        width = thickness / scale
        pdf_y = (y + pix.y) / scale
        if pdf_y >= 0.90 * page.rect.height:
            continue  # footer ornaments are not text decorators
        hlines.append(
            (
                (x0 + pix.x) / scale,
                pdf_y,
                (x1 + pix.x) / scale,
                width,
            )
        )

    # Dense, regularly repeated parallel rules are cell / advertisement-grid
    # structure. Real underlines remain locally sparse even on multi-line text.
    return [
        hline
        for hline in hlines
        if sum(abs(other[1] - hline[1]) <= 50 for other in hlines) <= 12
    ]


def _span_raw_chars(span, raw_chars):
    """Map a possibly merged get_raw_lines span back to RAWDICT characters."""
    bbox = pymupdf.Rect(span["bbox"])
    baseline = span["origin"][1]
    size = span["size"]
    chars = []
    seen = set()
    for char in raw_chars:
        cbbox = pymupdf.Rect(char["bbox"])
        center_x = (cbbox.x0 + cbbox.x1) / 2
        if not (bbox.x0 - 0.1 <= center_x <= bbox.x1 + 0.1):
            continue
        if abs(char["origin"][1] - baseline) > max(0.5, 0.1 * size):
            continue
        key = (char["c"], tuple(char["bbox"]), tuple(char["origin"]))
        if key in seen:
            continue
        seen.add(key)
        chars.append(char)
    chars.sort(key=lambda c: (c["origin"][0], c["bbox"][0]))
    if "".join(c["c"] for c in chars) != span["text"]:
        return []
    return chars


def _apply_decorators(textlines, raw_blocks, hlines, *, underline_only=False):
    """Recover strike/underline styling at exact character boundaries.

    ``underline_only`` is used for D9 raster candidates, which must never
    synthesize a strikeout.
    """
    if not textlines or not raw_blocks or not hlines:
        return

    raw_chars = []
    for block in raw_blocks:
        for line in block.get("lines", ()):
            if abs(line.get("dir", (1, 0))[0] - 1) > 1e-3:
                continue
            for span in line.get("spans", ()):
                for char in span.get("chars", ()):
                    item = dict(char)
                    item["size"] = span["size"]
                    item["char_flags"] = span["char_flags"]
                    raw_chars.append(item)

    spans = [span for line in textlines for span in line.get("spans", ())]
    mapped = [_span_raw_chars(span, raw_chars) for span in spans]
    char_refs = [
        (sno, cno, char)
        for sno, chars in enumerate(mapped)
        for cno, char in enumerate(chars)
    ]
    if not char_refs:
        return

    strike = pymupdf.mupdf.FZ_STEXT_STRIKEOUT
    underline = pymupdf.mupdf.FZ_STEXT_UNDERLINE
    underline_lower = -0.42 if underline_only else -0.30
    underline_upper = 0.02 if underline_only else 0.08
    recovered = {strike: set(), underline: set()}
    touched = {strike: set(), underline: set()}

    for x0, y, x1, width in hlines:
        possible = {underline: []} if underline_only else {strike: [], underline: []}
        for sno, cno, char in char_refs:
            cbbox = pymupdf.Rect(char["bbox"])
            cx = (cbbox.x0 + cbbox.x1) / 2
            if not (x0 - 0.1 <= cx <= x1 + 0.1):
                continue
            size = char["size"]
            baseline = char["origin"][1]
            rel_y = (baseline - y) / size
            if not underline_only and 0.15 <= rel_y <= 0.65:
                possible[strike].append((sno, cno, char, abs(rel_y - 0.28)))
            elif underline_lower <= rel_y <= underline_upper:
                possible[underline].append(
                    (sno, cno, char, abs(rel_y + 0.12))
                )

        choices = [
            (sum(item[3] for item in items) / len(items), flag, items)
            for flag, items in possible.items()
            if items and any(item[2]["c"].isalnum() for item in items)
        ]
        if not choices:
            continue
        _, flag, items = min(choices, key=lambda item: item[0])

        # A real decorator begins and ends at its text.  Reject form rules,
        # margin connectors and other lines which extend well outside the
        # character run they appear to cover.
        covered = [pymupdf.Rect(item[2]["bbox"]) for item in items]
        text_x0 = min(rect.x0 for rect in covered)
        text_x1 = max(rect.x1 for rect in covered)
        size = sorted(item[2]["size"] for item in items)[len(items) // 2]
        tolerance = max(1.5, 0.35 * size)
        if text_x0 - x0 > tolerance or x1 - text_x1 > tolerance:
            continue
        minimum_width_limit = 1.5 if underline_only else 1.0
        if width > max(minimum_width_limit, 0.15 * size):
            continue

        selected = {(item[0], item[1]) for item in items}
        if underline_only:
            # OCR boxes and antialiased raster lines can put the endpoint
            # inside a punctuation glyph even though its centre lies just
            # outside. Preserve adjacent punctuation whose box actually
            # intersects the detected line.
            additions = set()
            for sno, cno, char in char_refs:
                if char["c"].isalnum() or char["c"].isspace():
                    continue
                if not (
                    (sno, cno - 1) in selected or (sno, cno + 1) in selected
                ):
                    continue
                cbbox = pymupdf.Rect(char["bbox"])
                rel_y = (char["origin"][1] - y) / char["size"]
                if (
                    underline_lower <= rel_y <= underline_upper
                    and cbbox.x1 >= x0 - 0.1
                    and cbbox.x0 <= x1 + 0.1
                ):
                    additions.add((sno, cno))
            selected.update(additions)
        recovered[flag].update(selected)
        # MuPDF may assign a line's flag to the glyph immediately beyond an
        # endpoint.  Include endpoint-touching spans in the range we rebuild,
        # while still setting the recovered flag by character centre only.
        for sno, cno, char in char_refs:
            cbbox = pymupdf.Rect(char["bbox"])
            size = char["size"]
            baseline = char["origin"][1]
            rel_y = (baseline - y) / size
            aligned = (
                flag == strike
                and 0.15 <= rel_y <= 0.65
                or flag == underline
                and underline_lower <= rel_y <= underline_upper
            )
            if aligned and cbbox.x1 >= x0 - 0.1 and cbbox.x0 <= x1 + 0.1:
                touched[flag].add(sno)

    for sno, span in enumerate(spans):
        chars = mapped[sno]
        if not chars or not any(sno in touched[flag] for flag in touched):
            continue
        pieces = []
        for cno, char in enumerate(chars):
            flags = char["char_flags"]
            if underline_only:
                # Preserve D8 decorators and any underline already carried by
                # this (possibly split) span.
                flags |= span["char_flags"] & (strike | underline)
            for flag in (strike, underline):
                if sno in touched[flag]:
                    if underline_only and flag == underline and flags & underline:
                        continue
                    flags &= ~flag
                    if (sno, cno) in recovered[flag]:
                        flags |= flag
            if pieces and pieces[-1][0] == flags:
                pieces[-1][1].append(char)
            else:
                pieces.append([flags, [char]])

        replacements = []
        for flags, piece_chars in pieces:
            piece_text = "".join(char["c"] for char in piece_chars)
            if not piece_text.strip():
                # get_styled_text strips span text before adding its wrappers.
                # Keeping a whitespace-only split would therefore emit empty
                # markers (for example **<mark></mark>**) and interrupt the
                # surrounding bold/highlight run. Its layout gap still makes
                # get_styled_text insert the required separating space.
                continue
            replacement = dict(span)
            replacement["text"] = piece_text
            replacement["char_flags"] = flags
            replacement["bbox"] = pymupdf.Rect(piece_chars[0]["bbox"])
            for char in piece_chars[1:]:
                replacement["bbox"] |= pymupdf.Rect(char["bbox"])
            replacement["origin"] = piece_chars[0]["origin"]
            replacement["_restyle_split"] = True
            replacements.append(replacement)

        for line in textlines:
            if span in line.get("spans", ()):
                pos = line["spans"].index(span)
                line["spans"][pos : pos + 1] = replacements
                break

    # get_styled_text normally inserts a space between every pair of spans.
    # RAWDICT boundaries can instead be two touching fragments of one word;
    # remember those joins whenever D8 split either side of the boundary.
    for line in textlines:
        line_spans = line.get("spans", ())
        for pos in range(1, len(line_spans)):
            prev = line_spans[pos - 1]
            curr = line_spans[pos]
            if not (prev.get("_restyle_split") or curr.get("_restyle_split")):
                continue
            if prev["text"].endswith((" ", "\t")) or curr["text"].startswith(
                (" ", "\t")
            ):
                continue
            size = max(prev["size"], curr["size"])
            same_baseline = abs(prev["origin"][1] - curr["origin"][1]) <= 0.1 * size
            gap = curr["bbox"].x0 - prev["bbox"].x1
            if same_baseline and abs(gap) <= 0.1 * size:
                curr["_restyle_join_left"] = True


def _leading_bold_title_lines(textlines):
    """ReStyle D4 (TextStyle Recovery): count the leading, fully-bold, short lines
    that form a title at the START of a text box. The layout model only marks
    font-size/geometry titles, so a body-size bold title (`**Legal Notices**`, or
    `**DRAFT RED HERRING PROSPECTUS**` followed by non-bold `Dated April 24,
    2025 ...`) is otherwise emitted inline and misses `is_title`. Emit the leading
    bold line(s) as a heading and the rest of the box as body. The whole box is
    the title when every line is bold. Returns 0 if the box does not begin with a
    bold title, or if the bold prefix is too long to be a title (> 12 words).
    """
    n = 0
    words = 0
    for l in textlines:
        spans = [s for s in l.get("spans", []) if (s.get("text") or "").strip()]
        if not spans:
            break
        all_bold = all(
            (s["flags"] & pymupdf.TEXT_FONT_BOLD) or (s["char_flags"] & pymupdf.mupdf.FZ_STEXT_BOLD)
            for s in spans
        )
        if not all_bold:
            break
        words += len(" ".join((s.get("text") or "") for s in spans).split())
        n += 1
        if words > 12:  # too long to be a title
            return 0
    return n if words >= 1 else 0


def section_hdr_to_md(header_level, textlines):
    """
    Convert "section-header" bboxes to markdown.
    """
    spans = []
    for l in textlines:
        for s in l["spans"]:
            assert isinstance(s, dict)
            spans.append(s)
    output, suffix = get_styled_text(spans, geom_sup=False)
    return f"{'#' * header_level} {output}\n\n"


def title_to_md(header_level, textlines):
    """
    Convert "title" bboxes to markdown.
    The line text itself is handled like normal text.
    TODO: Consider joining with section_hdr.
    """
    spans = []
    for l in textlines:
        for s in l["spans"]:
            assert isinstance(s, dict)
            spans.append(s)
    output, suffix = get_styled_text(spans, geom_sup=False)
    return f"{'#' * header_level} {output}\n\n"


def code_block_to_md(textlines):
    """Output a code block in markdown format."""
    output = "```\n"
    for line in textlines:
        line_text = ""
        for s in line["spans"]:
            span_text = s["text"]
            line_text += span_text
        output += line_text.rstrip() + "\n"
    output += "```\n\n"
    return output


def text_to_md(textlines, ignore_code: bool = False):
    """
    Convert "text" bboxes to markdown, as well as other boxclasses
    not specifically handled elsewhere.
    The line text is written without line breaks. At the end,
    two newlines are added to separate from the next block.
    """
    if not textlines:
        return ""
    if is_superscripted(textlines[0]):
        # exec advanced superscript detector
        return footnote_to_md(textlines)
    if not ignore_code and is_monospaced(textlines):
        return code_block_to_md(textlines)

    spans = []
    for l in textlines:
        for s in l["spans"]:
            assert isinstance(s, dict)
            spans.append(s)
    output, suffix = get_styled_text(spans)
    return output + "\n\n"


def picture_text_to_md(textlines, ignore_code: bool = False, clip=None):
    """Convert text extracted from images to plain text format.

    In case text has been written inside a picture bbxox, we want to output it
    in some form. Because we cannot be sure about the formatting we simply
    write it line by line wrapped by markers.
    """
    if not textlines:
        return "\n"
    output = "<!-- Start of picture text -->\n"
    for tl in textlines:
        line_text = " ".join([s["text"] for s in tl["spans"]])
        output += line_text.rstrip() + "<br>"
    output += "<!-- End of picture text -->\n"
    return output + "\n"


def fallback_text_to_md(textlines, ignore_code: bool = False, clip=None):
    """
    Convert text extracted from images to markdown format.
    """
    span_count = max(len(tl["spans"]) for tl in textlines)
    output = "<!-- Start of picture text -->\n"
    output += "|" * (span_count + 1) + "\n"
    output += "|" + "|".join(["---"] * span_count) + "|\n"
    for tl in textlines:
        ltext = "|" + "|".join([s["text"].strip() for s in tl["spans"]]) + "|\n"
        output += ltext
    output += "\n<!-- End of picture text -->\n"
    return output + "\n"


@dataclass
class TableDetails:
    bbox: tuple = None
    row_count: int = None
    col_count: int = None
    cells: list = None  # list of list of cell bbox coordinates
    extract: list = None  # list of list of cell plain text content
    markdown: str = None  # table markdown content


@dataclass
class LayoutBox:
    x0: float
    y0: float
    x1: float
    y1: float
    boxclass: str  # e.g. 'text', 'picture', 'table', etc.

    # if boxclass == 'picture' or 'formula', store image bytes
    image: Optional[bytes] = None

    # if boxclass == 'table'
    table: Optional[Dict] = None

    # text line information for text-type boxclasses
    max_fontsize: Optional[int] = None
    header_level: Optional[int] = 0  # one of 1..6 for title/section-header
    textlines: Optional[List[Dict]] = None


@dataclass
class PageLayout:
    page_number: int
    width: float
    height: float
    boxes: List[LayoutBox]
    full_ocred: bool = False  # whether the page is an OCR'd page
    fulltext: Optional[List[Dict]] = None  # full page text in extractDICT format
    words: Optional[List[Dict]] = None  # list of words with bbox
    links: Optional[List[Dict]] = None


@dataclass
class ParsedDocument:
    filename: Optional[str] = None  # source file name
    page_count: int = None
    toc: Optional[List[List]] = None  # e.g. [{'title': 'Intro', 'page': 1}]
    pages: List[PageLayout] = None
    metadata: Optional[Dict] = None
    from_bytes: bool = False  # whether loaded from bytes
    image_dpi: int = 150  # image resolution
    image_format: str = "png"  # 'png' or 'jpg'
    image_path: str = ""  # path to save images
    use_ocr: OCRMode = OCRMode.SELECT_KEEP_OLD  # if beneficial invoke OCR

    def to_markdown(
        self,
        header: bool = True,
        footer: bool = True,
        write_images: bool = False,
        embed_images: bool = False,
        ignore_code: bool = False,
        show_progress: bool = False,
        page_separators: bool = False,
        page_chunks: bool = False,
        **kwargs,
    ) -> Union[str, List[Dict]]:
        """
        Serialize ParsedDocument to markdown text.
        """
        if page_chunks:
            document_output = []
        else:
            document_output = ""

        # ReStyle D4: level for promoting bold-title body boxes — one below the
        # deepest real title/section-header so promotion least disturbs hierarchy.
        _lvls = [
            b.header_level
            for p in self.pages
            for b in p.boxes
            if b.boxclass in ("title", "section-header") and b.header_level
        ]
        _promote_lvl = min(6, max(_lvls) + 1) if _lvls else 2

        if show_progress and len(self.pages) > 5:
            print(f"Generating markdown text...")
            this_iterator = ProgressBar(self.pages)
        else:
            this_iterator = self.pages
        for page in this_iterator:
            md_string = ""
            string_lengths = []
            # Make a mapping: box number -> list item hierarchy level
            list_item_levels = create_list_item_levels(page.boxes)

            for i, box in enumerate(page.boxes):
                clip = pymupdf.IRect(box.x0, box.y0, box.x1, box.y1)
                btype = box.boxclass

                # skip headers/footers if requested
                if btype == "page-header" and header is False:
                    string_lengths.append(len(md_string))
                    continue
                if btype == "page-footer" and footer is False:
                    string_lengths.append(len(md_string))
                    continue

                # pictures and formulas: either write image file or embed
                if btype in ("picture", "formula"):
                    if isinstance(box.image, str):
                        md_string += GRAPHICS_TEXT % box.image + "\n\n"
                    elif isinstance(box.image, bytes):
                        # make a base64 encoded string of the image
                        data = base64.b64encode(box.image).decode()
                        data = f"data:image/{self.image_format};base64," + data
                        md_string += GRAPHICS_TEXT % data + "\n\n"
                    else:
                        md_string += f"\n\n"

                    # output text in image if requested
                    if box.textlines:
                        if btype == "picture":
                            md_string += picture_text_to_md(
                                box.textlines,
                                ignore_code=ignore_code or page.full_ocred,
                                clip=clip,
                            )
                    string_lengths.append(len(md_string))
                    continue
                if btype == "table":
                    table_text = box.table["markdown"]
                    if page.full_ocred:
                        # remove code style if page was OCR'd
                        table_text = table_text.replace("`", "")
                    md_string += table_text + "\n\n"
                    string_lengths.append(len(md_string))
                    continue
                if not hasattr(box, "textlines"):
                    print(f"Warning: box {btype} has no textlines")
                    string_lengths.append(len(md_string))
                    continue
                if btype == "title":
                    md_string += title_to_md(box.header_level, box.textlines)
                    string_lengths.append(len(md_string))
                elif btype == "section-header":
                    md_string += section_hdr_to_md(box.header_level, box.textlines)
                    string_lengths.append(len(md_string))
                elif btype == "list-item":
                    md_string += list_item_to_md(box.textlines, list_item_levels[i])
                    string_lengths.append(len(md_string))
                elif btype == "footnote":
                    md_string += footnote_to_md(box.textlines)
                    string_lengths.append(len(md_string))
                else:  # normal text — or a leading bold title (ReStyle D4)
                    _n = _leading_bold_title_lines(box.textlines)
                    if _n:  # box starts with a bold title line -> promote it
                        md_string += section_hdr_to_md(_promote_lvl, box.textlines[:_n])
                        if _n < len(box.textlines):  # remaining lines are body text
                            md_string += text_to_md(
                                box.textlines[_n:],
                                ignore_code=ignore_code or page.full_ocred,
                            )
                    else:
                        md_string += text_to_md(
                            box.textlines, ignore_code=ignore_code or page.full_ocred
                        )
                    string_lengths.append(len(md_string))
            if page_separators:
                md_string += f"--- end of {page.page_number=} ---\n\n"
            if not page_chunks:
                document_output += md_string
            else:
                chunk = make_page_chunk(self, page, md_string, string_lengths)
                document_output.append(chunk)
        return document_output

    def to_json(self, show_progress=False) -> str:
        # Serialize to JSON
        _ = show_progress
        class LayoutEncoder(json.JSONEncoder):
            def default(self, s):
                if isinstance(s, (bytes, bytearray)):
                    return base64.b64encode(s).decode()
                if isinstance(
                    s,
                    (
                        pymupdf.Rect,
                        pymupdf.Point,
                        pymupdf.Matrix,
                        pymupdf.IRect,
                        pymupdf.Quad,
                    ),
                ):
                    return list(s)
                if hasattr(s, "__dict__"):
                    return s.__dict__
                return super().default(s)

        js = json.dumps(self, cls=LayoutEncoder, ensure_ascii=False)
        return js

    def to_text(
        self,
        header: bool = True,
        footer: bool = True,
        ignore_code: bool = False,
        show_progress: bool = False,
        page_chunks: bool = False,
        table_format: str = "grid",
        table_max_width: int = 100,
        table_min_col_width: int = 10,
        **kwargs,
    ) -> Union[str, List[Dict]]:
        """
        Serialize ParsedDocument to plain text. Optionally omit page headers or footers.
        """
        if table_format not in tabulate.tabulate_formats:
            print(f"Warning: invalid table format '{table_format}', using 'grid'.")
            table_format = "grid"

        if page_chunks:
            document_output = []
        else:
            document_output = ""

        if show_progress and len(self.pages) > 5:
            print(f"Generating plain text ..")
            this_iterator = ProgressBar(self.pages)
        else:
            this_iterator = self.pages
        for page in this_iterator:
            text_string = ""
            string_lengths = []
            list_item_levels = create_list_item_levels(page.boxes)
            for i, box in enumerate(page.boxes):
                clip = pymupdf.IRect(box.x0, box.y0, box.x1, box.y1)
                btype = box.boxclass
                if btype == "page-header" and header is False:
                    string_lengths.append(len(text_string))
                    continue
                if btype == "page-footer" and footer is False:
                    string_lengths.append(len(text_string))
                    continue
                if btype in ("picture", "formula"):
                    if box.textlines and btype == "picture":
                        text_string += picture_text_to_text(
                            box.textlines,
                            ignore_code=ignore_code or page.full_ocred,
                            clip=clip,
                        )
                    string_lengths.append(len(text_string))

                elif btype == "table":
                    wrapped_table = wrap_table_for_tabulate(
                        box.table["extract"],
                        max_width=table_max_width,
                        min_col_width=table_min_col_width,
                    )
                    text_string += (
                        tabulate.tabulate(
                            wrapped_table, disable_numparse=True, tablefmt=table_format
                        )
                        + "\n\n"
                    )
                    string_lengths.append(len(text_string))

                elif btype == "list-item":
                    text_string += list_item_to_text(box.textlines, list_item_levels[i])
                    string_lengths.append(len(text_string))

                elif btype == "footnote":
                    text_string += footnote_to_text(box.textlines)
                    string_lengths.append(len(text_string))

                else:  # handle other cases as normal text
                    text_string += text_to_text(
                        box.textlines, ignore_code=ignore_code or page.full_ocred
                    )
                    string_lengths.append(len(text_string))

            if not page_chunks:
                document_output += text_string
            else:
                chunk = make_page_chunk(self, page, text_string, string_lengths)
                document_output.append(chunk)
        return document_output


def select_ocr_function():
    """Check availability of OCR tools and language data.

    Return the best OCR function available or None.
    """
    tessdata = None
    rapidocr_available = False
    paddleocr_available = False
    try:
        tessdata = pymupdf.get_tessdata()
    except:
        tessdata = None

    try:
        import rapidocr_onnxruntime

        rapidocr_available = True
        paddleocr_available = True
    except:
        pass
    if {tessdata, rapidocr_available, paddleocr_available} == {None, False, False}:
        return None
    if tessdata:
        if rapidocr_available:
            from pymupdf4llm.ocr import rapidtess_api

            print(
                "Using RapidOCR and Tesseract for OCR processing.",
                file=INFO_MESSAGES,
            )
            return rapidtess_api.exec_ocr
        elif paddleocr_available:
            from pymupdf4llm.ocr import paddletess_api

            print(
                "Using PaddleOCR and Tesseract for OCR processing.", file=INFO_MESSAGES
            )
            return paddletess_api.exec_ocr
        else:
            from pymupdf4llm.ocr import tesseract_api

            print("Using Tesseract for OCR processing.", file=INFO_MESSAGES)
            return tesseract_api.exec_ocr
    else:
        if rapidocr_available:
            from pymupdf4llm.ocr import rapidocr_api

            print("Using RapidOCR for OCR processing.", file=INFO_MESSAGES)
            return rapidocr_api.exec_ocr
        elif paddleocr_available:
            from pymupdf4llm.ocr import paddleocr_api

            print("Using PaddleOCR for OCR processing.", file=INFO_MESSAGES)
            return paddleocr_api.exec_ocr


def update_header_tags(pages, header_fontsizes):
    """Update title/section-header boxes with HTML header tags."""
    # List of up to 6 integer font sizes in descending order
    header_fontsizes = sorted(header_fontsizes, reverse=True)[:6]
    for page in pages:
        for box in page.boxes:
            if box.boxclass in ("title", "section-header"):
                if box.max_fontsize >= header_fontsizes[-1]:
                    box.header_level = header_fontsizes.index(box.max_fontsize) + 1
                else:
                    box.header_level = 6


def make_ocr_decision(page, use_ocr):
    """Decide whether to OCR a page.

    Returns a tuple (needs_ocr, ocr_spans) where needs_ocr is a boolean
    indicating whether OCR is needed, and ocr_spans is the number of
    existing OCR spans on the page (if any).
    """
    # OCR not desired at all
    if use_ocr == OCRMode.NEVER:
        return False, 0

    page_analysis = utils.analyze_page(page)

    needs_ocr = page_analysis.get("needs_ocr", False)
    # may be > 0 even if needs_ocr is False:
    ocr_spans = page_analysis.get("ocr_spans", 0)

    if ocr_spans and use_ocr in (OCRMode.FORCE_KEEP_OLD, OCRMode.SELECT_KEEP_OLD):
        # return False if old OCR should be kept
        return False, ocr_spans

    return needs_ocr, 0


def parse_document(
    doc,
    filename="",
    image_dpi=150,
    ocr_dpi=300,
    image_format="png",
    image_path="",
    pages=None,
    show_progress=False,
    embed_images=False,
    write_images=False,
    force_text=False,
    use_ocr=OCRMode.SELECT_KEEP_OLD,
    force_ocr=False,
    ocr_language="eng",
    ocr_function=None,
) -> ParsedDocument:
    if isinstance(doc, pymupdf.Document):
        mydoc = doc
    else:
        mydoc = pymupdf.open(doc)

    if mydoc.metadata["format"] == "Image":
        # Re-open as PDF to ensure we can successfully OCR the image.
        data = mydoc.convert_to_pdf()
        mydoc.close()
        mydoc = pymupdf.open(stream=data)

    if mydoc.is_pdf:
        # Remove StructTreeRoot to avoid possible performance degradation.
        # This package will not use the structure tree anyway.
        mypdf = pymupdf._as_pdf_document(mydoc)
        root = mupdf.pdf_dict_get(mupdf.pdf_trailer(mypdf), pymupdf.PDF_NAME("Root"))
        root.pdf_dict_del(pymupdf.PDF_NAME("StructTreeRoot"))
    else:
        use_ocr = OCRMode.NEVER
        if force_ocr:
            print(
                "Warning: OCR disabled because document is no PDF.",
                file=INFO_MESSAGES,
            )
        force_ocr = False

    if embed_images and write_images:
        raise ValueError("Cannot both embed and write images.")

    # collect font sizes of title and section_header
    header_fontsizes = set()

    document = ParsedDocument()
    document.filename = mydoc.name if mydoc.name else filename
    document.toc = mydoc.get_toc(simple=True)
    document.page_count = mydoc.page_count
    document.metadata = mydoc.metadata
    document.form_fields = utils.extract_form_fields_with_pages(mydoc)
    document.image_dpi = image_dpi
    document.image_format = image_format
    document.image_path = image_path
    document.pages = []
    document.force_text = force_text
    document.embed_images = embed_images
    document.write_images = write_images

    if force_ocr:
        use_ocr = OCRMode.FORCE_KEEP_OLD

    if use_ocr:
        if callable(ocr_function):
            document.use_ocr = use_ocr
        else:
            ocr_function = select_ocr_function()
            if callable(ocr_function):
                document.use_ocr = use_ocr
            else:
                document.use_ocr = OCRMode.NEVER
    else:
        document.use_ocr = OCRMode.NEVER

    if not callable(ocr_function):
        if document.use_ocr in (
            OCRMode.FORCE_DROP_OLD,
            OCRMode.FORCE_KEEP_OLD,
        ):
            raise ValueError("Force OCR is True but no OCR engine available.")
        if document.use_ocr != OCRMode.NEVER:
            print("Warning: No OCR engine available, OCR disabled.")
            document.use_ocr = OCRMode.NEVER

    if pages is None:
        page_filter = range(mydoc.page_count)
    elif isinstance(pages, int):
        while pages < 0:
            pages += mydoc.page_count
        page_filter = [pages]
    elif not hasattr(pages, "__getitem__"):
        raise ValueError("'pages' parameter must be an int, or a sequence of ints")
    else:
        page_filter = sorted(set(pages))

    if (
        not all(isinstance(p, int) for p in page_filter)
        or page_filter[-1] >= mydoc.page_count
    ):
        raise ValueError(
            f"'pages' parameter must be None, int, or a sequence of ints < {mydoc.page_count}."
        )

    if show_progress and len(page_filter) >= 5:
        print(f"Parsing {len(page_filter)} pages of '{document.filename}'...")
        page_filter = ProgressBar(page_filter)

    for pno in page_filter:
        page = mydoc.load_page(pno)
        page.remove_rotation()
        page_full_ocred = False
        PAGE_ANALYSIS = {}
        OCR_SPANS = 0
        needs_ocr, OCR_SPANS = make_ocr_decision(page, document.use_ocr)

        if needs_ocr:
            # execute OCR for the page replacing any previous OCR spans
            ocr_function(
                page,
                dpi=ocr_dpi,
                language=ocr_language,
                keep_ocr_text=False,
            )
            print(f"OCR on {page.number=}/{page.number+1}.", file=INFO_MESSAGES)

        textpage = page.get_textpage(flags=FLAGS, clip=pymupdf.INFINITE_RECT())
        blocks = textpage.extractDICT()["blocks"]

        # Execute the Layout module AFTER any OCR
        page.get_layout(return_raw=True)

        # Determine if any tables are present. If False, we skip any table-related efforts.
        tables_exist = any(
            b for b in page.layout_information if b["class_name"] == "table"
        )
        hlines = _thin_hlines(page)
        pixel_hlines = _pixel_hlines(page, ocr_dpi) if _is_ocr_page(blocks) else []

        # Dictionary with details for all tables. Key is the bounding box
        # tuple, value is the original Layout info per table.
        table_infos = {}

        new_layout_info = []  # will contain Layout boxes in non-"raw" format
        for b in page.layout_information:
            bbox = tuple(b["group_bbox"] + [b["class_name"]])
            new_layout_info.append(bbox)

            # store table info for later use in table extraction
            # we use the bounding box tuple as key for later matching
            if b["class_name"] == "table":
                key = tuple(pymupdf.IRect(b["group_bbox"]))
                table_infos[key] = b

        page.layout_information = new_layout_info
        if not OCR_SPANS:  # some cleaning if no old OCR spans
            utils.clean_pictures(page, blocks)
            utils.add_image_orphans(page, blocks)

        # execute our own reading order function
        page.layout_information = utils.find_reading_order(
            page.rect, blocks, page.layout_information
        )
        fulltext = [b for b in blocks if b["type"] == 0]
        if tables_exist or hlines or pixel_hlines:
            raw_blocks = [
                b for b in textpage.extractRAWDICT()["blocks"] if b["type"] == 0
            ]
        else:
            raw_blocks = None
        if tables_exist:
            table_blocks = raw_blocks
        else:
            table_blocks = None

        words = []  # not yet activated
        links = [l for l in page.get_links() if l["kind"] == pymupdf.LINK_URI]
        pagelayout = PageLayout(
            page_number=page.number + 1,
            width=page.rect.width,
            height=page.rect.height,
            boxes=[],
            full_ocred=page_full_ocred,
            fulltext=fulltext,
            words=words,
            links=links,
        )
        for box in page.layout_information:
            layoutbox = LayoutBox(*box)
            clip = pymupdf.Rect(box[:4])

            if layoutbox.boxclass in ("picture", "formula"):
                if document.embed_images or document.write_images:
                    pix = page.get_pixmap(clip=clip, dpi=document.image_dpi)
                    irect = pymupdf.IRect(pix.irect)  # guard against empty images
                    if not irect.is_empty:
                        if document.embed_images:
                            layoutbox.image = pix.tobytes(document.image_format)
                        elif document.write_images:
                            img_filename = f"{document.filename}-{page.number+1:04d}-{len(pagelayout.boxes):02d}.{document.image_format}"
                            md_filename, save_img_filename = utils.md_path(
                                document.image_path, img_filename
                            )
                            layoutbox.image = md_filename
                            pix.save(save_img_filename)
                    else:
                        layoutbox.image = None
                else:
                    layoutbox.image = None
                if layoutbox.boxclass == "picture" and document.force_text:
                    # extract any text within the image box
                    layoutbox.textlines = [
                        {"bbox": l[0], "spans": l[1]}
                        for l in get_raw_lines(
                            textpage=None,
                            blocks=pagelayout.fulltext,
                            clip=clip,
                            ignore_invisible=False,
                            only_horizontal=False,
                        )
                    ]

            elif layoutbox.boxclass == "table":
                search_key = (layoutbox.x0, layoutbox.y0, layoutbox.x1, layoutbox.y1)

                # Because of intermediate processing, the bbox might not match
                # the original exactly. So we need to take the best fit.
                key = max(table_infos.keys(), key=lambda k: utils.iou(k, search_key))

                tab_dict = table_infos.get(key)
                tab_details = get_table_details(tab_dict, table_blocks)

                layoutbox.table = {
                    "bbox": list(tab_details.bbox),
                    "row_count": tab_details.row_count,
                    "col_count": tab_details.col_count,
                    "cells": tab_details.cells,
                    "extract": tab_details.extract,
                    "markdown": tab_details.markdown,
                }

            else:
                # Handle text-like box classes:
                # Extract text line information within the box.
                # Each line is represented as its bbox and a list of spans.
                layoutbox.textlines = [
                    {"bbox": l[0], "spans": l[1]}
                    for l in get_raw_lines(
                        textpage=None,
                        blocks=pagelayout.fulltext,
                        clip=clip,
                        ignore_invisible=False,
                    )
                ]
                if layoutbox.boxclass not in (
                    "title",
                    "section-header",
                    "page-header",
                    "page-footer",
                ):
                    _apply_decorators(layoutbox.textlines, raw_blocks, hlines)
                _apply_decorators(
                    layoutbox.textlines,
                    raw_blocks,
                    pixel_hlines,
                    underline_only=True,
                )
                # For each title/section_header compute and store the maximum
                # font size, to be used as a signal for header "#" prefix
                if layoutbox.boxclass in ("title", "section-header"):
                    max_fontsize = 0
                    for line in layoutbox.textlines:
                        for span in line["spans"]:
                            size = round(span["size"])
                            max_fontsize = max(max_fontsize, size)
                    header_fontsizes.add(max_fontsize)
                    layoutbox.max_fontsize = max_fontsize

            pagelayout.boxes.append(layoutbox)
        document.pages.append(pagelayout)
    if mydoc != doc:
        mydoc.close()
    msg_text = INFO_MESSAGES.getvalue()
    if msg_text:
        pymupdf.message("=== Document parser messages ===")
        pymupdf.message(msg_text)
    INFO_MESSAGES.truncate(0)  # empty the file-like object
    # Update title/section-header boxes with html header tags
    update_header_tags(document.pages, header_fontsizes)
    return document


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    filename = sys.argv[1]
    pdoc = parse_document(filename)
    # Path(filename).with_suffix(".json").write_text(pdoc.to_json())
    # Path(filename).with_suffix(".txt").write_text(pdoc.to_text(footer=False))
    md = pdoc.to_markdown(write_images=False, header=False, footer=False)
    Path(filename).with_suffix(".md").write_text(md)
