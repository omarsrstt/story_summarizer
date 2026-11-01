"""Locate speech-bubble text on a comic page and turn it into transcript blocks.

Comic pages arrive in two shapes: discrete pages (one image per reader page) and
a single continuous scroll. The detector only handles regions roughly the size
of its input, so the scroll has to be sliced before anything downstream runs.

Measured on a real chapter: PP-OCR reads this artwork at native resolution with
no upscaling -- glyphs are ~23px, 1x and 2x output were character-identical,
and 2x cost 31% more time. So this module never resamples.
"""

from __future__ import annotations

import re

import numpy as np

# A bubble-sized crop. Pages are ~700x1900 and bubbles ~120x80, so this is
# comfortably larger while still far smaller than a whole page.
_MIN_REGION_WIDTH = 60
_MIN_REGION_HEIGHT = 18


def slice_scroll(image, tile_height: int = 1200, overlap: int = 120) -> list[tuple[int, object]]:
    """Cut a continuous scroll into overlapping tiles.

    Returns ``(top, tile)`` pairs, ``top`` being the tile's absolute pixel row
    in the original image, so detections can be mapped back to one coordinate
    space. Overlap keeps a bubble that straddles a seam whole in at least one
    tile, at the cost of detecting some text twice.
    """

    width, height = image.size
    if height <= tile_height:
        return [(0, image.copy())]

    step = tile_height - overlap
    tiles = []
    for top in range(0, height, step):
        bottom = min(top + tile_height, height)
        tiles.append((top, image.crop((0, top, width, bottom))))
        if bottom >= height:
            break
    return tiles


# How far apart two readings of the same text may sit and still be one bubble
# read twice by neighbouring tiles.
_SEAM_DISTANCE = 40


def dedupe_seam_blocks(blocks: list[dict]) -> list[dict]:
    """Drop a bubble that two overlapping tiles each read.

    Duplicates are only possible at a seam, so a repeat is matched on page,
    text and position: the same words far apart down a page are content, not
    an artefact.
    """

    kept: list[dict] = []
    for block in blocks:
        _, y0, _, y1 = block["box"]
        duplicate = any(
            other.get("page") == block.get("page")
            and other["text"] == block["text"]
            and abs(other["box"][1] - y0) <= _SEAM_DISTANCE
            and abs(other["box"][3] - y1) <= _SEAM_DISTANCE
            for other in kept
        )
        if not duplicate:
            kept.append(block)
    return kept


def ink_mask(image, threshold: int = 200) -> np.ndarray:
    """Boolean mask of dark pixels, i.e. the lettering."""

    gray = np.asarray(image.convert("L"))
    return gray < threshold


def is_bubble_like(image, box: tuple[int, int, int, int], top: int = 0) -> bool:
    """Whether a crop looks like lettering on a pale bubble rather than artwork.

    Speech bubbles are pale and mostly empty apart from the glyphs; text placed
    directly on artwork sits on busy, coloured pixels. ``top`` maps a box in the
    original image's coordinates onto the tile actually held in ``image``.
    """

    x0, y0, x1, y1 = box
    crop = image.crop((x0, y0 - top, x1, y1 - top))
    if crop.size[0] < _MIN_REGION_WIDTH or crop.size[1] < _MIN_REGION_HEIGHT:
        return False
    gray = np.asarray(crop.convert("L")).astype(np.float32)
    pale = float((gray > 200).mean())
    return pale >= 0.45


def reading_order(blocks: list[dict]) -> list[dict]:
    """Sort blocks by page, then down the page, left-to-right within a row.

    Webtoon pages are read straight down, so vertical position dominates, but
    two bubbles side by side sit a few pixels apart vertically and must still
    be read left to right. Blocks whose vertical centres overlap a row band
    are therefore ordered by x. Page comes first so a transcript never returns
    to an earlier page.
    """

    ordered = sorted(
        blocks, key=lambda b: (b.get("page") or 0, b["box"][1], b["box"][0])
    )
    result: list[dict] = []
    page_rows: list[dict] = []
    for block in ordered:
        page = block.get("page") or 0
        if page_rows and (page_rows[0].get("page") or 0) == page:
            band_top = min(b["box"][1] for b in page_rows)
            band_bottom = max(b["box"][3] for b in page_rows)
            _, y0, _, y1 = block["box"]
            overlaps = y0 < band_bottom and y1 > band_top
        else:
            overlaps = False

        if overlaps:
            page_rows.append(block)
        else:
            result.extend(sorted(page_rows, key=lambda b: b["box"][0]))
            page_rows = [block]

    result.extend(sorted(page_rows, key=lambda b: b["box"][0]))
    return result


_CLEAN_LINE = re.compile(r"\s+")


def render_block_text(text: str) -> str:
    """Collapse whitespace so a block occupies one transcript line."""

    return _CLEAN_LINE.sub(" ", text).strip()


def format_transcript(title: str, blocks: list[dict]) -> str:
    """Render ordered text blocks as the chapter file the summariser reads.

    ``summarize.py`` takes ``chapter_NNNN.txt`` verbatim into the prompt, so the
    page and panel markers are plain text the model can follow rather than
    anything it has to parse.
    """

    lines = [f"# {title}", ""]
    current_page = None
    for block in reading_order(blocks):
        text = render_block_text(block.get("text", ""))
        if not text:
            continue
        page = block.get("page")
        if page != current_page:
            lines.append(f"--- page {page} ---")
            current_page = page
        lines.append(text)
    return "\n".join(lines).strip() + "\n"
