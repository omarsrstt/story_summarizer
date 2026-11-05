"""Comic chapter -> text, so the existing summariser can read it.

Comics need a reading stage the novels do not: images have to become text
before ``summarize.py`` can recap them. This script does only that, writing
``novels/<slug>/chapter_NNNN.txt`` -- the exact layout the summariser already
looks for -- so recap, ``--group-size``, ``--digest`` and ``--all`` all work
unchanged.

No upscaling: measured on real pages, PP-OCR reads this artwork at native
resolution, 1x and 2x output were character-identical, and 2x cost 31% more
time.

    uv run python src/comic_reader.py --series TBATE --comics comics/TBATE
    uv run python src/comic_reader.py --series TBATE --only 1,2 --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from comic_pages import (
    dedupe_seam_blocks,
    format_transcript,
    is_bubble_like,
    slice_scroll,
)
from comic_text import clean_ocr_text, is_meaningful, resegment

CHAPTER_PATTERNS = (
    re.compile(r"^c_(\d+)_(\d+)\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE),
    re.compile(r"^chapter[_-]?(\d+)[_-](\d+)\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE),
)

PAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def page_filename(ordinal: int, page: int, ext: str) -> str:
    """The file name for one page of chapter ``ordinal``.

    Zero-padding both numbers keeps lexical order equal to reading order, which
    is what lets ``group_chapters`` sort by name alone.
    """

    normalised = ext.strip().lower()
    if not normalised.startswith("."):
        normalised = "." + normalised
    if normalised not in PAGE_EXTENSIONS:
        raise ValueError(
            f"unsupported page extension {ext!r}; expected one of "
            f"{sorted(PAGE_EXTENSIONS)}"
        )
    return f"c_{ordinal:06d}_{page:03d}{normalised}"


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "comic_config.json"

#: The local half of the comic settings: which site a series lives on.
#:
#: Split out from ``comic_config.json`` for the same reason the novel scraper's
#: ``scraper_config.json`` is gitignored -- the tracked config says how to read a
#: page, this one says where to find it. Nothing else lives here, so the tracked
#: file stays publishable and the glossary stays versioned.
SITE_CONFIG_PATH = Path(__file__).resolve().parent / "comic_site.json"

#: Comics are read into their own namespace. A comic and a novel can share a
#: name ("TBATE" is both a comic here and a plausible novel title), and both
#: resolve to ``<root>/<slug>``, so an unprefixed slug would let one overwrite
#: the other's transcripts and summaries.
COMIC_PREFIX = "comic-"


def comic_slug(series: str) -> str:
    """Map a series name to the slug used under ``novels/`` and ``summaries/``.

    Idempotent, so re-running with the slug the script printed is harmless, and
    hostile to path separators: the slug becomes a directory name directly.
    """

    name = series.strip()
    if name.lower().startswith(COMIC_PREFIX):
        name = name[len(COMIC_PREFIX):]
    name = name.strip()
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"Not usable as a series name: {series!r}")
    return COMIC_PREFIX + name


def load_config(path: Path | None = None) -> dict:
    """Read ``comic_config.json``, with the local site settings merged in.

    The tracked config holds everything that is the same for every user (glossary,
    tile geometry, output dirs); ``comic_site.json`` holds the two keys that name
    the site and is gitignored, mirroring ``scraper_config.json``.

    The merge is shallow and the site file wins, so an existing single-file config
    that still carries ``comic_series_url``/``comic_site_host`` keeps working --
    the switch to two files needs no migration.

    A missing site file is not an error: a reader-only run needs no host, and the
    scraper reports a missing host when it actually tries to fetch. An unreadable
    or malformed *either* file is still an error here -- silently dropping the
    glossary over a typo would degrade every transcript without a word.
    """

    config_path = path or CONFIG_PATH
    config = _read_json(config_path)
    site_path = config_path.with_name(SITE_CONFIG_PATH.name)
    if site_path.is_file():
        config.update(_read_json(site_path))
    return config


def _read_json(path: Path) -> dict:
    """The JSON object at ``path``, or ``{}`` when the file is absent."""

    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def group_chapters(image_dir: Path) -> dict[int, list[Path]]:
    """Map chapter number to its page images, in page order.

    Two layouts occur in practice and both must be supported: one image per
    reader page (``c_1_7.jpg``), and a whole chapter stitched into a single
    continuous scroll (``c_0_1.jpg``).
    """

    chapters: dict[int, list[tuple[int, Path]]] = {}
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        for pattern in CHAPTER_PATTERNS:
            match = pattern.match(path.name)
            if match:
                chapter, page = int(match.group(1)), int(match.group(2))
                chapters.setdefault(chapter, []).append((page, path))
                break

    if not chapters:
        raise SystemExit(f"No comic pages found in {image_dir}")

    return {
        chapter: [path for _, path in sorted(pages)]
        for chapter, pages in sorted(chapters.items())
    }


def read_page(
    image, page_number: int, engine, glossary: list[str], top: int = 0
) -> list[dict]:
    """Detect and read the text on one page image (or one tile of a scroll).

    ``top`` is the tile's absolute pixel row, so boxes come back in the
    original image's coordinates and a bubble seen by two tiles lands in the
    same place both times.
    """

    result, _ = engine(image)
    blocks = []
    for item in result or []:
        points, text, score = item[0], item[1], item[2]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        box = (
            int(min(xs)),
            int(min(ys)) + top,
            int(max(xs)),
            int(max(ys)) + top,
        )

        cleaned = clean_ocr_text(text)
        if not cleaned or not is_meaningful(cleaned, known=glossary):
            continue
        blocks.append(
            {
                "page": page_number,
                "box": box,
                "text": resegment(cleaned, known=glossary),
                "confidence": round(float(score), 4),
                "kind": "dialogue" if is_bubble_like(image, box, top) else "placed",
            }
        )
    return blocks


def read_chapter(
    paths: list[Path],
    engine,
    glossary: list[str],
    tile_height: int = 1200,
    overlap: int = 120,
) -> tuple[list[dict], float]:
    """Read every page of a chapter, slicing a continuous scroll first."""

    from PIL import Image

    blocks: list[dict] = []
    started = time.perf_counter()

    for page_number, path in enumerate(paths, start=1):
        with Image.open(path) as image:
            page = image.convert("RGB")

        # A scroll taller than one detector tile is cut up; each tile is the
        # same logical page, so they share a page number and duplicates at the
        # seams are dropped afterwards.
        for top, tile in slice_scroll(page, tile_height=tile_height, overlap=overlap):
            blocks.extend(read_page(tile, page_number, engine, glossary, top))

    return dedupe_seam_blocks(blocks), time.perf_counter() - started


def write_jsonl(path: Path, blocks: list[dict]) -> None:
    """Keep the structured transcript next to the flat text, for later passes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for block in blocks:
            handle.write(json.dumps(block, ensure_ascii=False) + "\n")


def parse_only(text: str | None, available: list[int]) -> list[int]:
    """Parse ``--only`` with the same grammar as the summariser's ``-c``."""

    if not text or text.strip().lower() == "all":
        return available

    selected: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    return [chapter for chapter in available if chapter in selected]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Turn comic page images into chapter text files."
    )
    parser.add_argument("--series", required=True, help="Slug, e.g. TBATE")
    parser.add_argument("--comics", required=True,
                        help="Directory holding the page images")
    parser.add_argument("--only", default="all",
                        help="Chapters to read: 'all', '1', '1-3'")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-read chapters that already have a transcript")
    parser.add_argument("--glossary", default=None,
                        help="Extra JSON list of names/terms, on top of the config")
    args = parser.parse_args(argv)

    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        raise SystemExit(
            "comic reading needs the 'comic' extra:\n"
            "  uv sync --extra comic"
        ) from None

    image_dir = Path(args.comics)
    if not image_dir.is_dir():
        raise SystemExit(f"No such directory: {image_dir}")

    config = load_config()
    glossary: list[str] = list(config.get("glossary") or [])
    if args.glossary:
        glossary += json.loads(Path(args.glossary).read_text(encoding="utf-8"))

    chapters = group_chapters(image_dir)
    selected = parse_only(args.only, sorted(chapters))

    try:
        slug = comic_slug(args.series)
    except ValueError as error:
        raise SystemExit(str(error)) from None

    novels_dir = REPO_ROOT / config.get("novels_dir", "novels") / slug
    source_dir = REPO_ROOT / config.get("transcripts_dir", "transcripts") / slug
    novels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Series:  {args.series}")
    print(f"Source:  {image_dir}")
    print(f"Output:  {novels_dir}")
    print(f"Chapters: {len(selected)} of {len(chapters)}")

    todo = [
        chapter for chapter in selected
        if args.overwrite
        or not (novels_dir / f"chapter_{chapter:05d}.txt").is_file()
    ]
    if not todo:
        print("\nNothing to do: every selected chapter already has a transcript. "
              "Pass --overwrite to re-read.")
        return 0

    print(f"To read: {', '.join(str(c) for c in todo)}")

    engine = RapidOCR()

    tile_height = int(config.get("tile_height", 1200))
    overlap = int(config.get("tile_overlap", 120))

    for chapter in todo:
        paths = chapters[chapter]
        blocks, elapsed = read_chapter(
            paths, engine, glossary, tile_height=tile_height, overlap=overlap
        )

        title = f"{args.series} chapter {chapter}"
        transcript = format_transcript(title, blocks)
        if not transcript.strip() or len(transcript.split()) < 5:
            print(f"  chapter {chapter}: no text detected, skipping")
            continue

        target = novels_dir / f"chapter_{chapter:05d}.txt"
        target.write_text(transcript, encoding="utf-8", newline="\n")
        write_jsonl(source_dir / f"chapter_{chapter:05d}.jsonl", blocks)

        words = len(transcript.split())
        print(
            f"  chapter {chapter}: {len(paths)} image(s), {len(blocks)} block(s), "
            f"{words} words in {elapsed:.1f}s -> {target.name}"
        )

    # The reader owns the slug, so print the next command's argument rather than
    # leaving the caller to reconstruct "comic-" by hand.
    print(f'\nSummarise with:  uv run python src/summarize.py -n "{slug}"')

    return 0


if __name__ == "__main__":
    sys.exit(main())
