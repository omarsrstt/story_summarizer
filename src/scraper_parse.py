"""Pure parsing helpers for the fast scraper.

This module deliberately has no browser or network imports so its functions can
be unit-tested in isolation.

The chapter list is harvested from the site's ``select.select-chapter`` element
as ``(value, text)`` option pairs, where ``value`` is the chapter's URL slug
(e.g. ``chapter-2`` or the slugged ``chapter-530-end-story``) and ``text`` is the
human-readable title (e.g. ``Chapter 1: The Light At The End of The Tunnel``).

Local chapter numbers come from the **title**, never from an arithmetic offset
against the slug: the site's offset is not constant and the prologue entry
(``c-1: Prologue: Start``) is not one of the numbered corpus chapters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBER_RE = re.compile(r"chapter\s+(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ChapterRef:
    """A single chapter: its number, URL slug, and title."""

    number: int
    slug: str
    title: str


def parse_chapter_number(title: str) -> int | None:
    """Return the chapter number in ``title``, or ``None`` if there is none.

    >>> parse_chapter_number("Chapter 1: The Light At The End of The Tunnel")
    1
    >>> parse_chapter_number("c-1: Prologue: Start") is None
    True
    """
    match = _NUMBER_RE.search(title)
    return int(match.group(1)) if match else None


def build_chapter_map(options: list[tuple[str, str]]) -> dict[int, ChapterRef]:
    """Turn harvested ``(slug, title)`` option pairs into ``{number: ChapterRef}``.

    Entries whose title carries no chapter number (the prologue) are skipped, and
    a number that appears more than once keeps its first occurrence.
    """

    chapters: dict[int, ChapterRef] = {}
    for slug, title in options:
        number = parse_chapter_number(title)
        if number is None or number in chapters:
            continue
        chapters[number] = ChapterRef(number, slug, title)
    return chapters
