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

from bs4 import BeautifulSoup

_NUMBER_RE = re.compile(r"chapter\s+(\d+)", re.IGNORECASE)

# Substring of a Cloudflare interstitial body; shared so the challenge check and
# the plausibility guard cannot drift apart.
CHALLENGE_MARKER = "just a moment"


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


def extract_chapter(
    html: str, content_selector: str, title_selector: str
) -> tuple[str, str]:
    """Return ``(title, text)`` pulled from a chapter page.

    Missing nodes yield empty strings rather than raising, so callers can treat
    a challenge/partial page as "no content" instead of crashing.
    """

    soup = BeautifulSoup(html, "html.parser")
    title_node = soup.select_one(title_selector)
    content_node = soup.select_one(content_selector)
    title = title_node.get_text().strip() if title_node else ""
    if content_node:
        # Drop ad/script wrappers, as the legacy Selenium scraper does.
        for element in content_node.find_all(["script", "style", "div", "aside"]):
            element.decompose()
        paragraphs = content_node.find_all("p")
        # The site wraps its paragraphs in an outer <p> (<p><p>..</p></p>). A
        # browser auto-closes that outer tag, but html.parser nests it, which
        # would emit every paragraph twice. Skip any <p> that contains another.
        paragraphs = [p for p in paragraphs if not p.find("p")]
        text = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
        if not text:
            # Some layouts separate lines with <br> rather than <p>; fall back to
            # the node's own text so such chapters are not saved empty.
            text = content_node.get_text(strip=True)
    else:
        text = ""
    return title, text


def looks_like_chapter(text: str, min_chars: int = 400) -> bool:
    """Guard against saving a challenge page or a truncated body as a chapter."""

    if len(text) < min_chars:
        return False
    if CHALLENGE_MARKER in text.lower():
        return False
    return True

