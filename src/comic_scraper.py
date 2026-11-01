"""Read a comic series' chapter list and page images, on the fast scraper's stack.

This is the stage ``comic_reader.py`` cannot do: getting the pictures at all.
It saves them as ``c_<storage-chapter>_<page>.<ext>`` -- the naming the existing
downloads already use and that ``comic_reader.group_chapters`` parses -- so the
rest of the pipeline is unchanged.

What it reuses from the novel scraper, and why each one matters:

* ``scraper_parse.looks_like_challenge`` -- a Cloudflare page carries no reader
  images, which without this test is indistinguishable from the end of a series.
  Ignoring it would truncate a download silently and report success.
* ``scraper_fast._with_retries`` / ``post_transport`` -- the ``curl_cffi`` TLS
  impersonation the site's bot check requires, plus timeout/5xx retries and 429
  backoff. There is one retry policy for the whole project, not one per verb.
* ``comic_reader.load_config`` -- the same ``comic_config.json`` the reading
  stage uses, so the comics directory cannot disagree between the two.

Two things about this site were measured, not assumed:

* The reader's chapter N is stored on the CDN as ``chapter_<N-1>``. Chapter 1
  is ``chapter_0/c_0_1.jpg``; chapter 2 is ``chapter_1/c_1_1.jpg``. The local
  filename keeps the *storage* numbering, because that is what the existing
  downloads use and what ``comic_reader`` groups on -- so a chapter can look off
  by one in a filename and still be correct.
* The chapter list is *not* JavaScript-only, as it first appeared: a plain GET
  of the AJAX endpoint answers ``0``, but a POST carrying
  ``X-Requested-With: XMLHttpRequest`` returns all ~300 chapters in one
  response. Walking the next-chapter links was a workaround for a request we
  were making wrong, and it is gone.

Only ``comic-site.test`` and its subdomains are fetched, over HTTP(S) only. The
pages inject obfuscated scripts that redirect to ad hosts, so an off-site image
URL is refused rather than followed, and no JavaScript is ever executed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from comic_reader import load_config  # noqa: F401 - re-exported; one config reader
from scraper_fast import _with_retries, post_transport  # noqa: F401 - re-exported
from scraper_parse import looks_like_challenge

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "comic_config.json"

SITE = "https://comic-site.test"

#: Magic bytes, so an error page served with a ``.jpg`` name is not kept.
IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF", b"GIF8")

#: The extensions a page image may carry. Anything else is refused rather than
#: guessed, because a wrong extension is a file the reader cannot group.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})

#: Spellings of the same format, folded so one page has one name.
_EXTENSION_ALIASES = {".jpeg": ".jpg"}


class ScrapeError(Exception):
    """A refusal that retrying will not fix."""


def safe_image_url(url: str) -> bool:
    """Whether ``url`` is a comic image on this site that may be fetched.

    Rejects foreign hosts outright. ``comic-site.test.evil.example`` and
    ``http://comic-site.test@evil.example`` are different hosts despite reading
    like ours, and ``data:`` URLs carry no host at all.
    """

    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    return host == "comic-site.test" or host.endswith(".comic-site.test")


def image_ext(url: str) -> str:
    """The extension to save ``url`` under, normalised to a known one.

    ``.jpeg`` folds to ``.jpg`` so one page has one name. An unknown or missing
    extension raises rather than being guessed: a page saved as the wrong type
    is silently skipped later by ``comic_reader``.
    """

    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise ScrapeError(
            f"cannot tell the image type of {url!r}; expected one of "
            f"{sorted(IMAGE_EXTENSIONS)}"
        )
    return _EXTENSION_ALIASES.get(suffix, suffix)


# -- the chapter list -------------------------------------------------------


def chapter_url(series_url: str, slug: str) -> str:
    """The full URL of one chapter, from the series URL and the site's slug."""

    return series_url.rstrip("/") + "/" + slug.strip("/") + "/"


def fetch_chapter_list(series_url: str) -> list[tuple[str, str]]:
    """Every chapter of the series as ``(slug, label)``, newest first.

    One AJAX POST returns the whole list (~300 chapters), so no walking is
    needed. A non-200 or a Cloudflare interstitial raises: an empty or partial
    list would otherwise read as a complete one.
    """

    url = series_url.rstrip("/") + "/ajax/chapters/"
    status, body = post_transport(url, {"X-Requested-With": "XMLHttpRequest"}, {})
    if looks_like_challenge(body):
        raise ScrapeError(
            f"chapter list: Cloudflare challenge instead of the list at {url}. "
            "Wait a minute and retry; if it persists this site needs a browser "
            "harvest (scraper_session.harvest_session)."
        )
    if status != 200:
        raise ScrapeError(
            f"chapter list: HTTP {status} at {url}"
            + (", rate limited by Cloudflare" if status == 429 else "")
        )

    soup = BeautifulSoup(body, "html.parser")
    chapters: list[tuple[str, str]] = []
    for anchor in soup.select("a[href*=chapter]"):
        href = anchor.get("href") or ""
        slug = href.rstrip("/").rsplit("/", 1)[-1]
        if slug:
            chapters.append((slug, anchor.get_text(" ", strip=True)))
    return chapters


# -- the pages --------------------------------------------------------------


def page_urls(html: str) -> list[str]:
    """The reader images on a chapter page, in page order.

    Real chapter images are lazy-loaded: ``src`` is a placeholder loader and the
    address to fetch is in ``data-src``. Reading ``src`` downloads the loader
    GIF. Off-site URLs are dropped here as well as at fetch time, so nothing
    downstream ever holds one.
    """

    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    for img in soup.select("img.wp-manga-chapter-img"):
        url = (img.get("data-src") or img.get("src") or "").strip()
        if url and safe_image_url(url):
            urls.append(url)
    return urls


# -- network ---------------------------------------------------------------


def _looks_like_image(payload: bytes) -> bool:
    return any(payload.startswith(magic) for magic in IMAGE_MAGIC)


def open_session():
    """A ``curl_cffi`` session presenting a real browser's TLS fingerprint.

    The chapter pages sit behind a bot check that refuses a plain HTTP client
    with a 403 challenge. JavaScript is never executed, which is the point: the
    pages carry obfuscated redirect scripts.
    """

    try:
        from curl_cffi import requests
    except ImportError:
        raise SystemExit(
            "comic scraping needs the 'scraper' extra:\n"
            "  uv sync --extra comic --extra scraper"
        ) from None
    return requests.Session(impersonate="chrome")


# -- the run ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Not finished: the download loop is the next stage of this rewrite.

    Deliberately a stub rather than the old CLI. The old one drove the
    next-link walk that this change deletes, so keeping it would mean keeping
    ``Run``/``scrape_chapters`` alive for nothing. Arguments are not parsed,
    because advertising options the stub ignores is worse than refusing to run;
    a later change gives this a real CLI.
    """

    del argv
    print(
        "The comic download stage is being rebuilt and is not runnable yet: "
        "the chapter list now comes from one request, but downloading the "
        "pages is the next change."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
