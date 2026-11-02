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
  stage uses, so the comics directory cannot disagree between the two. Imported
  inside the function, not at module level: the reader pulls in the OCR extra's
  numpy/wordninja, and a download-only install must not need them to import this.
  The same deferral covers ``comic_reader.page_filename`` and the page extension
  list, used by the download stage below. Pages are named by that one rule, but
  the import stays local, so this module remains importable on an install that
  has only the ``scraper`` extra.

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
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from comic_manifest import ChapterRecord
from scraper_fast import (  # noqa: F401 - re-exported
    _with_retries,
    bytes_transport,
    default_transport,
    post_transport,
)
from scraper_parse import looks_like_challenge

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "comic_config.json"

SITE = "https://comic-site.test"

#: The extensions a page image may carry. Anything else is refused rather than
#: guessed, because a wrong extension is a file the reader cannot group.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})

#: Spellings of the same format, folded so one page has one name.
_EXTENSION_ALIASES = {".jpeg": ".jpg"}


class ScrapeError(Exception):
    """A refusal that retrying will not fix."""


def load_config(path=None):
    """Read ``comic_config.json`` via the reader's loader; imported lazily.

    Deferred so a download-only install (``--extra scraper``) does not need the
    OCR extra's numpy/wordninja just to import this module.
    """
    from comic_reader import load_config as _load

    return _load(path)


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
    address to fetch is in ``data-src``. Only ``data-src`` is read -- the
    placeholder lives on ``comic-site.test`` too, so it would pass
    ``safe_image_url`` and be downloaded as if it were a page. Off-site URLs are
    dropped here as well as at fetch time, so nothing downstream ever holds one.
    """

    soup = BeautifulSoup(html, "html.parser")
    urls: list[str] = []
    for img in soup.select("img.wp-manga-chapter-img"):
        url = (img.get("data-src") or "").strip()
        if url and safe_image_url(url):
            urls.append(url)
    return urls


# -- downloading the pages --------------------------------------------------


@dataclass(frozen=True)
class ChapterResult:
    """What one chapter's download actually did.

    ``saved`` and ``failed`` describe this run; ``on_disk`` is how many pages
    the chapter has afterwards, so a resumed chapter reports what the reader
    will see rather than what this run fetched. A chapter with ``saved == 0``
    and ``failed == 0`` and ``on_disk == 0`` is a chapter with no images, which
    is not the same as a success.
    """

    record: ChapterRecord
    saved: int
    failed: int
    on_disk: int


def pages_on_disk(out_dir: Path, ordinal: int) -> int:
    """How many pages are already saved on disk for chapter ``ordinal``.

    The match is on the zero-padded ordinal (``c_000005_``), never a bare
    prefix: chapter 5 and chapter 50 both begin with ``c_0000``, so a prefix
    match would count one chapter's pages into another's total and silently
    merge them.
    """

    from comic_reader import PAGE_EXTENSIONS  # deferred: see module docstring

    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return 0
    prefix = f"c_{ordinal:06d}_"
    return sum(
        1
        for path in out_dir.glob("c_*")
        if path.name.startswith(prefix) and path.suffix.lower() in PAGE_EXTENSIONS
    )


def download_chapter(
    series_url: str,
    record: ChapterRecord,
    out_dir: Path,
    *,
    overwrite: bool = False,
    delay: float = 0.4,
) -> ChapterResult:
    """Save every page of one chapter into ``out_dir``, and count honestly.

    A page is only counted as saved once its bytes are on disk and passed
    ``looks_like_image``; anything else is counted as failed and named on
    stdout. That is the whole point of the count: the previous version of this
    script reported success while downloading nothing, so a chapter that saves
    zero pages must not look like a chapter that went fine.

    A chapter page that is a Cloudflare challenge raises instead of returning
    zero pages: a challenge carries no images, and "no images" read as a
    complete chapter is exactly how a download gets truncated silently.
    """

    from comic_reader import page_filename  # deferred: see module docstring
    from comic_images import looks_like_image

    out_dir = Path(out_dir)
    url = chapter_url(series_url, record.slug)
    status, body = default_transport(url, {}, {})
    if status != 200:
        raise ScrapeError(f"{record.label}: HTTP {status} at {url}")
    if looks_like_challenge(body):
        raise ScrapeError(
            f"{record.label}: Cloudflare challenge instead of the chapter page "
            f"at {url}. A challenge page has no images; refusing rather than "
            "reporting a zero-page chapter."
        )

    urls = page_urls(body)
    # The chapter lists this many pages, whatever happens to each of them
    # below. Saved bytes are not the measure: a page that failed still exists on
    # the site, and the scan pass uses this to notice a chapter that came up
    # short.
    record.pages = len(urls)

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    failed = 0
    for index, image_url in enumerate(urls, start=1):
        try:
            ext = image_ext(image_url)
        except ScrapeError as exc:
            failed += 1
            print(f"  page {index}: failed: {exc}")
            continue

        name = page_filename(record.ordinal, index, ext)
        path = out_dir / name
        if path.exists() and not overwrite:
            continue

        if delay:
            time.sleep(delay)

        status, payload = bytes_transport(image_url, {"Referer": series_url}, {})
        if status != 200:
            failed += 1
            print(f"  {name}: failed: HTTP {status}")
            continue
        if not looks_like_image(payload):
            failed += 1
            print(
                f"  {name}: failed: not an image "
                f"({len(payload)} bytes, likely an error page)"
            )
            continue

        # Written under its final name only after the bytes are known to be an
        # image, so a half-finished run never leaves a file that looks like a
        # page but cannot be read.
        path.write_bytes(payload)
        saved += 1
        print(f"  {name}: saved")

    on_disk = pages_on_disk(out_dir, record.ordinal)
    print(
        f"{record.label}: {saved} saved, {failed} failed, {on_disk} on disk "
        f"of {record.pages} listed"
    )
    return ChapterResult(record=record, saved=saved, failed=failed, on_disk=on_disk)


# -- network ---------------------------------------------------------------


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
