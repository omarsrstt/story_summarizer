"""Read a comic series' chapter list and page images, on the fast scraper's stack.

This is the stage ``comic_reader.py`` cannot do: getting the pictures at all.
It saves them as ``c_<ordinal>_<page>.<ext>``, where the ordinal is the
chapter's pinned position in reading order -- the naming
``comic_reader.group_chapters`` parses -- so the rest of the pipeline is
unchanged.

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

* A chapter's reading-order position and its CDN directory number are different
  things, and the CDN number cannot identify a chapter: the site files chapter N
  under ``chapter_<N-1>``, and a re-upload can publish the same number again
  (``Chapter 144`` and ``Chapter 144.0`` are two chapters that both carry 144.0).
  Local files are therefore named by the ordinal pinned in ``chapters.json``,
  which never changes once assigned.
* The chapter list is *not* JavaScript-only, as it first appeared: a plain GET
  of the AJAX endpoint answers ``0``, but a POST carrying
  ``X-Requested-With: XMLHttpRequest`` returns all ~300 chapters in one
  response. Walking the next-chapter links was a workaround for a request we
  were making wrong, and it is gone.

Only the host named by ``comic_site_host`` in ``comic_config.json`` and its
subdomains are fetched, over HTTP(S) only. The pages inject obfuscated scripts
that redirect to ad hosts, so an off-site image URL is refused rather than
followed, and no JavaScript is ever executed.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from comic_manifest import (
    ChapterRecord,
    Manifest,
    assign,
    load_manifest,
    save_manifest,
)
from scraper import parse_chapter_input
from scraper_fast import (  # noqa: F401 - re-exported
    _with_retries,
    bytes_transport,
    default_transport,
    post_transport,
)
from scraper_parse import looks_like_challenge

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "comic_config.json"

#: Where a series lives, keyed by slug. Overridable via ``comic_series_url`` so
#: the site can move without editing code.
DEFAULT_SERIES_URL = "https://comic-site.test/manga/{slug}/"

#: The host page images may come from. Overridable via ``comic_site_host``.
DEFAULT_SITE_HOST = "comic-site.test"

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

    The site's host comes from ``comic_site_host`` in the config rather than
    being spelled here, so moving the site is a config change. The signature
    keeps its single argument -- a caller asks about a URL, not about a config.
    """

    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    site_host = (load_config().get("comic_site_host") or DEFAULT_SITE_HOST).lower()
    host = (parsed.hostname or "").lower()
    return host == site_host or host.endswith("." + site_host)


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


# -- deciding which chapters ------------------------------------------------
#
# A site label is a chapter's identity; its number is not. The site publishes
# "Chapter 144" and "Chapter 144.0" as two different chapters that both carry the
# number 144.0, so selecting by number is ambiguous between them while selecting
# by label is not. Everything below therefore matches labels first.


def normalise_label(text: str) -> str:
    """``text`` folded so a site label and a selection compare equal.

    Case, a leading ``chapter`` and repeated whitespace are all noise: the site
    writes ``Chapter 175.1``, a user types ``175.1``, and both name the same
    chapter. Folding here keeps that rule in one place instead of at every call
    site.
    """

    text = " ".join(text.split()).casefold()
    if text.startswith("chapter"):
        text = text[len("chapter"):]
    return " ".join(text.split())


def _label_number(text: str) -> float:
    """The number in ``text``, or ``ValueError`` if it is not just a number."""

    try:
        return float(text.strip())
    except ValueError:
        raise ValueError(f"not a chapter label: {text!r}") from None


def _decimals(text: str) -> int:
    """How many digits ``text`` writes after the point (0 when it writes none).

    This is what fixes a range's step: ``174-176`` steps by 1 and
    ``175.1-175.5`` by 0.1, because that is how finely the endpoints are
    written. Guessing a fixed 0.1 would make a whole-number range ask for
    174.1, 174.2 ..., which the site does not publish.
    """

    _, dot, fraction = text.strip().partition(".")
    return len(fraction) if dot else 0


def _label_terms(text: str) -> list[tuple[str, float]]:
    """``(label text, number)`` for every chapter a selection names.

    The label text is kept separately from the number because the two are not
    interchangeable: ``"144"`` and ``"144.0"`` are different labels that parse
    to the same number, and only the text tells them apart.

    A part containing ``-`` is an inclusive range. Values are counted from the
    start and rounded to the endpoints' precision, because accumulating a
    fractional step yields ``175.29999999999998``, which matches no label and
    would silently select nothing.

    Note that a float list cannot name ``175.10`` and ``175.1`` separately --
    they are the same number. ``select_plan`` therefore treats a range as a
    span of the manifest's labels rather than of this list; this parser is the
    answer to "which numbers does the text mention".
    """

    terms: list[tuple[str, float]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty selection in {text!r}")
        if _is_range(part):
            start_text, _, end_text = part.partition("-")
            start, end = _label_number(start_text), _label_number(end_text)
            places = max(_decimals(start_text), _decimals(end_text))
            step = 10.0 ** -places
            count = round((end - start) / step)
            for index in range(count + 1):
                number = round(start + index * step, places)
                # ``:g`` keeps a whole number spelled "175", which is how the
                # site writes one; "175.0" matches no label.
                terms.append((f"{number:g}", number))
        else:
            terms.append((normalise_label(part), round(_label_number(part), 3)))
    return terms


def parse_label_selection(text: str) -> list[float]:
    """The numbers named by a ``-c`` selection: ``"175.1"``, ``"144,150"``.

    Raises ``ValueError`` on anything that is not numbers, ranges and commas; an
    unreadable selection must not read as an empty one. This is the conversion a
    human's selection goes through, not the matcher -- ``select_plan`` matches
    the label text, because numbers alone cannot distinguish ``144`` from
    ``144.0``.
    """

    return [number for _, number in _label_terms(text)]


def _record_for(manifest: Manifest, text: str) -> ChapterRecord:
    """The one chapter ``text`` names, or ``ValueError`` saying what is missing.

    Labels first: ``normalise_label(record.label)`` is the site's own identifier,
    so ``144`` and ``144.0`` resolve to different chapters. The number is only a
    fallback, within 0.05 -- enough for a label written ``Ch. 144`` where the
    user typed ``144``, not enough to reach the neighbouring ``144.1``.
    """

    label = normalise_label(text)
    number = _label_number(label)

    for record in manifest.chapters:
        if normalise_label(record.label) == label:
            return record

    candidates = [
        candidate
        for candidate in manifest.chapters
        if abs(candidate.number - number) <= 0.05
    ]
    if candidates:
        return min(candidates, key=lambda c: c.ordinal)

    hint = ""
    if number % 1 == 0:
        # -c takes labels and --ordinals takes positions; a plain number is the
        # one input that reads the same in both, so name the other one.
        hint = f" (if you meant position {int(number)}, use --ordinals)"
    raise ValueError(f"label {label} is not in the chapter list{hint}")


def select_plan(manifest: Manifest, text: str | None) -> list[ChapterRecord]:
    """The chapters ``text`` names, in manifest (reading) order.

    ``None`` or ``"all"`` means everything. Otherwise ``text`` is a comma list of
    site labels and label ranges, each resolved by ``_record_for``.

    A range is a **span of the manifest**, not a run of numbers: it covers every
    chapter between its two endpoints in reading order. That is what the user
    means by ``175.1-175.19``, and it is the only way to select those chapters
    honestly -- ``175.10`` and ``175.1`` are the same number, but different
    labels, so counting in floats cannot name the set.

    A label that matches nothing is an error, not an empty selection: quietly
    downloading nothing because of a typo is worse than naming the label that was
    not found.
    """

    if not text:
        return list(manifest.chapters)
    selection = normalise_label(text)
    if selection in ("", "all"):
        return list(manifest.chapters)

    chapters = manifest.chapters
    chosen: dict[int, ChapterRecord] = {}
    for part in text.split(","):
        part = normalise_label(part)
        if not part:
            raise ValueError(f"empty selection in {text!r}")
        if _is_range(part):
            start_text, _, end_text = part.partition("-")
            start, end = _record_for(manifest, start_text), _record_for(manifest, end_text)
            first, last = chapters.index(start), chapters.index(end)
            if first > last:
                raise ValueError(
                    f"range {part} runs backwards in reading order "
                    f"({start.label} is chapter {start.ordinal}, {end.label} is "
                    f"{end.ordinal})"
                )
            for record in chapters[first:last + 1]:
                chosen[record.ordinal] = record
        else:
            record = _record_for(manifest, part)
            chosen[record.ordinal] = record

    return [record for record in chapters if record.ordinal in chosen]


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


# -- deciding what not to download -----------------------------------------


def scan_chapter(series_url: str, record: ChapterRecord) -> ChapterRecord:
    """Fill in ``record.pages`` and ``record.content_hash`` without saving a page.

    One GET for the chapter page and one for page 1: the count of listed pages,
    plus a fingerprint of the first one. That is the minimum needed to decide
    whether two chapters are the same before any bytes are written -- the whole
    point is that a duplicate costs one request, not eighty.

    Nothing is written to disk. The hash is the empty string when page 1 cannot
    be fetched, is not an image, or is an image that will not decode, which
    ``mark_duplicates`` treats as "not a candidate" rather than as a value two
    chapters can share.

    Raises ``ScrapeError`` on a non-200 or a Cloudflare challenge, matching
    ``download_chapter``: a challenge page lists zero images, and recording that
    as a zero-page chapter would hide the fact that nothing was scanned.
    """

    from comic_images import average_hash, looks_like_image

    url = chapter_url(series_url, record.slug)
    status, body = default_transport(url, {}, {})
    if status != 200:
        raise ScrapeError(f"{record.label}: HTTP {status} at {url}")
    if looks_like_challenge(body):
        raise ScrapeError(
            f"{record.label}: Cloudflare challenge instead of the chapter page "
            f"at {url}. A challenge page lists no images, so it cannot be "
            "scanned for duplicates."
        )

    urls = page_urls(body)
    record.pages = len(urls)
    # Only page 1 is fetched. Pages 2-10 of the real duplicate pairs measured
    # 0 bits apart on the average hash, so one page settles it; hashing all of
    # them would cost a chapter's bandwidth to answer the same question.
    record.content_hash = ""
    if urls:
        status, payload = bytes_transport(urls[0], {"Referer": series_url}, {})
        if status == 200 and looks_like_image(payload):
            try:
                record.content_hash = average_hash(payload)
            except ValueError:
                # Magic bytes but no decodable picture: a response cut off
                # mid-transfer still starts like a JPEG. ``looks_like_image``
                # is a prefix test and cannot see that, but ``average_hash``
                # decodes and does. One truncated page must not abort a scan of
                # the whole series, so it reads as unhashable instead.
                print(f"{record.label}: page 1 has image magic bytes but is not decodable")

    print(
        f"{record.label}: scanned {record.pages} pages, "
        f"hash {'none' if not record.content_hash else record.content_hash[:12]}"
    )
    return record


def mark_duplicates(records: list[ChapterRecord]) -> None:
    """Mark every chapter that duplicates another, keeping one from each group.

    Chapters are grouped by ``content_hash`` and, within a group, the record with
    the most pages survives; ties go to the one earliest in reading order. Every
    other member gets ``duplicate_of`` set to the survivor's ordinal, which is
    what lets ``run`` skip it before fetching anything.

    An empty hash is never grouped. A chapter whose page 1 could not be fetched
    has no fingerprint, and treating ``""`` as a value would make every
    unhashable chapter a duplicate of the first one -- which, on a series where a
    scan failure is routine, is most of the series.

    Re-running this on an already-marked list is safe and is the intended use:
    every flag is cleared as it is recomputed, so the answer is always the
    current hashes' and never a mixture of this scan's and the last one's.

    Mutates ``records`` in place; returns nothing so there is no second copy of
    the answer to disagree with it.
    """

    groups: dict[str, list[ChapterRecord]] = {}
    for record in records:
        # A re-scan decides the whole question again, so a flag left from the
        # previous run is cleared first: a chapter that stops hashing (its page 1
        # failed this time) would otherwise stay marked as a duplicate of a
        # chapter that is no longer known to be the same one.
        record.duplicate_of = None
        if not record.content_hash:
            continue
        groups.setdefault(record.content_hash, []).append(record)

    for group in groups.values():
        if len(group) < 2:
            continue
        # Most pages wins; a tie goes to the lower ordinal, i.e. the chapter
        # earliest in reading order. Sorting on the ordinal explicitly makes the
        # winner independent of the order the caller passed the records in --
        # `max` alone would hand a tie to whichever came first *in the list*.
        winner = max(group, key=lambda r: (r.pages, -r.ordinal))
        for record in group:
            if record is not winner:
                record.duplicate_of = winner.ordinal


def has_legacy_files(out_dir: Path) -> bool:
    """Whether ``out_dir`` holds pages named before ordinals were pinned.

    The old script wrote ``c_<chapter>_<page>.jpg`` with the chapter number
    unpadded (``c_7_1.jpg``). ``comic_reader`` reads any digit run, so those
    files would be merged into the new six-digit numbering -- chapter 7's old
    pages and its new pages would both claim ``c_000007_*`` and every later
    chapter would read as having twelve pages where it has ten.

    The test is the width of the chapter digits, not the presence of the file:
    six digits is the shape this module writes, anything else is the old one. A
    missing directory has no files, so it is not legacy.
    """

    from comic_reader import CHAPTER_PATTERNS  # deferred: see module docstring

    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return False
    pattern = CHAPTER_PATTERNS[0]
    for path in out_dir.iterdir():
        match = pattern.match(path.name)
        if match and len(match.group(1)) != 6:
            return True
    return False


def archive_hint(out_dir: Path) -> str:
    """The message for a tree holding pages named before ordinals were pinned.

    Two pasteable PowerShell lines, not a mix of Unix ``mkdir -p`` and a Windows
    ``move``: the command is meant to be run, not read.
    """

    return (
        f"{out_dir} holds pages named before ordinals were pinned "
        f"(c_<chapter>_<page>.jpg, e.g. c_7_1.jpg). Reading them alongside the "
        f"new c_000007_001.jpg names would merge two chapters' pages. Archive "
        f"them first, from the repo root, into unused/2026-11-01-pre-ordinal:\n"
        f"  mkdir unused\\2026-11-01-pre-ordinal\n"
        f"  move {out_dir}\\c_*_*.* unused\\2026-11-01-pre-ordinal\\"
    )


# -- the run ---------------------------------------------------------------


def run(
    series_url: str,
    out_dir: Path,
    manifest: Manifest,
    plan: list[ChapterRecord],
    *,
    overwrite: bool = False,
    delay: float = 0.4,
    dry_run: bool = False,
) -> int:
    """Download ``plan`` into ``out_dir``, reporting honestly. Returns an exit code.

    Duplicate detection is the caller's job, not this function's. A scan costs a
    request per chapter and decides what should not be fetched at all, so it
    belongs before the run, not inside it: the caller scans the plan, calls
    ``mark_duplicates``, and passes the same records here. ``run`` only honours
    ``duplicate_of`` flags that are already set. Scanning again here would fetch
    page 1 twice for every chapter and would overwrite the page count of a
    chapter that is already fully downloaded with the count of a page it cannot
    see -- so ``run`` leaves ``record.pages`` to ``download_chapter``.

    The manifest is saved after every chapter, including a failed one. That is
    what makes an interrupted run resumable without renumbering: the ordinals
    already handed out are on disk, so the next run assigns the same ones.

    ``dry_run`` prints what each chapter would do -- ordinal, label, pages
    already on disk -- and saves the manifest, but writes no pages.

    The exit code is about what is on disk, not what this run did: ``0`` when
    every non-duplicate chapter has pages on disk and none failed, ``1``
    otherwise. A fully resumed run that downloaded nothing is a success; a run
    that saved nothing is not.
    """

    out_dir = Path(out_dir)
    manifest_path = out_dir / "chapters.json"

    if has_legacy_files(out_dir):
        print(archive_hint(out_dir))
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    skipped_duplicates = 0

    for record in plan:
        if record.duplicate_of is not None:
            skipped_duplicates += 1
            print(
                f"{record.label} (chapter {record.ordinal}): skipped, duplicate "
                f"of chapter {record.duplicate_of}"
            )
            continue

        if dry_run:
            on_disk = pages_on_disk(out_dir, record.ordinal)
            print(
                f"{record.label} (chapter {record.ordinal}): would download, "
                f"{on_disk} page(s) already on disk"
            )
            # The manifest still records this chapter: a dry run that decided
            # the numbering and then discarded it would assign different
            # ordinals on the real run.
            manifest.chapters = _with_record(manifest.chapters, record)
            save_manifest(manifest_path, manifest)
            continue

        try:
            result = download_chapter(
                series_url, record, out_dir, overwrite=overwrite, delay=delay
            )
        except ScrapeError as exc:
            # A refused chapter page still gets its manifest entry saved: the
            # ordinal this run assigned to it must survive, or the next run
            # would hand it a different one. It carries its cause verbatim and
            # the run continues -- one unavailable chapter out of 400 is not a
            # reason to abandon the other 399, and the exit code at the end
            # still reports it.
            print(f"{record.label}: FAILED: {exc}")
            failures.append(f"{record.label}: {exc}")
            manifest.chapters = _with_record(manifest.chapters, record)
            save_manifest(manifest_path, manifest)
            continue

        manifest.chapters = _with_record(manifest.chapters, record)
        # Saved per chapter, before the next one starts: an interrupted run keeps
        # every ordinal it has already assigned.
        save_manifest(manifest_path, manifest)

        if result.failed > 0:
            failures.append(
                f"{record.label}: {result.failed} page(s) failed"
            )
        if result.on_disk == 0:
            failures.append(f"{record.label}: no pages on disk")

    print(
        f"{len(plan)} chapter(s) in the plan: "
        f"{len(plan) - skipped_duplicates} to download, "
        f"{skipped_duplicates} duplicate(s) skipped"
    )
    if failures:
        print(f"{len(failures)} problem(s):")
        for line in failures:
            print(f"  {line}")
        return 1
    return 0


def _with_record(
    chapters: list[ChapterRecord], record: ChapterRecord
) -> list[ChapterRecord]:
    """``chapters`` with ``record`` stored by ordinal, appended when new.

    The plan and the manifest can disagree -- a chapter added by this scan is
    not in the manifest yet -- so the record is matched on ordinal and replaced
    rather than assumed to be present. Replacing matters: the manifest is what
    the next run reads its page counts and duplicate flags from.
    """

    for index, existing in enumerate(chapters):
        if existing.ordinal == record.ordinal:
            chapters[index] = record
            return chapters
    return chapters + [record]


# -- the command line -------------------------------------------------------


@dataclass
class Setup:
    """What ``main`` resolved before a single page is fetched.

    Grouped so the plan can be printed (and, in a dry run, only printed) without
    the setup code having written anything: the manifest is held in memory until
    the caller decides the run is real.
    """

    series_url: str
    out_dir: Path
    manifest_path: Path
    manifest: Manifest
    plan: list[ChapterRecord]


def series_url_for(slug: str, config: dict | None = None) -> str:
    """The series URL for ``slug``, from ``comic_series_url`` in the config."""

    if config is None:
        config = load_config()
    template = config.get("comic_series_url") or DEFAULT_SERIES_URL
    return template.format(slug=slug.strip("/"))


def _ordinal_selection(manifest: Manifest, text: str) -> list[ChapterRecord]:
    """The chapters at the manifest positions ``text`` names.

    Positions are the ordinals in ``chapters.json``, which is why the integer
    parser is the right tool here: an ordinal is always an integer, unlike a
    site label, which can be ``175.1``.
    """

    try:
        wanted = parse_chapter_input(text)
    except ValueError:
        raise ScrapeError(
            f"--ordinals: cannot read {text!r}; expected positions like '1-5,8'"
        ) from None
    if not wanted:
        raise ScrapeError(f"--ordinals: {text!r} names no position")

    by_ordinal = {record.ordinal: record for record in manifest.chapters}
    missing = [number for number in wanted if number not in by_ordinal]
    if missing:
        highest = max(by_ordinal, default=0)
        raise ScrapeError(
            f"--ordinals: no chapter at position(s) {missing}; the manifest has "
            f"positions 1 to {highest}"
        )
    chosen = set(wanted)
    return [record for record in manifest.chapters if record.ordinal in chosen]


def _is_range(text: str) -> bool:
    """Whether ``text`` is a label range like ``174-176``.

    Both sides must be numbers: a single label can contain a hyphen
    (``chapter-175-1`` is the site's slug form of ``175.1``), and treating that
    as a range would silently select a span nobody asked for.
    """

    start, dash, end = text.partition("-")
    if not dash or not start.strip() or not end.strip():
        return False
    try:
        float(start)
        float(end)
    except ValueError:
        return False
    return True


def _label_at(manifest: Manifest, label: str, flag: str) -> ChapterRecord:
    """The single chapter ``label`` names, with the selection error made clean.

    Uses ``_record_for`` rather than ``select_plan`` so a range cannot be passed
    where one label is expected: ``--from 1-5`` would otherwise silently mean
    chapter 1.
    """

    if _is_range(label):
        raise ScrapeError(f"{flag}: expected one label, not a range ({label!r})")
    try:
        return _record_for(manifest, label)
    except ValueError as exc:
        raise ScrapeError(f"{flag}: {exc}") from None


def _bounds_selection(
    manifest: Manifest, from_label: str | None, to_label: str | None
) -> list[ChapterRecord]:
    """Every chapter between two site labels, inclusive, in reading order."""

    chapters = manifest.chapters
    start = _label_at(manifest, from_label, "--from") if from_label is not None else chapters[0]
    end = _label_at(manifest, to_label, "--to") if to_label is not None else chapters[-1]
    first, last = chapters.index(start), chapters.index(end)
    if first > last:
        raise ScrapeError(
            f"--from {start.label!r} comes after --to {end.label!r} in reading "
            f"order (ordinals {start.ordinal} and {end.ordinal}); swap them"
        )
    return chapters[first:last + 1]


def prepare(args) -> Setup:
    """Resolve the config, the chapter list, the manifest and the plan.

    Nothing is written and no page is fetched: one chapter-list request is the
    whole cost, so a dry run can stop here.
    """

    config = load_config()
    series_url = series_url_for(args.slug, config)
    out_dir = REPO_ROOT / config.get("comics_dir", "comics") / args.name
    manifest_path = out_dir / "chapters.json"

    manifest = load_manifest(manifest_path)
    # The site and series are known from the arguments, so they are written
    # rather than inherited: a manifest that disagrees with the URL it was built
    # from is worse than one without them.
    manifest.series = args.name
    manifest.site = series_url

    listing = fetch_chapter_list(series_url)
    if not listing:
        raise ScrapeError(f"the chapter list at {series_url} is empty")
    manifest = assign(manifest, listing)

    if args.ordinals is not None:
        plan = _ordinal_selection(manifest, args.ordinals)
    elif args.from_label is not None or args.to_label is not None:
        plan = _bounds_selection(manifest, args.from_label, args.to_label)
    else:
        # ``select_plan`` treats None and "all" alike; ``--all`` and no selection
        # both land here, and an ambiguous combination was refused in parse_args.
        try:
            plan = select_plan(manifest, args.chapters)
        except ValueError as exc:
            raise ScrapeError(f"-c: {exc}") from None

    if not plan:
        raise ScrapeError("the selection named no chapters")
    return Setup(series_url, out_dir, manifest_path, manifest, plan)


def print_plan(setup: Setup) -> None:
    """Say what the resolved plan is, without asking the site anything else."""

    plan = setup.plan
    print(f"{setup.out_dir}  ({setup.series_url})")
    print(
        f"resolved plan: {len(plan)} chapter(s), first {plan[0].label} "
        f"(ordinal {plan[0].ordinal}), last {plan[-1].label} "
        f"(ordinal {plan[-1].ordinal})"
    )
    for record in plan:
        print(f"  {record.ordinal:06d}  {record.label}")


def parse_args(argv: list[str] | None = None):
    """The download stage's arguments.

    ``-c`` and ``--ordinals`` are separate flags on purpose: ``-c`` takes the
    site's own labels, ``--ordinals`` takes positions in ``chapters.json``. They
    overlap numerically often enough (``-c 144`` and ``--ordinals 144``) that one
    flag would blur the distinction this stage exists to keep.
    """

    parser = argparse.ArgumentParser(
        prog="comic_scraper.py",
        description=(
            "Download a comic series' pages into comics/<name>/, named by the "
            "ordinal pinned in chapters.json."
        ),
        epilog=(
            "A run scans one chapter page and one image per chapter before "
            "downloading anything, so duplicate re-uploads are settled before "
            "bytes are written. --dry-run skips that scan."
        ),
    )
    parser.add_argument(
        "-n", "--name", required=True,
        help="Folder under comics/ to download into (required)")
    parser.add_argument(
        "--slug", default=None,
        help="Series slug from the site URL (default: the value of --name)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "-c", "--chapters", default=None, metavar="LABELS",
        help="Site labels to download: '5', '1-5', '175.1-175.19', '144,150'")
    selection.add_argument(
        "--ordinals", default=None, metavar="POSITIONS",
        help="Positions in chapters.json instead of site labels: '5', '1-5,8'")
    selection.add_argument(
        "--all", dest="select_all", action="store_true",
        help="Download every chapter in the series")
    parser.add_argument(
        "--from", dest="from_label", default=None, metavar="LABEL",
        help="First site label to download, inclusive")
    parser.add_argument(
        "--to", dest="to_label", default=None, metavar="LABEL",
        help="Last site label to download, inclusive")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download pages that already exist")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report the resolved plan without scanning or downloading")
    parser.add_argument(
        "--delay", type=float, default=0.4, metavar="SECONDS",
        help="Pause before each page request (default: 0.4)")
    args = parser.parse_args(argv)

    if args.slug is None:
        args.slug = args.name
    if args.from_label is not None or args.to_label is not None:
        if args.chapters or args.ordinals or args.select_all:
            parser.error(
                "--from/--to cannot be combined with -c, --ordinals or --all")
    if args.chapters is not None and not args.chapters.strip():
        parser.error("-c needs a selection, e.g. -c 1-5")
    if args.ordinals is not None and not args.ordinals.strip():
        parser.error("--ordinals needs a selection, e.g. --ordinals 1-5")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    return args


def execute(setup: Setup, args) -> int:
    """Scan the plan for duplicates, then download it.

    The scan always runs, even when the manifest already carries
    ``duplicate_of``: that flag is this run's answer to a question about the
    pages as they are now, and it is cleared and recomputed as the scan goes.
    Trusting the file's value would let an old decision survive a re-upload.
    """

    if has_legacy_files(setup.out_dir):
        # Checked before the scan, not after: a scan of a doomed tree is a
        # request per chapter spent to reach a refusal already known here.
        print(archive_hint(setup.out_dir))
        return 1

    setup.out_dir.mkdir(parents=True, exist_ok=True)
    # Pinned before any page is written: an interrupted run then keeps the
    # ordinals it had already decided, and the next run assigns the same ones.
    save_manifest(setup.manifest_path, setup.manifest)

    print(
        f"scanning {len(setup.plan)} chapter(s) for duplicates: the chapter page "
        f"and page 1 of each are fetched first, so no page is downloaded twice"
    )
    scanned: list[ChapterRecord] = []
    for record in setup.plan:
        try:
            scanned.append(scan_chapter(setup.series_url, record))
        except ScrapeError as exc:
            # One unavailable chapter page must not abort the scan of the rest.
            # The record stays unhashed, so it is not grouped as a duplicate and
            # ``run`` will try it properly and report its own failure.
            print(f"{record.label}: scan skipped: {exc}")
            scanned.append(record)

    mark_duplicates(scanned)
    # Re-derived duplicates are only worth having if they are on disk: the next
    # run reads this file, and a flag that exists only in memory is a decision
    # nobody can inspect.
    save_manifest(setup.manifest_path, setup.manifest)

    return run(
        setup.series_url,
        setup.out_dir,
        setup.manifest,
        setup.plan,
        overwrite=args.overwrite,
        delay=args.delay,
    )


def main(argv: list[str] | None = None) -> int:
    """Download a comic series' pages. Returns 0 on success, 1 on any failure."""

    args = parse_args(argv)
    try:
        setup = prepare(args)
        print_plan(setup)
        if args.dry_run:
            # Deliberately no scan: --dry-run answers "what would you download?",
            # and a scan answers "which are duplicates?" -- a question the user
            # did not ask, costing a request per chapter on a 300-chapter series.
            print("dry run: nothing scanned, nothing written")
            return 0
        return execute(setup, args)
    except ScrapeError as exc:
        # A refusal that retrying will not fix, reported as one line: a traceback
        # buries the cause, which is the whole message.
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
