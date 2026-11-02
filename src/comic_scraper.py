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

from comic_manifest import ChapterRecord, Manifest, save_manifest
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
        print(
            f"{out_dir} holds pages named before ordinals were pinned "
            f"(c_<chapter>_<page>.jpg, e.g. c_7_1.jpg). Reading them alongside "
            f"the new c_000007_001.jpg names would merge two chapters' pages. "
            f"Archive them first:\n"
            f"  mkdir -p unused/2026-11-01-pre-ordinal\n"
            f"  move {out_dir}\\c_*_*.* unused/2026-11-01-pre-ordinal\\"
        )
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
