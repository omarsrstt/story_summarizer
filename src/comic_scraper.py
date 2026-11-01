"""Download comic chapters as page images, reusing the fast scraper's stack.

This is the stage ``comic_reader.py`` cannot do: getting the pictures at all.
It saves them as ``<comics>/<name>/c_<storage-chapter>_<page>.jpg`` -- the naming
the existing downloads already use and that ``comic_reader.group_chapters``
parses -- so the rest of the pipeline is unchanged:

    uv run python src/comic_scraper.py -n TBATE -c 1-5
    uv run python src/comic_reader.py -s TBATE -c comics/TBATE --only 0-4

What it reuses from the novel scraper, and why each one matters:

* ``scraper_parse.looks_like_challenge`` -- a Cloudflare page carries no reader
  images, which without this test is indistinguishable from the end of a series.
  Ignoring it would truncate a download silently and report success.
* ``scraper_fast.default_transport`` -- the ``curl_cffi`` TLS impersonation the
  site's bot check requires, plus timeout/5xx retries and 429 backoff.
* ``scraper.parse_chapter_input`` -- the ``1-5,8`` selection syntax.
* ``scraper_store.chapter_filename`` -- the five-digit, zero-padded chapter
  number the rest of the pipeline uses.

Two things about this site were measured, not assumed:

* The reader's chapter N is stored on the CDN as ``chapter_<N-1>``. Chapter 1
  is ``chapter_0/c_0_1.jpg``; chapter 2 is ``chapter_1/c_1_1.jpg``. The local
  filename keeps the *storage* numbering, because that is what the existing
  downloads use and what ``comic_reader`` groups on -- so a chapter can look off
  by one in a filename and still be correct.
* The chapter list is loaded by JavaScript and its endpoint answers ``0`` to a
  plain request, but each chapter page links to its neighbours. A run therefore
  walks those links (``--all``) and stops when a page carries no images, which
  is exactly what the site returns for a chapter that does not exist.

Only ``comic-site.test`` and its subdomains are fetched, over HTTP(S) only. The
pages inject obfuscated scripts that redirect to ad hosts, so an off-site image
URL is refused rather than followed, and no JavaScript is ever executed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from scraper import parse_chapter_input
from scraper_fast import default_transport
from scraper_parse import looks_like_challenge

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "comic_config.json"

SITE = "https://comic-site.test"

#: Reader images are named ``c_<chapter>_<page>.<ext>`` -- the same shape the
#: existing downloads use, so a scrape and a manual save are interchangeable.
IMAGE_NAME = re.compile(r"^c_(\d+)_(\d+)\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)

#: Magic bytes, so an error page served with a ``.jpg`` name is not kept.
IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"RIFF", b"GIF8")

#: Real chapter images are lazy-loaded: ``src`` is a placeholder and the address
#: to fetch is in ``data-src``. Reading ``src`` downloads the loader GIF.
_IMG_TAG = re.compile(r"<img\b[^>]*wp-manga-chapter-img[^>]*>", re.IGNORECASE)
_DATA_SRC = re.compile(r'data-src\s*=\s*"([^"]+)"', re.IGNORECASE)
_CHAPTER_HREF = re.compile(r'href\s*=\s*"([^"]*?/chapter-(\d+)/?)"', re.IGNORECASE)

#: Image retries. Mirrors ``scraper_fast``'s policy for chapter bodies: a blip
#: is retried with backoff, a throttle is waited out, a 404 is not retried.
_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 0.5
_RATE_LIMIT_MAX_ATTEMPTS = 4
_RATE_LIMIT_BASE = 10.0

#: Stop a ``--all`` walk after this many chapters with no new number, so a
#: misconfigured slug cannot spin forever.
_MAX_WALK = 2000


class ScrapeError(Exception):
    """A refusal that retrying will not fix."""


@dataclass
class ChapterPage:
    """What one reader page offered."""

    chapter: int
    images: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.images)


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


def parse_chapter_page(html: str, series_url: str, chapter: int) -> ChapterPage:
    """Collect the reader images on a chapter page, in page order."""

    images: list[str] = []
    for tag in _IMG_TAG.findall(html):
        match = _DATA_SRC.search(tag)
        if not match:
            continue
        url = match.group(1)
        if safe_image_url(url):
            images.append(url)
    return ChapterPage(chapter=chapter, images=images)


def next_chapter(html: str, series_url: str, chapter: int) -> int | None:
    """The chapter after ``chapter``, from the page's own navigation.

    Only links belonging to this series count -- a recommendations block links
    elsewhere, and following it would silently download a different comic.
    """

    prefix = series_url.rstrip("/") + "/"
    candidates = set()
    for href, number in _CHAPTER_HREF.findall(html):
        full = href if href.startswith("http") else SITE + href
        if not full.startswith(prefix):
            continue
        value = int(number)
        if value > chapter:
            candidates.add(value)
    return min(candidates) if candidates else None


def image_filename(url: str) -> str:
    """The local filename for an image URL.

    Keeps the site's own name, which encodes chapter and page, rather than
    inventing one: ``comic_reader.group_chapters`` parses this pattern.
    """

    name = urlparse(url).path.rsplit("/", 1)[-1]
    if not IMAGE_NAME.match(name):
        raise ValueError(f"Not a chapter image name: {name!r}")
    return name


def load_config(path: Path | None = None) -> dict:
    config_path = path or CONFIG_PATH
    if not config_path.is_file():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


# -- network ---------------------------------------------------------------


def _looks_like_image(payload: bytes) -> bool:
    return any(payload.startswith(magic) for magic in IMAGE_MAGIC)


@dataclass
class Run:
    """One scraping run: a reused connection, and the pages already fetched.

    A run needs each chapter page twice (images, then navigation), so the HTML
    is cached. Reusing one ``curl_cffi`` session keeps a 50-page chapter to one
    TLS handshake instead of fifty.
    """

    series_url: str
    session: object
    pages: dict[int, str] = field(default_factory=dict)

    def chapter_url(self, chapter: int) -> str:
        return f"{self.series_url.rstrip('/')}/chapter-{chapter}/"

    def html(self, chapter: int) -> str | None:
        """The chapter page, or ``None`` if the site has no such chapter.

        A Cloudflare page raises instead of returning ``None``: it also carries
        no images, and treating it as "no such chapter" is how a walk ends early
        while looking like it finished.
        """

        if chapter not in self.pages:
            status, body = default_transport(self.chapter_url(chapter), {}, {})
            if status == 404:
                return None
            if status != 200:
                # Not a missing chapter. A throttled or blocked request also
                # yields no images, so reporting it as "end of series" would
                # end a walk early and still look like a clean finish.
                raise ScrapeError(
                    f"chapter {chapter}: HTTP {status} at {self.chapter_url(chapter)}"
                    + (", rate limited by Cloudflare" if status == 429 else "")
                )
            if looks_like_challenge(body):
                raise ScrapeError(
                    f"chapter {chapter}: Cloudflare challenge instead of a page. "
                    "Wait a minute and retry; if it persists this site needs a "
                    "browser harvest (scraper_session.harvest_session)."
                )
            self.pages[chapter] = body
        return self.pages[chapter]

    def image_bytes(self, url: str) -> bytes:
        """Fetch one image, retrying blips and waiting out throttles.

        Separate from ``default_transport`` because that decodes to text, which
        would corrupt binary JPEG data.
        """

        referer = {"Referer": self.series_url}
        status, payload = 0, b""
        attempts = rate_attempts = 0
        while attempts < _MAX_ATTEMPTS:
            try:
                response = self.session.get(url, headers=referer, timeout=120)
                status, payload = response.status_code, response.content
            except Exception:  # noqa: BLE001 - timeouts surface as status 0
                status, payload = 0, b""
            if status == 429:
                rate_attempts += 1
                if rate_attempts >= _RATE_LIMIT_MAX_ATTEMPTS:
                    break
                time.sleep(_RATE_LIMIT_BASE * rate_attempts)
                continue
            if status == 200 and _looks_like_image(payload):
                return payload
            if status != 0 and status < 500:
                break
            attempts += 1
            if attempts < _MAX_ATTEMPTS:
                time.sleep(_BACKOFF_BASE * (2 ** (attempts - 1)))
        if status == 0:
            raise ScrapeError("no response after retries")
        raise ScrapeError(
            f"HTTP {status}" if status != 200
            else "not an image (served an HTML page?)"
        )


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


@dataclass
class ChapterPlan:
    """Which chapters to fetch: an explicit list, or a walk from a start point."""

    numbers: list[int] = field(default_factory=list)
    walk: bool = False
    first: int = 1

    def __bool__(self) -> bool:
        return self.walk or bool(self.numbers)


def plan_chapters(
    selection: str | None, first: int, last: int | None, walk: bool
) -> ChapterPlan:
    """Resolve the CLI into chapters to fetch.

    An explicit ``-c`` selection wins; then ``--from``/``--to``; then ``--all``,
    which walks the next-chapter links because the site's chapter list is not
    available to a plain request.
    """

    if selection is not None:
        # A blank `-c ""` is a mistake, not a request for chapter 1: report it
        # as an empty plan instead of quietly downloading something.
        stripped = selection.strip()
        if not stripped:
            return ChapterPlan()
        return ChapterPlan(numbers=parse_chapter_input(stripped) or [])
    if last is not None:
        return ChapterPlan(numbers=list(range(first, last + 1)))
    if walk:
        return ChapterPlan(walk=True, first=first)
    return ChapterPlan(numbers=[first])


def download_chapter(
    run: Run, page: ChapterPage, out_dir: Path, overwrite: bool, delay: float,
    dry_run: bool,
) -> int:
    """Save one chapter's images; returns the number of pages that failed."""

    pending = []
    for url in page.images:
        try:
            name = image_filename(url)
        except ValueError as error:
            print(f"  skipping: {error}")
            continue
        target = out_dir / name
        if target.is_file() and not overwrite:
            continue
        pending.append((url, target))

    print(f"\nChapter {page.chapter}: {len(page.images)} image(s), "
          f"{len(pending)} to download")
    if dry_run:
        print(f"  [dry run] would fetch into {out_dir}")
        return 0

    failures = 0
    for url, target in pending:
        try:
            payload = run.image_bytes(url)
            target.write_bytes(payload)
            print(f"  {target.name}  {len(payload) / 1024:.0f} KiB")
        except Exception as error:  # noqa: BLE001 - one page must not stop the run
            failures += 1
            print(f"  {target.name}  FAILED: {error}")
        finally:
            if delay:
                time.sleep(delay)
    if failures:
        # A chapter missing pages still looks complete to the reader, so say so
        # rather than leaving a hole that only surfaces in a bad recap.
        print(f"  {failures} page(s) failed; re-run to retry them.")
    return failures


def scrape_chapters(
    run: Run,
    out_dir: Path,
    plan: ChapterPlan,
    overwrite: bool,
    delay: float,
    dry_run: bool,
) -> int:
    """Fetch the planned chapters; returns how many chapters were handled."""

    out_dir.mkdir(parents=True, exist_ok=True)
    done = failures = 0

    if plan.walk:
        chapter: int | None = plan.first
        remaining: "iter[int]" = iter(())
    else:
        chapter = plan.numbers[0] if plan.numbers else None
        remaining = iter(plan.numbers[1:])

    while chapter is not None:
        html = run.html(chapter)
        if html is None:
            if plan.walk:
                print(f"\nChapter {chapter} does not exist: end of the series.")
                break
            print(f"Chapter {chapter}: no such chapter on the site; skipping.")
            chapter = next(remaining, None)
            continue

        page = parse_chapter_page(html, run.series_url, chapter)
        if not page:
            if plan.walk:
                print(f"\nChapter {chapter} has no reader images: end of the series.")
                break
            print(f"Chapter {chapter}: no reader images; skipping.")
            chapter = next(remaining, None)
            continue

        failures += download_chapter(run, page, out_dir, overwrite, delay, dry_run)
        done += 1

        if not plan.walk:
            chapter = next(remaining, None)
            continue

        nxt = next_chapter(html, run.series_url, chapter)
        if nxt is None:
            print(f"\nChapter {chapter} is the last one reachable.")
            break
        if done >= _MAX_WALK:
            print(f"\nStopped after {_MAX_WALK} chapters; re-run to continue.")
            break
        chapter = nxt

    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download comic chapters from comic-site.test as page images."
    )
    parser.add_argument("--slug", default=None,
                        help="Series slug in the site URL (default: --name)")
    parser.add_argument("--name", "-n", required=True,
                        help="Local name, e.g. TBATE (folder under the comics dir)")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--chapters", "-c", default=None,
                        help="Chapter selection, e.g. '1-5,8' (site numbering)")
    target.add_argument("--all", action="store_true",
                        help="Walk from --from until the series runs out")
    parser.add_argument("--from", dest="first", type=int, default=1,
                        help="First chapter when no -c is given (default: 1)")
    parser.add_argument("--to", dest="last", type=int, default=None,
                        help="Last chapter when no -c is given")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download images that already exist")
    parser.add_argument("--delay", type=float, default=0.4,
                        help="Seconds between image downloads (default: 0.4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be fetched, then stop")
    args = parser.parse_args(argv)

    config = load_config()
    comics_root = REPO_ROOT / config.get("comics_dir", "comics")
    out_dir = comics_root / args.name
    series_url = f"{SITE}/manga/{args.slug or args.name}/"

    plan = plan_chapters(args.chapters, args.first, args.last, args.all)
    if not plan:
        print("No chapters selected.")
        return 1

    print(f"Series:  {series_url}")
    print(f"Output:  {out_dir}")
    print(f"Chapters: {_describe(plan)}")

    run = Run(series_url=series_url, session=open_session())

    try:
        done = scrape_chapters(run, out_dir, plan, args.overwrite, args.delay,
                               args.dry_run)
    except ScrapeError as error:
        print(f"\nStopped: {error}")
        return 1

    if done:
        print(f"\nRead them with:  uv run python src/comic_reader.py "
              f"-s {args.name} -c comics/{args.name}")
    return 0


def _describe(plan: ChapterPlan) -> str:
    if plan.walk:
        return "walk from --from until the series ends"
    numbers = plan.numbers
    if len(numbers) <= 6:
        return ", ".join(str(n) for n in numbers)
    return f"{numbers[0]}-{numbers[-1]} ({len(numbers)} chapters)"


if __name__ == "__main__":
    sys.exit(main())
