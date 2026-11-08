"""List every series a site publishes, without downloading any of them.

This is the "what is even on this site?" stage that the pipeline lacked: the
downloader answers "give me this series", but choosing *which* series to download
needs a catalogue first. Nothing here touches a manifest or an image, so a
listing run cannot disturb a download already in progress.

Where the list comes from
-------------------------
The theme this was built against paginates a catalogue page at ``/manga/page/N/``
-- 12 series a page, cleanly ending in a 404. Two other routes were measured on
the real site and both are wrong to use here:

* **Sitemaps are truncated.** They carry *chapter* URLs, so they are partitioned
  by chapter rather than by series, and the later catalogue pages held series
  absent from every sitemap read. Worse, an out-of-range sitemap number still
  answers 200 with some sitemap's content, so "walk until 404" does not
  terminate there.
* **The REST API does not expose the series post type.** ``wp/v2/types`` lists
  only core types, and ``wp/v2/search`` indexes blog posts only.

The catalogue is therefore the only complete source, and completion is *measured*
rather than assumed: the walk stops at the first page with no series cards, and
that count is what gets reported.

The ordering is not cosmetic
----------------------------
The default catalogue order is "latest update". A walk over it is **unsound**: a
series bumped while the walk is running jumps to the front, every later page
shifts by one card, and the page the walk asks for next now shows what it has
already seen. Measured over pages 1-20 the default order gave 240 cards but only
233 distinct slugs, and a full walk stalled on 14 pages and came out 11,015
series. ``?m_orderby=title`` is a total order: the same 240 cards were 240
distinct, the walk reached the same last page, and nothing stalled. That is why
the shipped ``series_list_url`` carries it.

Dedup and the stall check remain anyway. They cost one set lookup and one retry,
and they are what turns a future ordering change into a warning instead of a
silently short list.

Why a link count is not a series count
--------------------------------------
A catalogue page carries more ``/manga/`` links than cards (17 against 12 on the
theme measured), the extras being sidebar and navigation links. Counting links
over-reports by five on every page, so the item selector points at the grid card
instead.

So `page * 12` is wrong twice over: the sidebar links would be counted as series,
and on an unstable ordering the same series appears on two pages. The walk
therefore keys on the slug and dedups, and it only ever reports a count it
actually collected. Consequently a run starting past page 1 is a *partial*
listing, not the catalogue, and is written as such.

Output
------
One JSON file per site under ``comics/_catalog/``, gitignored with the rest of
``comics/`` because every entry holds a series URL and therefore the host. The
file is written atomically and checkpointed during the walk, so an interrupted
run leaves the previous catalogue intact rather than a half-written file that
still parses -- and if a checkpoint had already replaced it, the run says so
instead of claiming otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

import comic_scraper as cs
from comic_scraper import ScrapeError, Site
from scraper_fast import default_transport
from scraper_parse import looks_like_challenge

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Where the catalogue files live. Under ``comics/`` so it inherits that
#: directory's gitignore rule -- these files carry the host in every URL.
CATALOG_DIR = "comics/_catalog"

#: How many pages may be gathered before the file is rewritten. A full walk of
#: the site measured is ~950 pages, and losing 900 of them to a crash on the last
#: one is the kind of loss a checkpoint costs almost nothing to avoid.
CHECKPOINT_EVERY = 50


@dataclass(frozen=True)
class Series:
    """One row of the catalogue."""

    slug: str
    title: str
    url: str

    def as_dict(self) -> dict:
        return {"slug": self.slug, "title": self.title, "url": self.url}


def parse_catalogue_page(
    html: str, site: Site | None = None
) -> list[tuple[str, str]]:
    """The ``(slug, title)`` pairs on one catalogue page, in page order.

    Pure, so the parsing rule can be tested against a saved page instead of the
    live site. An anchor without a usable href is skipped rather than guessed at:
    a made-up slug would list a series that cannot then be downloaded.

    The slug is taken from the href, not the title: titles repeat across a site
    (``Series`` and ``Series (Season 2)`` both render as ``Series``), while the
    slug is what the downloader actually needs.
    """

    site = site or cs.site_settings()
    soup = BeautifulSoup(html, "html.parser")
    rows: list[tuple[str, str]] = []
    for anchor in soup.select(site.series_item_selector):
        href = (anchor.get("href") or "").strip()
        if not href:
            continue
        path = urlparse(href).path.rstrip("/")
        slug = path.rsplit("/", 1)[-1]
        if not slug:
            continue
        rows.append((slug, anchor.get_text(" ", strip=True)))
    return rows


def fetch_catalogue_page(
    page: int, site: Site, delay: float = 0.0
) -> list[tuple[str, str]]:
    """One catalogue page, or ``[]`` when the pages have run out.

    An empty page and a missing page are both "past the end": the walk scans for
    the last page rather than trusting a fixed total, since the total changes as
    series are published. Anything that is neither a 200 nor a 404 raises -- a
    Cloudflare interstitial or a 5xx would otherwise read as "the catalogue ends
    here" and silently truncate the list, which is the one failure that looks
    exactly like success.
    """

    url = site.catalogue_url(page)
    if delay:
        time.sleep(delay)
    status, body = default_transport(url, {}, {})
    if looks_like_challenge(body):
        raise ScrapeError(
            f"catalogue page {page}: Cloudflare challenge instead of the page. "
            "Wait a minute and retry."
        )
    if status == 404:
        return []
    if status != 200:
        raise ScrapeError(
            f"catalogue page {page}: HTTP {status} at {url}"
            + (", rate limited by Cloudflare" if status == 429 else "")
        )
    return parse_catalogue_page(body, site)


@dataclass(frozen=True)
class Walk:
    """What a catalogue walk collected.

    ``stalled_pages`` is not decoration: a page whose every card was already seen
    means the site served a stale or repeated page, so the series that page
    should have introduced were *missed*. The walk cannot tell which they were,
    so it reports the hole instead of presenting the result as complete.
    """

    series: list[Series]
    last_page: int
    stalled_pages: list[int]


def walk(
    site: Site,
    delay: float = 0.4,
    max_pages: int | None = None,
    first_page: int = 1,
    progress=None,
    checkpoint=None,
    stall_retries: int = 1,
) -> Walk:
    """Walk the catalogue until a page yields nothing.

    Returns the series in page order, the last page read, and any pages that
    stalled. ``max_pages`` caps the walk for a trial run; the caller must report a
    capped walk as partial, because a truncated catalogue is indistinguishable
    from a complete one in the file.

    ``checkpoint`` is called with ``(series, last_page)`` every
    ``CHECKPOINT_EVERY`` pages so a crash late in a long walk does not discard
    the pages already paid for.

    A page of nothing but already-seen slugs is **retried once** before being
    recorded as stalled. The measured cause is the default "latest update"
    ordering: a series bumped mid-walk shifts every later page, so the page the
    walk asks for next returns what it has already seen. A full walk over that
    order stalled on 14 pages and lost their contents. The shipped ordering
    (``?m_orderby=title``) is stable and did not stall, but the check stays as a
    detector: it turns a future ordering change into a warning instead of a
    silently short list. A retry recovers a transient stall; a persistent one is
    recorded, and the caller marks the catalogue incomplete rather than done.
    """

    series: list[Series] = []
    seen: set[str] = set()
    stalled: list[int] = []
    page = first_page
    pages_since_checkpoint = 0
    while True:
        rows = fetch_catalogue_page(page, site, delay=delay)
        if not rows:
            return Walk(series, page - 1, stalled)

        fresh = [(slug, title) for slug, title in rows if slug not in seen]

        # A page that introduces nothing is a hole, not progress. Retry it while
        # retries remain: the content was observed to change between fetches.
        attempts = 0
        while not fresh and attempts < stall_retries:
            attempts += 1
            rows = fetch_catalogue_page(page, site, delay=delay)
            fresh = [(slug, title) for slug, title in rows if slug not in seen]
            if not rows:
                return Walk(series, page - 1, stalled)

        duplicates = len(rows) - len(fresh)
        for slug, title in fresh:
            seen.add(slug)
            series.append(
                Series(slug=slug, title=title, url=site.series_url.format(slug=slug))
            )

        if not fresh:
            # Still nothing new after the retries: record the hole and keep
            # walking, because later pages were observed to resume normally.
            stalled.append(page)

        if progress:
            progress(page, len(rows), len(series), duplicates, bool(stalled) and not fresh)

        pages_since_checkpoint += 1
        if checkpoint and pages_since_checkpoint >= CHECKPOINT_EVERY:
            checkpoint(series, page)
            pages_since_checkpoint = 0

        if max_pages is not None and page >= first_page + max_pages - 1:
            return Walk(series, page, stalled)
        page += 1


def catalog_path(name: str) -> Path:
    """Where the catalogue for site ``name`` is written.

    The name is sanitised because it becomes a filename, and a config is a file a
    user edits: ``../`` in a site name must not choose the write location. A dot
    is not in the allowed set at all -- ``..`` cannot then appear anywhere in the
    stem, rather than relying on a strip to remove it.
    """

    safe = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in name
    ).strip("_") or "site"
    return REPO_ROOT / CATALOG_DIR / f"{safe}.json"


def save(
    path: Path,
    site: Site,
    series: list[Series],
    pages: int,
    complete: bool,
    checkpoint: bool = False,
    stalled_pages: list[int] | None = None,
) -> None:
    """Write the catalogue atomically.

    Written to a sibling temp file and renamed, so a reader never sees a partial
    file. A checkpoint rewrites the whole file rather than appending, which is
    what keeps the JSON valid at every moment.

    ``stalled_pages`` is written even when it is empty, so a reader can tell "no
    holes detected" from "this file predates hole detection".
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": [
            "Generated by src/comic_catalog.py. Do not commit: every entry",
            "carries the site host in its url.",
        ],
        "site": site.name,
        "series_base": site.series_base(),
        "pages": pages,
        "complete": complete,
        "checkpoint": checkpoint,
        "stalled_pages": stalled_pages or [],
        "count": len(series),
        "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "series": [entry.as_dict() for entry in series],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(tmp, path)


def load(path: Path) -> dict:
    """Read a saved catalogue, for a later stage that wants to pick from it."""

    return json.loads(path.read_text(encoding="utf-8"))


def site_label(site: Site) -> str:
    """What to call the site in output, without printing its host.

    The host is the one value this repo must never publish, and terminal output
    is the easiest place for it to end up pasted. The configured *name* says
    which site a run was about; the host adds nothing a user does not know.
    """

    return site.name or "(unnamed)"


def path_label(url: str) -> str:
    """``url`` as a path, for reporting progress without the host."""

    parsed = urlparse(url)
    return parsed.path or "/"


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="comic_catalog.py",
        description=(
            "List every series a site publishes, into "
            f"{CATALOG_DIR}/<site>.json. Writes nothing else and downloads no "
            "comic pages."
        ),
        epilog=(
            "The walk stops at the first catalogue page with no series, so the "
            "count is measured. --pages caps the walk for a trial run and marks "
            "the result incomplete."
        ),
    )
    parser.add_argument(
        "--site", default=None, metavar="NAME",
        help="Which configured site to list (default: its 'default_site')")
    parser.add_argument(
        "--pages", type=int, default=None, metavar="N",
        help="Read at most N catalogue pages (a partial listing)")
    parser.add_argument(
        "--first-page", type=int, default=1, metavar="N",
        help="Start at this catalogue page (default: 1)")
    parser.add_argument(
        "--delay", type=float, default=0.4, metavar="SECONDS",
        help="Pause before each page request (default: 0.4)")
    parser.add_argument(
        "--out", default=None, metavar="PATH",
        help="Write here instead of the default catalogue path")
    args = parser.parse_args(argv)

    if args.pages is not None and args.pages < 1:
        parser.error("--pages must be at least 1")
    if args.first_page < 1:
        parser.error("--first-page must be at least 1")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        site = cs.resolve_site(cs.load_config(), args.site)
        cs.ensure_configured(site)
    except ScrapeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    cs.use_site(site)
    out = Path(args.out) if args.out else catalog_path(site_label(site))
    # Any run that is not a whole-catalogue-from-page-1 walk is partial. A cap is
    # the obvious case; a late start is the subtle one, because the pages it does
    # read are real and the file would otherwise claim to be the whole list.
    capped = args.pages is not None or args.first_page != 1

    print(
        f"listing series for site {site_label(site)!r} "
        f"from {path_label(site.catalogue_url(args.first_page))}"
    )

    def progress(page, on_page, total, duplicates, stalled=False):
        print(
            f"  page {page:4}: {on_page:3} on page, {total:5} series so far"
            + (f", {duplicates} duplicate(s) skipped" if duplicates else "")
            + ("  [STALLED: no new series, retried]" if stalled else "")
        )

    def checkpoint(series, last_page):
        # Marked complete=False and checkpoint=True so a reader can tell a
        # resumed-from file from a finished one.
        save(out, site, series, last_page, complete=False, checkpoint=True)
        print(f"  checkpoint at page {last_page} ({len(series)} series)")

    wrote = {"any": False}

    def checkpoint_and_record(series, last_page, stalled=()):
        checkpoint(series, last_page)
        wrote["any"] = True

    try:
        result = walk(
            site,
            delay=args.delay,
            max_pages=args.pages,
            first_page=args.first_page,
            progress=progress,
            checkpoint=checkpoint_and_record,
        )
    except ScrapeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        # Truthful about what the failure cost: a checkpoint replaces the file,
        # so a long run that dies has already overwritten the previous catalogue
        # with a partial one. Saying "left as it was" there would be a lie.
        if wrote["any"]:
            print(
                f"{out} now holds a PARTIAL catalogue (last checkpoint, marked "
                '"checkpoint": true)',
                file=sys.stderr,
            )
        else:
            print(f"the previous catalogue at {out} was left as it was",
                  file=sys.stderr)
        return 1

    series, last_page, stalled_pages = (
        result.series, result.last_page, result.stalled_pages
    )
    missing_titles = sum(1 for entry in series if not entry.title.strip())
    save(
        out, site, series, last_page,
        complete=not capped and not stalled_pages,
        stalled_pages=stalled_pages,
    )

    print(
        f"\n{len(series)} series across {last_page} page(s)"
        + ("  [PARTIAL: --pages was given]" if args.pages is not None else "")
        + ("  [PARTIAL: started after page 1]" if args.first_page != 1 else "")
    )
    if stalled_pages:
        # A stalled page is a hole the walk cannot fill: the site served a
        # repeated page, so whatever it should have introduced is missing. Saying
        # so is the only honest report -- the count looks plausible either way.
        shown = ", ".join(str(p) for p in stalled_pages[:10])
        more = "" if len(stalled_pages) <= 10 else f" (+{len(stalled_pages) - 10} more)"
        print(
            f"warning: {len(stalled_pages)} page(s) served no new series even "
            f"after a retry: {shown}{more}",
            file=sys.stderr,
        )
        print(
            "         marked incomplete; re-run to fill the holes",
            file=sys.stderr,
        )
    if missing_titles:
        # Not a failure: the listing is still usable by slug. But it means the
        # selector matched anchors outside a title, which is worth saying once.
        print(f"warning: {missing_titles} series had no title text")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
