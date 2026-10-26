"""Orchestration for the fast, non-intrusive scraper engine.

The browser is used once (off-screen) to pass Cloudflare and list the chapters;
every chapter is then a plain HTTP GET. Existing chapters are skipped unless
``overwrite`` is set, and a response that does not look like a chapter is never
written.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scraper import parse_chapter_input
from scraper_parse import ChapterRef, extract_chapter, looks_like_chapter
from scraper_session import (
    ChallengeError,
    FastSession,
    SessionCache,
    harvest_session,
)
from scraper_store import existing_chapter, write_chapter

__all__ = ["ChapterRef", "chapter_url", "make_session", "run_fast"]

CACHE_DIRNAME = ".scraper_cache"


def chapter_url(site: dict, novel_slug: str, slug: str) -> str:
    """Full chapter URL from the site base, novel slug, and chapter slug."""

    return f"{site['base_url'].rstrip('/')}/book/{novel_slug}/{slug}"


def default_transport(url: str, headers: dict, cookies: dict) -> tuple[int, str]:
    """Fetch ``url`` with curl_cffi impersonating Chrome. Returns (status, body)."""

    from curl_cffi import requests as cffi

    resp = cffi.get(
        url,
        impersonate="chrome",
        headers=headers,
        cookies=cookies,
        timeout=30,
    )
    return resp.status_code, resp.text


def make_session(
    *, site: dict, novel_slug: str, cache_path, chrome_major: int = 155, refresh: bool = False
) -> FastSession:
    """Build a :class:`FastSession` wired to the real browser harvest and transport.

    Thin seam: tests monkeypatch this to inject a fake session. ``refresh`` drops
    any cached harvest so the next call re-mints it via the browser.
    """

    cache_path = Path(cache_path)
    if refresh:
        cache_path.unlink(missing_ok=True)

    def harvest():
        return harvest_session(site, novel_slug, chrome_major)

    return FastSession(site, novel_slug, SessionCache(cache_path), default_transport, harvest)


def run_fast(
    config: dict,
    site: dict,
    novel_slug: str,
    chapter_sel: str,
    save_dir,
    overwrite: bool,
    concurrency: int,
    refresh: bool,
    repo_root,
) -> int:
    """Scrape the requested chapters with the fast engine; return the count written."""

    repo_root = Path(repo_root)
    root = Path(save_dir)
    if not root.is_absolute():
        root = repo_root / root
    cache_path = repo_root / CACHE_DIRNAME / f"{novel_slug}.json"

    session = make_session(
        site=site, novel_slug=novel_slug, cache_path=cache_path, refresh=refresh
    )

    chapter_map = session.chapter_map()
    if not chapter_map:
        print("No chapter list available from the site; nothing to do.")
        return 0

    numbers = parse_chapter_input(chapter_sel, max(chapter_map)) or []
    targets: list[int] = []
    for number in numbers:
        if number not in chapter_map:
            print(f"Chapter {number} is not in the site's chapter list; skipping.")
            continue
        if not overwrite and existing_chapter(root, novel_slug, number) is not None:
            print(f"Chapter {number} already exists; skipping (--overwrite to replace).")
            continue
        targets.append(number)

    if not targets:
        return 0

    content_selector = site["content_selector"]
    title_selector = site["chapter_title_selector"]

    def fetch_one(number: int) -> tuple[int, str, str]:
        ref = chapter_map[number]
        body = session.fetch(chapter_url(site, novel_slug, ref.slug))
        title, text = extract_chapter(body, content_selector, title_selector)
        return number, title, text

    written = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(fetch_one, number): number for number in targets}
        for future in as_completed(futures):
            number = futures[future]
            try:
                number, title, text = future.result()
            except ChallengeError as exc:
                print(f"Chapter {number}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - one bad chapter must not stop the run
                print(f"Chapter {number}: fetch failed: {exc}")
                continue

            if not looks_like_chapter(text):
                print(f"Chapter {number}: response did not look like a chapter; not saved.")
                continue

            saved = write_chapter(root, novel_slug, number, title, text, overwrite=overwrite)
            if saved is not None:
                written += 1
                print(f"Saved chapter {number} to {saved}")

    return written
