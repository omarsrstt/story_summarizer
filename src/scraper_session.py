"""Cloudflare session handling for the fast scraper.

A single off-screen browser visit passes Cloudflare and yields (a) the
``cf_clearance`` cookie plus the browser's User-Agent and (b) the full chapter
list. Both are cached together so later runs can fetch chapters over plain HTTP
without opening a browser.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from scraper_parse import ChapterRef, build_chapter_map, looks_like_challenge


class ChallengeError(Exception):
    """Raised when a Cloudflare challenge persists after a single cookie refresh."""


@dataclass
class SessionData:
    """Cookies + User-Agent to impersonate, plus the raw ``(slug, title)`` list."""

    user_agent: str
    cookies: dict[str, str] = field(default_factory=dict)
    chapter_map: list[tuple[str, str]] = field(default_factory=list)


class SessionCache:
    """JSON-backed store for a single ``SessionData``."""

    def __init__(self, path):
        self.path = Path(path)

    def load(self) -> SessionData | None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return SessionData(
            user_agent=raw.get("user_agent", ""),
            cookies=dict(raw.get("cookies", {})),
            chapter_map=[tuple(pair) for pair in raw.get("chapter_map", [])],
        )

    def save(self, data: SessionData) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "user_agent": data.user_agent,
            "cookies": data.cookies,
            "chapter_map": [list(pair) for pair in data.chapter_map],
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _is_challenge(body: str) -> bool:
    """Structural challenge test (see ``looks_like_challenge``).

    Deliberately not a "just a moment" substring match: chapter prose contains
    that phrase and must never be mistaken for a challenge page.
    """

    return looks_like_challenge(body)


# Cloudflare's block page ("Access denied | <host> used Cloudflare to restrict
# access") carries none of the interactive-challenge markers, so it needs its own
# test. It is a hard refusal: never a chapter, and never the end of a chapter list.
#
# Only Cloudflare-specific phrasing is listed. A bare "access denied" is ordinary
# English and could appear in chapter prose — the same trap as "just a moment".
_BLOCK_MARKERS = (
    "used cloudflare to restrict access",
    "attention required! | cloudflare",
)


def _is_blocked(body: str) -> bool:
    lowered = body.lower()
    return any(marker in lowered for marker in _BLOCK_MARKERS)


def _parse_major(raw) -> int | None:
    """First integer in a Chrome version string (``"Google Chrome 156.0.8.48"`` -> ``156``)."""

    import re

    if not raw:
        return None
    match = re.search(r"\d+", str(raw))
    return int(match.group()) if match else None


def _windows_chrome_version() -> str | None:
    """Installed Chrome version from its update beacon (``BLBeacon\\version``)."""

    import winreg

    keys = [
        (winreg.HKEY_CURRENT_USER, r"Software\Google\Chrome\BLBeacon"),
        (winreg.HKEY_LOCAL_MACHINE, r"Software\Google\Chrome\BLBeacon"),
    ]
    for hive, path in keys:
        try:
            with winreg.OpenKey(hive, path) as key:
                return winreg.QueryValueEx(key, "version")[0]
        except OSError:
            continue
    return None


def _posix_chrome_version() -> str | None:
    """Installed Chrome version from ``google-chrome --version`` (best effort)."""

    import subprocess

    for exe in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        try:
            out = subprocess.run(
                [exe, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if out:
            return out.strip()
    return None


def detect_chrome_major() -> int | None:
    """Major version of the installed Chrome, or ``None`` if it cannot be found.

    Pinning ``undetected_chromedriver`` to a wrong major makes it download a
    mismatched driver and fail to start, so the pin must follow the installed
    browser. When detection fails we pass ``None`` and let the library
    auto-detect rather than guess.
    """

    import sys

    raw = _windows_chrome_version() if sys.platform == "win32" else _posix_chrome_version()
    return _parse_major(raw)


class FastSession:
    """Authorised HTTP session backed by a cached harvest.

    ``transport(url, headers, cookies) -> (status, body)`` performs the HTTP call
    (a seam so tests need no network). ``harvest() -> SessionData`` mints a fresh
    cookie + chapter list via the browser. When a response is a Cloudflare
    challenge, the cookies are refreshed **once** and the request retried; a
    second challenge raises ``ChallengeError``.
    """

    def __init__(self, site, novel_slug, cache, transport, harvest):
        self.site = site
        self.novel_slug = novel_slug
        self.cache = cache
        self._transport = transport
        self._harvest = harvest
        self._refresh_lock = threading.Lock()
        self._data = cache.load()

    def _ensure_data(self) -> SessionData:
        if self._data is None:
            self._store_harvest(None)
        return self._data

    def _store_harvest(self, previous: SessionData | None) -> SessionData:
        """Harvest, preserve a good chapter list if the new one is empty, and save."""

        fresh = self._harvest()
        if previous is not None and not fresh.chapter_map and previous.chapter_map:
            fresh = SessionData(fresh.user_agent, fresh.cookies, previous.chapter_map)
        self._data = fresh
        self.cache.save(self._data)
        return self._data

    def _refresh(self, stale: SessionData) -> SessionData:
        """Refresh the harvest, collapsing concurrent refreshes into one.

        ``stale`` is the :class:`SessionData` whose cookie was challenged. If
        another worker has already replaced it while we waited for the lock, that
        newer data is reused instead of harvesting again.
        """

        with self._refresh_lock:
            if self._data is not stale:
                return self._data
            return self._store_harvest(stale)

    def _get(self, url: str, data: SessionData) -> tuple[int, str]:
        headers = {"User-Agent": data.user_agent} if data.user_agent else {}
        return self._transport(url, headers, dict(data.cookies))

    def _can_refresh(self) -> bool:
        """Whether a cookie refresh can plausibly help.

        In paginated (plain-HTTP) mode there are no cookies to renew, so
        re-walking the chapter index cannot fix a challenged request; failing
        fast avoids fetching the whole list again for every blocked chapter.
        """

        return self.site.get("harvest") != "paginated"

    def fetch(self, url: str) -> str:
        """Return the page body, refreshing cookies once if challenged."""

        data = self._ensure_data()
        status, body = self._get(url, data)
        if _is_challenge(body):
            if not self._can_refresh():
                raise ChallengeError(
                    f"challenge persisted for {url}; this site uses plain-HTTP "
                    "harvesting and has no cookies to refresh. If it now "
                    'requires Cloudflare clearance, set "harvest": "browser" in '
                    "its config entry."
                )
            data = self._refresh(data)
            status, body = self._get(url, data)
            if _is_challenge(body):
                raise ChallengeError(f"challenge persisted for {url}")
        elif _is_blocked(body):
            # Reported separately from a challenge: no cookie refresh can help,
            # and it must not be mistaken for chapter text that failed to parse.
            raise ChallengeError(
                f"{url} was refused by Cloudflare (HTTP {status}"
                + (", rate limited" if status == 429 else "")
                + "); retry later or with fewer parallel requests."
            )
        return body

    def chapter_map(self) -> dict[int, ChapterRef]:
        """Return the number→ChapterRef map, harvesting it if not cached.

        A cache that holds a cookie but no chapter list (e.g. from a failed
        harvest) is re-harvested rather than reported as empty forever.
        """

        if not self._ensure_data().chapter_map:
            with self._refresh_lock:
                if not self._data.chapter_map:
                    self._store_harvest(self._data)
        return build_chapter_map(self._data.chapter_map)


def harvest_paginated(site: dict, novel_slug: str, transport) -> SessionData:
    """List chapters over plain HTTP by walking a paginated chapter index.

    For sites whose chapter list is server-rendered and public (no Cloudflare
    challenge on plain requests). Fetches the configured list page, collects
    ``(slug, title)`` from the configured link selector, and then walks the
    remaining pages. **Never opens a browser.**

    Two walk styles are supported, chosen by the config template:

    * ``chapter_list_page_url`` contains ``{page}`` -> the walk increments the
      page number until a page yields nothing new. Needed for sites that expose
      numbered pages but no "next" link (and that silently repeat their last
      page for out-of-range numbers).
    * otherwise ``{page}`` is absent -> the walk follows the
      ``chapter_list_next_selector`` link until it is exhausted.

    Raises :class:`ChallengeError` — never returns an empty map — when the site
    challenges plain requests, so the operator can switch the site to
    ``harvest: browser`` in the config.
    """

    from urllib.parse import urljoin

    from bs4 import BeautifulSoup

    template = site["chapter_list_page_url"]
    link_selector = site["chapter_list_link_selector"]
    next_selector = site.get("chapter_list_next_selector")
    numbered = "{page}" in template

    first_url = template.format(novel_slug=novel_slug, page=1)
    max_pages = 200  # hard stop: a misconfigured site must not walk forever

    entries: list[tuple[str, str]] = []
    seen_slugs: set[str] = set()
    seen_pages: set[str] = set()
    url: str | None = first_url
    page = 1

    while url and url not in seen_pages and page <= max_pages:
        seen_pages.add(url)
        status, body = transport(url, {}, {})
        if status != 200 or _is_challenge(body) or _is_blocked(body):
            raise ChallengeError(
                f"chapter list for {novel_slug} at {url} could not be read "
                f"(HTTP {status}"
                + (", Cloudflare rate limit" if status == 429 else "")
                + "): plain-HTTP harvesting is unavailable for this site. "
                + (
                    "Wait a minute and retry with fewer chapters per run."
                    if status == 429
                    else 'Set "harvest": "browser" in its config entry to use the '
                    "off-screen browser harvest instead."
                )
            )
        soup = BeautifulSoup(body, "html.parser")
        new_this_page = 0
        for a in soup.select(link_selector):
            href = a.get("href") or ""
            if not href:
                continue
            title = (a.get("title") or a.get_text() or "").strip()
            if not title:
                continue
            slug = href.rstrip("/").rsplit("/", 1)[-1]
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            entries.append((slug, title))
            new_this_page += 1

        if numbered:
            # A page with no new slugs means the index is exhausted: either the
            # page is empty or the site repeated the previous page verbatim.
            if not new_this_page:
                break
            page += 1
            url = template.format(novel_slug=novel_slug, page=page)
        else:
            nxt = soup.select_one(next_selector) if next_selector else None
            url = urljoin(url, nxt.get("href")) if nxt and nxt.get("href") else None

    if not entries:
        raise ChallengeError(
            f"no chapter links found on the chapter list for {novel_slug} "
            f"at {first_url}; check the site's chapter_list_link_selector."
        )

    return SessionData(user_agent="", cookies={}, chapter_map=entries)


def default_driver_factory(chrome_major: int | None):
    """Build an off-screen, headful undetected-chromedriver instance.

    Off-screen (``--window-position=-32000,-32000``) so it never covers the
    screen or takes focus, and headful (``headless=False``) because Cloudflare
    blocks headless Chrome. ``version_main`` is pinned to the installed Chrome
    major version; otherwise undetected-chromedriver downloads a mismatched
    driver and fails to start. ``None`` defers to the library's auto-detection.
    Imports are local so importing this module (and its tests) never requires
    the browser stack.
    """

    import undetected_chromedriver as uc

    options = uc.ChromeOptions()
    options.add_argument("--window-position=-32000,-32000")
    options.add_argument("--window-size=1280,900")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    return uc.Chrome(
        options=options,
        use_subprocess=True,
        version_main=chrome_major,
        headless=False,
    )


def _read_options(driver, selector: str) -> list[tuple[str, str]]:
    """Return ``(value, text)`` for the option-rich ``select-chapter`` element."""

    selects = driver.find_elements("css selector", selector)
    best = None
    for select in selects:
        opts = select.find_elements("tag name", "option")
        if best is None or len(opts) > len(best):
            best = opts
    if not best:
        return []
    result = []
    for opt in best:
        value = opt.get_attribute("value")
        if value:
            result.append((value, opt.text))
    return result


def harvest_session(
    site: dict,
    novel_slug: str,
    chrome_major: int | None = None,
    driver_factory=default_driver_factory,
) -> SessionData:
    """Open an off-screen browser once to pass Cloudflare and list the chapters.

    Loads a chapter page, reads the User-Agent and cookies, then opens the
    chapter-list dropdown and reads every ``select.select-chapter`` option. The
    browser is always closed, even on error. When ``chrome_major`` is omitted the
    installed Chrome version is detected so the driver matches the browser.
    """

    import time

    start_url = site["fast_chapter_url"].format(
        novel_slug=novel_slug, slug=site["fast_seed_slug"]
    )
    icon_selector = site["chapter_list_icon_selector"]
    list_selector = site["chapter_list_selector"]

    if chrome_major is None:
        chrome_major = detect_chrome_major()

    driver = driver_factory(chrome_major)
    try:
        driver.get(start_url)
        user_agent = driver.execute_script("return navigator.userAgent") or ""

        # Wait for the chapter-list icon to render before clicking it; clicking
        # before the page has loaded silently yields no options.
        icon = []
        for _ in range(30):
            icon = driver.find_elements("css selector", icon_selector)
            if icon:
                break
            time.sleep(0.5)
        if icon:
            driver.execute_script("arguments[0].click();", icon[0])

        options: list[tuple[str, str]] = []
        for _ in range(30):
            options = _read_options(driver, list_selector)
            if options:
                break
            time.sleep(0.5)

        cookies = {
            c["name"]: c["value"]
            for c in driver.get_cookies()
            if c.get("name") and c.get("value") is not None
        }
        return SessionData(user_agent, cookies, options)
    finally:
        try:
            driver.quit()
        except Exception:  # noqa: BLE001 - cleanup noise from uc's __del__
            pass


