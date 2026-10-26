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

from scraper_parse import CHALLENGE_MARKER, ChapterRef, build_chapter_map


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


_CHALLENGE_MARKER = CHALLENGE_MARKER


def _is_challenge(body: str) -> bool:
    return _CHALLENGE_MARKER in body.lower()


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

    def fetch(self, url: str) -> str:
        """Return the page body, refreshing cookies once if challenged."""

        data = self._ensure_data()
        status, body = self._get(url, data)
        if _is_challenge(body):
            data = self._refresh(data)
            status, body = self._get(url, data)
            if _is_challenge(body):
                raise ChallengeError(f"challenge persisted for {url}")
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


def default_driver_factory(chrome_major: int):
    """Build an off-screen, headful undetected-chromedriver instance.

    Off-screen (``--window-position=-32000,-32000``) so it never covers the
    screen or takes focus, and headful (``headless=False``) because Cloudflare
    blocks headless Chrome. ``version_main`` is pinned to the installed Chrome
    major version; otherwise undetected-chromedriver downloads a mismatched
    driver and fails to start. Imports are local so importing this module (and
    its tests) never requires the browser stack.
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
    site: dict, novel_slug: str, chrome_major: int, driver_factory=default_driver_factory
) -> SessionData:
    """Open an off-screen browser once to pass Cloudflare and list the chapters.

    Loads a chapter page, reads the User-Agent and cookies, then opens the
    chapter-list dropdown and reads every ``select.select-chapter`` option. The
    browser is always closed, even on error.
    """

    import time

    start_url = site["fast_chapter_url"].format(
        novel_slug=novel_slug, slug=site["fast_seed_slug"]
    )
    icon_selector = site["chapter_list_icon_selector"]
    list_selector = site["chapter_list_selector"]

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


