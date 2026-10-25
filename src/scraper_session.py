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

from scraper_parse import ChapterRef, build_chapter_map


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


_CHALLENGE_MARKER = "just a moment"


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
            self._data = self._harvest()
            self.cache.save(self._data)
        return self._data

    def _refresh(self) -> SessionData:
        """Refresh the harvest, collapsing concurrent refreshes into one."""

        with self._refresh_lock:
            self._data = self._harvest()
            self.cache.save(self._data)
            return self._data

    def _get(self, url: str, data: SessionData) -> tuple[int, str]:
        headers = {"User-Agent": data.user_agent} if data.user_agent else {}
        return self._transport(url, headers, dict(data.cookies))

    def fetch(self, url: str) -> str:
        """Return the page body, refreshing cookies once if challenged."""

        data = self._ensure_data()
        status, body = self._get(url, data)
        if _is_challenge(body):
            data = self._refresh()
            status, body = self._get(url, data)
            if _is_challenge(body):
                raise ChallengeError(f"challenge persisted for {url}")
        return body

    def chapter_map(self) -> dict[int, ChapterRef]:
        """Return the number→ChapterRef map, harvesting it if not cached."""

        return build_chapter_map(self._ensure_data().chapter_map)

