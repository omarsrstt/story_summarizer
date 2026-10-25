"""Cloudflare session handling for the fast scraper.

A single off-screen browser visit passes Cloudflare and yields (a) the
``cf_clearance`` cookie plus the browser's User-Agent and (b) the full chapter
list. Both are cached together so later runs can fetch chapters over plain HTTP
without opening a browser.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


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
