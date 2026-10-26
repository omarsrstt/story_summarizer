from scraper_session import SessionCache, SessionData


def test_roundtrip(tmp_path):
    cache = SessionCache(tmp_path / "c.json")
    assert cache.load() is None
    cache.save(
        SessionData("UA", {"cf_clearance": "v"}, [("chapter-2", "Chapter 1: X")])
    )
    loaded = cache.load()
    assert loaded.user_agent == "UA"
    assert loaded.cookies["cf_clearance"] == "v"
    assert loaded.chapter_map == [("chapter-2", "Chapter 1: X")]


def test_defaults_are_empty(tmp_path):
    data = SessionData("UA")
    assert data.cookies == {}
    assert data.chapter_map == []


def test_corrupt_file_loads_none(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{ not json", encoding="utf-8")
    assert SessionCache(p).load() is None


import pytest

from scraper_parse import ChapterRef
from scraper_session import ChallengeError, FastSession


def _session(transport, harvest, tmp_path):
    return FastSession(
        {"base_url": "https://x/"},
        "novel",
        SessionCache(tmp_path / "c.json"),
        transport,
        harvest,
    )


def test_fetch_returns_body(tmp_path):
    s = _session(
        lambda u, h, c: (200, "<html>ok</html>"),
        lambda: SessionData("UA", {}, []),
        tmp_path,
    )
    assert s.fetch("https://x/book/novel/chapter-2") == "<html>ok</html>"


def _seeded_session(transport, harvest, tmp_path):
    cache = SessionCache(tmp_path / "c.json")
    cache.save(SessionData("UA", {"cf_clearance": "old"}, []))
    return FastSession({"base_url": "https://x/"}, "novel", cache, transport, harvest)


def test_fetch_refreshes_once_then_succeeds(tmp_path):
    calls = {"harvest": 0, "n": 0}

    def transport(u, h, c):
        calls["n"] += 1
        return (200, "Just a moment...") if calls["n"] == 1 else (200, "<html>real</html>")

    def harvest():
        calls["harvest"] += 1
        return SessionData("UA2", {"cf_clearance": "new"}, [])

    s = _seeded_session(transport, harvest, tmp_path)
    assert s.fetch("https://x/book/novel/chapter-2") == "<html>real</html>"
    assert calls["harvest"] == 1


def test_fetch_gives_up_after_one_refresh(tmp_path):
    calls = {"harvest": 0}

    def harvest():
        calls["harvest"] += 1
        return SessionData("UA", {"cf_clearance": "new"}, [])

    s = _seeded_session(lambda u, h, c: (403, "Just a moment..."), harvest, tmp_path)
    with pytest.raises(ChallengeError):
        s.fetch("https://x/book/novel/chapter-2")
    assert calls["harvest"] == 1


def test_empty_cache_harvests_once_before_first_fetch(tmp_path):
    calls = {"harvest": 0}

    def harvest():
        calls["harvest"] += 1
        return SessionData("UA", {"cf_clearance": "v"}, [])

    s = _session(lambda u, h, c: (200, "<html>ok</html>"), harvest, tmp_path)
    assert s.fetch("https://x/book/novel/chapter-2") == "<html>ok</html>"
    assert calls["harvest"] == 1


def test_fetch_does_not_swallow_non_challenge_body(tmp_path):
    s = _session(
        lambda u, h, c: (404, "<html>not found</html>"),
        lambda: SessionData("UA", {}, []),
        tmp_path,
    )
    assert "not found" in s.fetch("https://x/book/novel/chapter-2")


def test_fetch_sends_cached_user_agent(tmp_path):
    seen = {}

    def transport(u, h, c):
        seen["ua"] = h.get("User-Agent")
        seen["cookies"] = c
        return (200, "<html>ok</html>")

    cache = SessionCache(tmp_path / "c.json")
    cache.save(SessionData("CachedUA", {"cf_clearance": "v"}, []))
    s = FastSession({"base_url": "https://x/"}, "novel", cache, transport, lambda: None)
    s.fetch("https://x/book/novel/chapter-2")
    assert seen["ua"] == "CachedUA"
    assert seen["cookies"] == {"cf_clearance": "v"}


def test_chapter_map_builds_from_harvested_options(tmp_path):
    s = _session(
        lambda u, h, c: (200, "<html>ok</html>"),
        lambda: SessionData("UA", {}, [("chapter-2", "Chapter 1: X")]),
        tmp_path,
    )
    assert s.chapter_map() == {1: ChapterRef(1, "chapter-2", "Chapter 1: X")}


class _StubOption:
    def __init__(self, v, t):
        self._v, self.text = v, t

    def get_attribute(self, _):
        return self._v


class _StubSelect:
    def __init__(self, opts):
        self._opts = opts

    def find_elements(self, by, sel):
        return self._opts


class _StubDriver:
    def __init__(self):
        self.quit_called = False

    def get(self, url):
        pass

    def execute_script(self, js, *args):
        return "UA"

    def get_cookies(self):
        return [{"name": "cf_clearance", "value": "abc"}]

    def find_elements(self, by, sel):
        if "select-chapter" in sel:
            return [_StubSelect([_StubOption("chapter-2", "Chapter 1: X")])]
        return [object()]

    def quit(self):
        self.quit_called = True


def test_harvest_reads_cookies_ua_and_options(tmp_path):
    from scraper_session import harvest_session

    driver = _StubDriver()
    site = {
        "base_url": "https://x/",
        "chapter_list_icon_selector": "i",
        "chapter_list_selector": "select.select-chapter",
    }
    data = harvest_session(site, "novel", 155, driver_factory=lambda major: driver)
    assert data.user_agent == "UA"
    assert data.cookies["cf_clearance"] == "abc"
    assert data.chapter_map == [("chapter-2", "Chapter 1: X")]
    assert driver.quit_called is True


def test_harvest_quits_browser_even_on_error(tmp_path):
    from scraper_session import harvest_session

    class _Boom(_StubDriver):
        def get(self, url):
            raise RuntimeError("boom")

    driver = _Boom()
    site = {
        "base_url": "https://x/",
        "chapter_list_icon_selector": "i",
        "chapter_list_selector": "select.select-chapter",
    }
    try:
        harvest_session(site, "novel", 155, driver_factory=lambda major: driver)
    except RuntimeError:
        pass
    assert driver.quit_called is True


def test_harvest_waits_for_icon_before_clicking(tmp_path, monkeypatch):
    import time

    from scraper_session import harvest_session

    monkeypatch.setattr(time, "sleep", lambda *_: None)

    class _SlowDriver(_StubDriver):
        def __init__(self):
            super().__init__()
            self.icon_calls = 0
            self.clicked = False

        def find_elements(self, by, sel):
            if sel == "i":  # the icon selector: not present at first
                self.icon_calls += 1
                return [object()] if self.icon_calls >= 3 else []
            if "select-chapter" in sel:
                return (
                    [_StubSelect([_StubOption("chapter-2", "Chapter 1: X")])]
                    if self.clicked
                    else []
                )
            return []

        def execute_script(self, js, *args):
            if "click" in js:
                self.clicked = True
            return "UA"

    driver = _SlowDriver()
    site = {
        "base_url": "https://x/",
        "chapter_list_icon_selector": "i",
        "chapter_list_selector": "select.select-chapter",
    }
    data = harvest_session(site, "novel", 155, driver_factory=lambda major: driver)
    assert data.chapter_map == [("chapter-2", "Chapter 1: X")]


