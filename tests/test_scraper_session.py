from scraper_session import SessionCache, SessionData

import pytest

from scraper_parse import ChapterRef
from scraper_session import ChallengeError, FastSession

# A genuine Cloudflare interstitial: the visible title plus the challenge config
# script. Only the structural markers identify it — the title phrase alone occurs
# in chapter prose.
CHALLENGE = (
    "<html><head><title>Just a moment...</title></head>"
    "<body><script>window._cf_chl_opt={};</script></body></html>"
)


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
        return (200, CHALLENGE) if calls["n"] == 1 else (200, "<html>real</html>")

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

    s = _seeded_session(lambda u, h, c: (403, CHALLENGE), harvest, tmp_path)
    with pytest.raises(ChallengeError):
        s.fetch("https://x/book/novel/chapter-2")
    assert calls["harvest"] == 1


def test_concurrent_challenge_refreshes_once(tmp_path):
    # A stale cookie challenges every in-flight request at once; the refresh must
    # be collapsed so only ONE off-screen harvest happens, not one per worker.
    import threading

    calls = {"harvest": 0}
    barrier = threading.Barrier(4)

    def transport(u, h, c):
        barrier.wait()  # keep all four requests in flight together
        if c.get("cf_clearance") == "old":
            return (200, CHALLENGE)
        return (200, "<html>real</html>")

    def harvest():
        calls["harvest"] += 1
        return SessionData("UA2", {"cf_clearance": "new"}, [])

    s = _seeded_session(transport, harvest, tmp_path)
    results = []

    def do_fetch():
        results.append(s.fetch("https://x/book/novel/chapter-2"))

    threads = [threading.Thread(target=do_fetch) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == ["<html>real</html>"] * 4
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


def test_block_detection_ignores_prose_saying_access_denied(tmp_path):
    """\"access denied\" is ordinary English; a chapter using it must not be
    rejected as a Cloudflare block page."""
    body = (
        "<html><body><div id='content'><p>Her request was met with access "
        "denied, and the gates stayed shut.</p></div></body></html>"
    )
    s = _seeded_session(lambda u, h, c: (200, body), lambda: None, tmp_path)
    assert s.fetch("https://x/book/novel/chapter-2") == body


def test_fetch_reports_a_blocked_chapter_clearly(tmp_path):
    """A rate-limited or blocked chapter must say so, not masquerade as
    \"did not look like a chapter\"."""
    blocked = (
        "<html><head><title>Access denied | x used Cloudflare to restrict "
        "access</title></head><body>Access denied</body></html>"
    )
    s = _seeded_session(lambda u, h, c: (429, blocked), lambda: None, tmp_path)
    with pytest.raises(ChallengeError) as exc:
        s.fetch("https://x/book/novel/chapter-2")
    assert "chapter-2" in str(exc.value)


def test_fetch_does_not_treat_prose_as_a_challenge(tmp_path):
    """A real chapter whose prose says "for just a moment" must be returned as-is
    instead of triggering a pointless cookie refresh."""
    calls = {"harvest": 0}
    body = (
        "<html><body><div id='content'><p>I grinned, and for just a moment, "
        "the burden was lessened.</p></div></body></html>"
    )

    def harvest():
        calls["harvest"] += 1
        return SessionData("UA2", {"cf_clearance": "new"}, [])

    s = _seeded_session(lambda u, h, c: (200, body), harvest, tmp_path)
    assert s.fetch("https://x/book/novel/chapter-2") == body
    assert calls["harvest"] == 0


def test_paginated_session_does_not_reharvest_on_challenge(tmp_path):
    """In paginated mode there are no cookies to renew, so a challenged chapter
    must fail fast rather than re-walk the whole chapter index."""
    calls = {"harvest": 0}

    def harvest():
        calls["harvest"] += 1
        return SessionData("", {}, [("chapter-2", "Chapter 1: X")])

    challenge = "<html><title>Just a moment...</title>window._cf_chl_opt={}</html>"
    s = FastSession(
        {"base_url": "https://x/", "harvest": "paginated"},
        "novel",
        SessionCache(tmp_path / "c.json"),
        lambda u, h, c: (403, challenge),
        harvest,
    )
    with pytest.raises(ChallengeError):
        s.fetch("https://x/book/novel/chapter-2")
    # one harvest to build the map; no second one triggered by the challenge
    assert calls["harvest"] == 1


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


def test_refresh_preserves_map_when_harvest_returns_empty(tmp_path):
    # A transient harvest failure (a fresh cookie but no chapter list) must not
    # wipe the good cached map, or every later run finds nothing to fetch.
    def transport(u, h, c):
        return (403, CHALLENGE) if c.get("cf_clearance") == "old" else (200, "<html>real</html>")

    def harvest():
        return SessionData("UA2", {"cf_clearance": "new"}, [])  # empty list

    cache = SessionCache(tmp_path / "c.json")
    cache.save(SessionData("UA", {"cf_clearance": "old"}, [("chapter-2", "Chapter 1: X")]))
    s = FastSession({"base_url": "https://x/"}, "novel", cache, transport, harvest)
    assert s.fetch("https://x/book/novel/chapter-2") == "<html>real</html>"
    assert s.chapter_map() == {1: ChapterRef(1, "chapter-2", "Chapter 1: X")}


def test_empty_cached_map_triggers_reharvest(tmp_path):
    # A cache holding a cookie but no chapter list (e.g. from an earlier bad
    # harvest) must re-harvest instead of reporting "nothing to fetch" forever.
    calls = {"harvest": 0}

    def harvest():
        calls["harvest"] += 1
        return SessionData("UA", {"cf_clearance": "v"}, [("chapter-2", "Chapter 1: X")])

    cache = SessionCache(tmp_path / "c.json")
    cache.save(SessionData("UA", {"cf_clearance": "old"}, []))
    s = FastSession({"base_url": "https://x/"}, "novel", cache, lambda u, h, c: (200, "ok"), harvest)
    assert s.chapter_map() == {1: ChapterRef(1, "chapter-2", "Chapter 1: X")}
    assert calls["harvest"] == 1


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("156.0.8078.48", 156),
        ("155.0.0.0", 155),
        ("156", 156),
        ("  157.1  ", 157),
        ("", None),
        (None, None),
        ("garbage", None),
        ("Google Chrome", None),
    ],
)
def test_parse_major_from_version(raw, expected):
    from scraper_session import _parse_major

    assert _parse_major(raw) == expected


def test_detect_chrome_major_reads_windows_beacon(monkeypatch):
    import sys

    import scraper_session

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        scraper_session, "_windows_chrome_version", lambda: "156.0.8078.48"
    )
    assert scraper_session.detect_chrome_major() == 156


def test_detect_chrome_major_uses_posix_reader_off_windows(monkeypatch):
    import sys

    import scraper_session

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        scraper_session, "_posix_chrome_version", lambda: "Google Chrome 156.0.1.2"
    )
    assert scraper_session.detect_chrome_major() == 156


def test_detect_chrome_major_none_when_unknown(monkeypatch):
    import sys

    import scraper_session

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(scraper_session, "_windows_chrome_version", lambda: None)
    assert scraper_session.detect_chrome_major() is None


def test_harvest_detects_chrome_major_when_not_supplied(monkeypatch):
    import scraper_session
    from scraper_session import harvest_session

    monkeypatch.setattr(scraper_session, "detect_chrome_major", lambda: 156)
    seen = {}

    def factory(major):
        seen["major"] = major
        return _StubDriver()

    site = {
        "base_url": "https://x/",
        "fast_chapter_url": "https://x/book/{novel_slug}/{slug}",
        "fast_seed_slug": "chapter-2",
        "chapter_list_icon_selector": "i",
        "chapter_list_selector": "select.select-chapter",
    }
    harvest_session(site, "novel", driver_factory=factory)
    assert seen["major"] == 156


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


def test_harvest_looks_up_icon_with_css_selector(tmp_path):
    from scraper_session import harvest_session

    class _RecDriver(_StubDriver):
        def __init__(self):
            super().__init__()
            self.icon_by = None

        def find_elements(self, by, sel):
            if sel == ".the-icon":
                self.icon_by = by
                return [object()]
            if "select-chapter" in sel:
                return [_StubSelect([_StubOption("chapter-2", "Chapter 1: X")])]
            return []

    driver = _RecDriver()
    site = {
        "base_url": "https://x/",
        "fast_chapter_url": "https://x/book/{novel_slug}/{slug}",
        "fast_seed_slug": "chapter-2",
        "chapter_list_icon_selector": ".the-icon",
        "chapter_list_selector": "select.select-chapter",
    }
    harvest_session(site, "novel", 155, driver_factory=lambda major: driver)
    assert driver.icon_by == "css selector"


def test_harvest_reads_cookies_ua_and_options(tmp_path):
    from scraper_session import harvest_session

    driver = _StubDriver()
    site = {
        "base_url": "https://x/",
        "fast_chapter_url": "https://x/book/{novel_slug}/{slug}",
        "fast_seed_slug": "chapter-2",
        "chapter_list_icon_selector": "i",
        "chapter_list_selector": "select.select-chapter",
    }
    data = harvest_session(site, "novel", 155, driver_factory=lambda major: driver)
    assert data.user_agent == "UA"
    assert data.cookies["cf_clearance"] == "abc"
    assert data.chapter_map == [("chapter-2", "Chapter 1: X")]
    assert driver.quit_called is True


def test_harvest_uses_config_url_template_and_seed(tmp_path):
    from scraper_session import harvest_session

    driver = _StubDriver()
    seen = {}
    driver.get = lambda url: seen.__setitem__("url", url)
    site = {
        "base_url": "https://x/",
        "fast_chapter_url": "https://x/read/{novel_slug}/{slug}",
        "fast_seed_slug": "start-here",
        "chapter_list_icon_selector": "i",
        "chapter_list_selector": "select.select-chapter",
    }
    harvest_session(site, "novel", 155, driver_factory=lambda major: driver)
    assert seen["url"] == "https://x/read/novel/start-here"


def test_harvest_quits_browser_even_on_error(tmp_path):
    from scraper_session import harvest_session

    class _Boom(_StubDriver):
        def get(self, url):
            raise RuntimeError("boom")

    driver = _Boom()
    site = {
        "base_url": "https://x/",
        "fast_chapter_url": "https://x/book/{novel_slug}/{slug}",
        "fast_seed_slug": "chapter-2",
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
        "fast_chapter_url": "https://x/book/{novel_slug}/{slug}",
        "fast_seed_slug": "chapter-2",
        "chapter_list_icon_selector": "i",
        "chapter_list_selector": "select.select-chapter",
    }
    data = harvest_session(site, "novel", 155, driver_factory=lambda major: driver)
    assert data.chapter_map == [("chapter-2", "Chapter 1: X")]


