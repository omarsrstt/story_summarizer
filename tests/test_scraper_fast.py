import pytest

import scraper_fast as fast


def test_transport_retries_on_5xx_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def raw(url, headers, cookies):
        calls["n"] += 1
        return (503, "busy") if calls["n"] < 3 else (200, "<html>ok</html>")

    monkeypatch.setattr(fast, "_raw_transport", raw)
    monkeypatch.setattr(fast.time, "sleep", lambda *_: None)
    status, body = fast.default_transport("u", {}, {})
    assert (status, body) == (200, "<html>ok</html>")
    assert calls["n"] == 3


def test_transport_does_not_retry_403_challenge(monkeypatch):
    calls = {"n": 0}

    def raw(url, headers, cookies):
        calls["n"] += 1
        return (403, "<html>Just a moment... but a real challenge</html>")

    monkeypatch.setattr(fast, "_raw_transport", raw)
    monkeypatch.setattr(fast.time, "sleep", lambda *_: None)
    assert fast.default_transport("u", {}, {})[0] == 403
    assert calls["n"] == 1


def test_transport_retries_429_then_succeeds(monkeypatch):
    """Cloudflare rate-limits bursts with 429 + Retry-After; a short page burst
    must back off and continue rather than fail or truncate the chapter list."""
    calls = {"n": 0}
    slept = []

    def raw(url, headers, cookies):
        calls["n"] += 1
        return (429, "Access denied") if calls["n"] < 3 else (200, "<html>ok</html>")

    monkeypatch.setattr(fast, "_raw_transport", raw)
    monkeypatch.setattr(fast.time, "sleep", lambda s: slept.append(s))

    status, body = fast.default_transport("u", {}, {})
    assert (status, body) == (200, "<html>ok</html>")
    assert calls["n"] == 3
    assert slept  # backed off between attempts


class _FakeSession:
    def __init__(self, html):
        self.html = html

    def fetch(self, url):
        return self.html

    def chapter_map(self):
        return {1: fast.ChapterRef(1, "chapter-2", "Chapter 1: X")}


HTML = (
    '<div class="top"><span class="chapter">Chapter 1: X</span></div>'
    '<div class="txt">' + ("word " * 200) + "</div>"
)
SITE = {
    "base_url": "https://x/",
    "fast_chapter_url": "https://x/book/{novel_slug}/{slug}",
    "fast_seed_slug": "chapter-2",
    "content_selector": "div.txt",
    "chapter_title_selector": "div.top span.chapter",
}


def test_run_fast_writes_one_chapter(tmp_path, monkeypatch):
    monkeypatch.setattr(fast, "make_session", lambda **k: _FakeSession(HTML))
    written = fast.run_fast(
        SITE, "novel", "1", str(tmp_path), False, 2, False, tmp_path
    )
    assert written == 1
    path = tmp_path / "novel" / "chapter_00001.txt"
    assert path.exists()
    assert "word" in path.read_text(encoding="utf-8")


def test_run_fast_skips_implausible_body(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fast, "make_session", lambda **k: _FakeSession("<html>Just a moment...</html>")
    )
    assert (
        fast.run_fast(SITE, "novel", "1", str(tmp_path), False, 2, False, tmp_path)
        == 0
    )


def test_run_fast_reports_unmapped_numbers_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(fast, "make_session", lambda **k: _FakeSession(HTML))
    # chapter 99 is not in the map, so nothing is fetched or written.
    assert (
        fast.run_fast(SITE, "novel", "99", str(tmp_path), False, 2, False, tmp_path)
        == 0
    )
    assert not (tmp_path / "novel").exists()


def test_run_fast_delay_is_passed_to_fetch(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(fast, "make_session", lambda **k: _FakeSession(HTML))
    monkeypatch.setattr(fast.time, "sleep", lambda s: slept.append(s))
    fast.run_fast(SITE, "novel", "1", str(tmp_path), False, 1, False, tmp_path, delay=0.5)
    assert slept == [0.5]


def test_chapter_url_uses_config_template():
    site = {"fast_chapter_url": "https://x/book/{novel_slug}/{slug}"}
    assert fast.chapter_url(site, "novel", "chapter-530-end-story") == (
        "https://x/book/novel/chapter-530-end-story"
    )
