import scraper_fast as fast


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
    "content_selector": "div.txt",
    "chapter_title_selector": "div.top span.chapter",
}


def test_run_fast_writes_one_chapter(tmp_path, monkeypatch):
    monkeypatch.setattr(fast, "make_session", lambda **k: _FakeSession(HTML))
    written = fast.run_fast(
        {"save_dir": "novels"}, SITE, "novel", "1", str(tmp_path), False, 2, False, tmp_path
    )
    assert written == 1
    assert (tmp_path / "novel" / "chapter_1.txt").exists()
    assert "word" in (tmp_path / "novel" / "chapter_1.txt").read_text(encoding="utf-8")


def test_run_fast_skips_implausible_body(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fast, "make_session", lambda **k: _FakeSession("<html>Just a moment...</html>")
    )
    assert (
        fast.run_fast(
            {"save_dir": "novels"}, SITE, "novel", "1", str(tmp_path), False, 2, False, tmp_path
        )
        == 0
    )


def test_run_fast_reports_unmapped_numbers_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(fast, "make_session", lambda **k: _FakeSession(HTML))
    # chapter 99 is not in the map, so nothing is fetched or written.
    assert (
        fast.run_fast(
            {"save_dir": "novels"}, SITE, "novel", "99", str(tmp_path), False, 2, False, tmp_path
        )
        == 0
    )
    assert not (tmp_path / "novel").exists()


def test_chapter_url_joins_base_and_slug():
    assert fast.chapter_url(SITE, "novel", "chapter-530-end-story") == (
        "https://x/book/novel/chapter-530-end-story"
    )
