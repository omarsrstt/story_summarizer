"""Tests for the comic scraper's parsing and run logic.

Network code is not exercised here; these pin the decisions that silently
corrupt a download instead of failing loudly: which images a page yields, which
chapter comes next, what an image is saved as, and whether a blocked or throttled
response can be mistaken for the end of a series.
"""

from pathlib import Path

import pytest

import comic_scraper as cs
from comic_scraper import (
    ChapterPage,
    ChapterPlan,
    Run,
    image_filename,
    next_chapter,
    parse_chapter_page,
    plan_chapters,
    safe_image_url,
    scrape_chapters,
)

SERIES = "https://comic-site.test/manga/the-beginning-after-the-end-series/"

# A trimmed copy of real output: the lazy-load pattern is what matters -- `src`
# is a placeholder and the real address lives in `data-src`.
CHAPTER_HTML = """
<div class="reading-content">
  <div class="page-break ">
    <img id="image-0" alt="Load image..."
         src="https://comic-site.test/wp-content/themes/madara/images/loading6.svg"
         data-src="https://c1.comic-site.test/manga_abc/chapter_0/c_0_1.jpg"
         class="wp-manga-chapter-img img-responsive lazyload effect-fade">
  </div>
  <div class="page-break ">
    <img id="image-1" alt="Load image..."
         src="https://comic-site.test/wp-content/themes/madara/images/loading6.svg"
         data-src="https://c1.comic-site.test/manga_abc/chapter_0/c_0_2.jpg"
         class="wp-manga-chapter-img img-responsive lazyload effect-fade">
  </div>
</div>
<a href="https://comic-site.test/manga/the-beginning-after-the-end-series/chapter-1/">prev</a>
<a href="https://comic-site.test/manga/the-beginning-after-the-end-series/chapter-3/">next</a>
"""

# A real Cloudflare interstitial, trimmed to the markers scraper_parse tests for.
CHALLENGE_HTML = """
<html><head><script>window._cf_chl_opt = {}</script></head>
<body>Enable JavaScript and cookies to continue</body></html>
"""

JPEG = b"\xff\xd8\xff\xe0" + b"0" * 64


# -- the page decides what gets downloaded ---------------------------------


def test_images_are_read_from_data_src_not_the_placeholder():
    page = parse_chapter_page(CHAPTER_HTML, SERIES, chapter=2)
    assert page.images == [
        "https://c1.comic-site.test/manga_abc/chapter_0/c_0_1.jpg",
        "https://c1.comic-site.test/manga_abc/chapter_0/c_0_2.jpg",
    ]


def test_a_page_without_chapter_images_is_empty():
    page = parse_chapter_page("<html><body>Nothing here</body></html>", SERIES, chapter=300)
    assert page.images == []
    assert not page


def test_load_placeholders_are_never_returned():
    page = parse_chapter_page(CHAPTER_HTML, SERIES, chapter=2)
    assert all("loading6.svg" not in url for url in page.images)


def test_an_offsite_image_url_on_a_page_is_skipped():
    html = (
        '<img class="wp-manga-chapter-img" '
        'data-src="https://evil.example.com/banner.jpg">'
    )
    assert parse_chapter_page(html, SERIES, chapter=2).images == []


# -- walking to the next chapter -------------------------------------------


def test_the_next_chapter_is_found_from_the_nav_links():
    assert next_chapter(CHAPTER_HTML, SERIES, chapter=2) == 3


def test_the_last_chapter_has_no_next():
    html = f'<a href="{SERIES}chapter-2/">prev</a>'
    assert next_chapter(html, SERIES, chapter=2) is None


def test_links_to_other_series_are_ignored():
    # A recommendations block links elsewhere; following it would silently
    # download a different comic.
    html = (
        f'<a href="{SERIES}chapter-3/">next</a>'
        '<a href="https://comic-site.test/manga/other-series/chapter-99/">other</a>'
    )
    assert next_chapter(html, SERIES, chapter=2) == 3


# -- naming, so comic_reader groups it correctly ---------------------------


def test_an_image_keeps_the_site_name_that_encodes_chapter_and_page():
    url = "https://c1.comic-site.test/manga_abc/chapter_0/c_0_1.jpg"
    assert image_filename(url) == "c_0_1.jpg"


def test_the_storage_chapter_differs_from_the_reader_chapter():
    # The reader's "chapter 1" is stored as chapter_0. The filename keeps the
    # storage numbering because that is what the existing downloads use and what
    # comic_reader groups on.
    url = "https://c1.comic-site.test/manga_abc/chapter_1/c_1_54.jpg"
    assert image_filename(url) == "c_1_54.jpg"


def test_an_unrecognised_name_is_rejected():
    with pytest.raises(ValueError):
        image_filename("https://c1.comic-site.test/manga_abc/chapter_0/banner.png")


# -- refusing to fetch anything that is not the comic ----------------------


def test_only_the_configured_image_host_is_allowed():
    assert safe_image_url("https://c1.comic-site.test/manga_abc/chapter_0/c_0_1.jpg")
    assert safe_image_url("https://comic-site.test/wp-content/uploads/x.jpg")


def test_an_ad_or_foreign_host_is_refused():
    # These pages carry obfuscated scripts that redirect to ad hosts; an image
    # URL pointing off-site must never be fetched.
    for bad in (
        "https://evil.example.com/tracker.jpg",
        "https://comic-site.test.evil.example/c_0_1.jpg",
        "http://comic-site.test@evil.example/c_0_1.jpg",
        "data:image/jpeg;base64,AAAA",
    ):
        assert not safe_image_url(bad), bad


# -- the record of one chapter ---------------------------------------------


def test_a_chapter_page_is_truthy_only_when_it_has_images():
    assert parse_chapter_page(CHAPTER_HTML, SERIES, chapter=2)
    assert not ChapterPage(chapter=5, images=[])


# -- planning ---------------------------------------------------------------


def test_an_explicit_selection_wins():
    plan = plan_chapters("1-3,7", first=5, last=None, walk=False)
    assert plan.numbers == [1, 2, 3, 7]
    assert not plan.walk


def test_a_range_is_expanded():
    assert plan_chapters(None, first=3, last=5, walk=False).numbers == [3, 4, 5]


def test_no_selection_fetches_a_single_chapter():
    assert plan_chapters(None, first=4, last=None, walk=False).numbers == [4]


def test_all_asks_for_a_walk():
    plan = plan_chapters(None, first=1, last=None, walk=True)
    assert plan.walk
    assert plan


def test_an_empty_selection_is_not_a_plan():
    assert not plan_chapters("", first=1, last=None, walk=False)
    assert not ChapterPlan()


# -- a run must not mistake a block for the end of the series --------------


class FakeSession:
    """Stands in for the curl_cffi session."""

    def __init__(self, pages: dict[str, bytes]):
        self.pages = pages
        self.requests: list[str] = []

    def get(self, url, headers=None, timeout=None):
        self.requests.append(url)

        class Response:
            def __init__(self, status, content):
                self.status_code = status
                self.content = content

        if url not in self.pages:
            return Response(404, b"")
        return Response(200, self.pages[url])


def make_run(pages: dict[str, bytes]) -> Run:
    return Run(series_url=SERIES, session=FakeSession(pages))


def test_a_challenge_page_stops_the_run_instead_of_ending_the_series(monkeypatch):
    # The bug this guards: a challenge carries no images, so treating it as
    # "no such chapter" would truncate the series and still print a clean finish.
    monkeypatch.setattr(cs, "default_transport", lambda url, h, c: (200, CHALLENGE_HTML))
    run = make_run({})

    with pytest.raises(cs.ScrapeError, match="challenge"):
        scrape_chapters(run, Path("unused"), ChapterPlan(walk=True), False, 0, False)


def test_a_throttled_page_is_not_reported_as_the_end_of_the_series(monkeypatch):
    monkeypatch.setattr(cs, "default_transport", lambda url, h, c: (429, ""))
    run = make_run({})

    with pytest.raises(cs.ScrapeError, match="429"):
        scrape_chapters(run, Path("unused"), ChapterPlan(walk=True), False, 0, False)


def test_a_missing_chapter_ends_a_walk_cleanly(monkeypatch, tmp_path):
    monkeypatch.setattr(cs, "default_transport", lambda url, h, c: (404, ""))
    run = make_run({})

    assert scrape_chapters(run, tmp_path, ChapterPlan(walk=True), False, 0, False) == 0


def test_images_are_written_and_existing_ones_are_skipped(monkeypatch, tmp_path):
    url_a = "https://c1.comic-site.test/manga_abc/chapter_0/c_0_1.jpg"
    url_b = "https://c1.comic-site.test/manga_abc/chapter_0/c_0_2.jpg"
    html = f"""
    <img class="wp-manga-chapter-img" data-src="{url_a}">
    <img class="wp-manga-chapter-img" data-src="{url_b}">
    """
    monkeypatch.setattr(cs, "default_transport", lambda url, h, c: (200, html))
    run = make_run({url_a: JPEG, url_b: JPEG})

    scrape_chapters(run, tmp_path, ChapterPlan(numbers=[2]), False, 0, False)
    assert sorted(p.name for p in tmp_path.glob("*.jpg")) == ["c_0_1.jpg", "c_0_2.jpg"]

    # Resume: the second pass downloads nothing.
    run2 = make_run({url_a: JPEG, url_b: JPEG})
    scrape_chapters(run2, tmp_path, ChapterPlan(numbers=[2]), False, 0, False)
    assert run2.session.requests == []


def test_an_html_error_page_is_never_saved_as_an_image(monkeypatch, tmp_path):
    url = "https://c1.comic-site.test/manga_abc/chapter_0/c_0_1.jpg"
    html = f'<img class="wp-manga-chapter-img" data-src="{url}">'
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, html))
    # Status 200 but an HTML body: what Cloudflare and WordPress both do.
    run = make_run({url: b"<html>error</html>"})

    scrape_chapters(run, tmp_path, ChapterPlan(numbers=[2]), False, 0, False)
    assert list(tmp_path.glob("*.jpg")) == []
