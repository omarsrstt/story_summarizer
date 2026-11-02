import pytest

import comic_scraper as cs
from comic_manifest import ChapterRecord

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
"""

# A real Cloudflare interstitial, trimmed to the markers scraper_parse tests for.
CHALLENGE_HTML = """
<html><head><script>window._cf_chl_opt = {}</script></head>
<body>Enable JavaScript and cookies to continue</body></html>
"""

JPEG = b"\xff\xd8\xff\xe0" + b"0" * 64

LIST_HTML = """
<li class="wp-manga-chapter"><a href="{s}chapter-177/">Chapter 177</a></li>
<li class="wp-manga-chapter"><a href="{s}chapter-176/">Chapter 176</a></li>
"""

PAGE_HTML = """
<img class="wp-manga-chapter-img" src="loader.svg"
     data-src="https://c4.comic-site.test/manga_x/chapter_276/ch_276_1.jpg">
<img class="wp-manga-chapter-img"
     data-src="https://c4.comic-site.test/manga_x/chapter_276/ch_276_2.webp">
<img class="wp-manga-chapter-img" data-src="https://evil.example/banner.jpg">
"""


# -- the chapter list, in one request --------------------------------------


def test_the_list_is_returned_in_the_sites_own_order(monkeypatch):
    monkeypatch.setattr(
        cs, "post_transport", lambda u, h, c: (200, LIST_HTML.format(s=SERIES))
    )
    assert cs.fetch_chapter_list(SERIES) == [
        ("chapter-177", "Chapter 177"),
        ("chapter-176", "Chapter 176"),
    ]


def test_the_list_request_asks_for_ajax(monkeypatch):
    seen = {}

    def fake(url, headers, cookies):
        seen.update(headers)
        return 200, LIST_HTML.format(s=SERIES)

    monkeypatch.setattr(cs, "post_transport", fake)
    cs.fetch_chapter_list(SERIES)
    assert seen["X-Requested-With"] == "XMLHttpRequest"


def test_a_challenge_where_the_list_should_be_stops_the_run(monkeypatch):
    monkeypatch.setattr(
        cs, "post_transport", lambda u, h, c: (200, CHALLENGE_HTML)
    )
    with pytest.raises(cs.ScrapeError, match="challenge"):
        cs.fetch_chapter_list(SERIES)


def test_a_non_200_list_is_an_error(monkeypatch):
    monkeypatch.setattr(cs, "post_transport", lambda u, h, c: (503, ""))
    with pytest.raises(cs.ScrapeError, match="503"):
        cs.fetch_chapter_list(SERIES)


# -- the page decides what gets downloaded ---------------------------------


def test_images_are_read_from_data_src_not_the_placeholder():
    assert cs.page_urls(CHAPTER_HTML) == [
        "https://c1.comic-site.test/manga_abc/chapter_0/c_0_1.jpg",
        "https://c1.comic-site.test/manga_abc/chapter_0/c_0_2.jpg",
    ]


def test_load_placeholders_are_never_returned():
    assert all("loading6.svg" not in url for url in cs.page_urls(CHAPTER_HTML))


def test_a_page_without_chapter_images_yields_nothing():
    assert cs.page_urls("<html><body>Nothing here</body></html>") == []


def test_an_offsite_image_url_on_a_page_is_skipped():
    assert cs.page_urls(PAGE_HTML) == [
        "https://c4.comic-site.test/manga_x/chapter_276/ch_276_1.jpg",
        "https://c4.comic-site.test/manga_x/chapter_276/ch_276_2.webp",
    ]


def test_every_image_shape_is_read_in_page_order():
    urls = (
        "https://x.comic-site.test/a/c_1_1.jpg",
        "https://x.comic-site.test/a/ch_1_2.webp",
        "https://x.comic-site.test/a/67.webp",
        "https://x.comic-site.test/a/1-(2).jpeg",
        "https://x.comic-site.test/a/ch-259-img-3.jpg",
        "https://x.comic-site.test/a/9.png",
    )
    html = "".join(
        f'<img class="wp-manga-chapter-img" data-src="{u}">' for u in urls
    )
    assert cs.page_urls(html) == list(urls)


def test_a_placeholder_src_without_data_src_yields_nothing():
    html = (
        '<img class="wp-manga-chapter-img" '
        'src="https://comic-site.test/wp-content/themes/madara/images/loading6.svg">'
    )
    assert cs.page_urls(html) == []


def test_a_chapter_url_is_built_from_the_slug():
    assert cs.chapter_url(SERIES, "chapter-175-19") == SERIES + "chapter-175-19/"


# -- refusing to fetch anything that is not the comic ----------------------


def test_only_the_configured_image_host_is_allowed():
    assert cs.safe_image_url("https://c1.comic-site.test/manga_abc/chapter_0/c_0_1.jpg")
    assert cs.safe_image_url("https://comic-site.test/wp-content/uploads/x.jpg")


def test_an_ad_or_foreign_host_is_refused():
    # These pages carry obfuscated scripts that redirect to ad hosts; an image
    # URL pointing off-site must never be fetched.
    for bad in (
        "https://evil.example.com/tracker.jpg",
        "https://comic-site.test.evil.example/c_0_1.jpg",
        "http://comic-site.test@evil.example/c_0_1.jpg",
        "data:image/jpeg;base64,AAAA",
    ):
        assert not cs.safe_image_url(bad), bad


# -- extensions, so a page is saved under one name -------------------------


@pytest.mark.parametrize(
    "url,ext",
    [
        ("https://x.comic-site.test/a/c_1_1.jpg", ".jpg"),
        ("https://x.comic-site.test/a/ch_1_1.webp", ".webp"),
        ("https://x.comic-site.test/a/1-(2).png", ".png"),
        ("https://x.comic-site.test/a/ch-259-img-1.JPG", ".jpg"),
        ("https://x.comic-site.test/a/ch_1_1.jpeg", ".jpg"),
    ],
)
def test_extensions_are_normalised(url, ext):
    assert cs.image_ext(url) == ext


@pytest.mark.parametrize(
    "url",
    [
        "https://x.comic-site.test/a/c_1_1.gif",
        "https://x.comic-site.test/a/c_1_1",
    ],
)
def test_an_unknown_or_missing_extension_is_refused(url):
    with pytest.raises(cs.ScrapeError):
        cs.image_ext(url)


# -- downloading one chapter -----------------------------------------------

JPEG = b"\xff\xd8\xff\xe0" + b"0" * 400


def _page_html(*urls):
    return "".join(
        f'<img class="wp-manga-chapter-img" data-src="{u}">' for u in urls
    )


def test_pages_are_saved_under_the_ordinal(tmp_path, monkeypatch):
    page = "https://c4.comic-site.test/x/chapter_9/ch_9_1.jpg"
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, _page_html(page)))
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (200, JPEG))
    record = ChapterRecord(ordinal=5, slug="chapter-5", label="Chapter 5", number=5.0)

    result = cs.download_chapter(SERIES, record, tmp_path, delay=0)

    assert [p.name for p in tmp_path.glob("*.jpg")] == ["c_000005_001.jpg"]
    assert (result.saved, result.failed, result.on_disk) == (1, 0, 1)


def test_a_page_that_saves_nothing_is_reported_as_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cs, "default_transport", lambda u, h, c: (200, _page_html("https://c4.comic-site.test/x/a.jpg"))
    )
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (404, b""))
    record = ChapterRecord(ordinal=5, slug="chapter-5", label="Chapter 5", number=5.0)

    result = cs.download_chapter(SERIES, record, tmp_path, delay=0)

    assert (result.saved, result.failed, result.on_disk) == (0, 1, 0)
    assert list(tmp_path.glob("*.jpg")) == []


def test_an_html_error_page_is_never_saved_as_an_image(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cs, "default_transport", lambda u, h, c: (200, _page_html("https://c4.comic-site.test/x/a.jpg"))
    )
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (200, b"<html>error</html>"))
    record = ChapterRecord(ordinal=5, slug="chapter-5", label="Chapter 5", number=5.0)

    result = cs.download_chapter(SERIES, record, tmp_path, delay=0)

    assert result.saved == 0 and result.failed == 1
    assert list(tmp_path.glob("*.jpg")) == []


def test_existing_pages_are_not_downloaded_again(tmp_path, monkeypatch):
    page = "https://c4.comic-site.test/x/chapter_9/ch_9_1.jpg"
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, _page_html(page)))
    seen = []
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (seen.append(u), (200, JPEG))[1])
    (tmp_path / "c_000005_001.jpg").write_bytes(JPEG)
    record = ChapterRecord(ordinal=5, slug="chapter-5", label="Chapter 5", number=5.0)

    result = cs.download_chapter(SERIES, record, tmp_path, delay=0)

    assert seen == []
    assert (result.saved, result.failed, result.on_disk) == (0, 0, 1)


def test_overwrite_refetches_what_is_already_there(tmp_path, monkeypatch):
    page = "https://c4.comic-site.test/x/chapter_9/ch_9_1.jpg"
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, _page_html(page)))
    seen = []
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (seen.append(u), (200, JPEG))[1])
    (tmp_path / "c_000005_001.jpg").write_bytes(b"old")
    record = ChapterRecord(ordinal=5, slug="chapter-5", label="Chapter 5", number=5.0)

    result = cs.download_chapter(SERIES, record, tmp_path, overwrite=True, delay=0)

    assert seen == [page]
    assert result.saved == 1


def test_a_chapter_with_no_images_is_zero_pages_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, "<html></html>"))
    record = ChapterRecord(ordinal=5, slug="chapter-5", label="Chapter 5", number=5.0)

    result = cs.download_chapter(SERIES, record, tmp_path, delay=0)

    assert (result.saved, result.failed, result.on_disk) == (0, 0, 0)


def test_a_challenge_page_raises_rather_than_reporting_zero_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, CHALLENGE_HTML))
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)

    with pytest.raises(cs.ScrapeError, match="challenge"):
        cs.download_chapter(SERIES, record, tmp_path, delay=0)


def test_a_non_200_chapter_page_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (503, ""))
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)

    with pytest.raises(cs.ScrapeError, match="503"):
        cs.download_chapter(SERIES, record, tmp_path, delay=0)


# -- duplicates are settled before any bytes are written --------------------
import base64  # noqa: E402
import json  # noqa: E402

from comic_images import average_hash  # noqa: E402
from comic_manifest import Manifest  # noqa: E402

#: A real, decodable 16x16 checkerboard PNG.
#:
#: The ``JPEG`` above is only magic bytes -- enough for ``looks_like_image``,
#: not enough for ``average_hash``, which decodes the picture. A hash test needs
#: an image that actually decodes, so this fixture exists rather than
#: redefining ``JPEG`` (which would silently change what the download tests
#: above feed through).
PAGE_IMAGE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAAAAAA6mKC9AAAAEElEQVR4nGNg+E8AMowo"
    "FQDA/X+BQnmBUwAAAABJRU5ErkJggg=="
)


def test_the_chapter_with_more_pages_wins_a_duplicate_pair():
    short = ChapterRecord(ordinal=1, slug="a", label="Chapter 144.0", number=144.0, pages=80)
    long = ChapterRecord(ordinal=2, slug="b", label="Chapter 144", number=144.0, pages=81)
    short.content_hash = long.content_hash = "f" * 64

    cs.mark_duplicates([short, long])

    assert short.duplicate_of == 2
    assert long.duplicate_of is None


def test_a_duplicate_pair_is_decided_regardless_of_reading_order():
    short = ChapterRecord(ordinal=1, slug="a", label="Chapter 144.0", number=144.0, pages=80)
    long = ChapterRecord(ordinal=2, slug="b", label="Chapter 144", number=144.0, pages=81)
    short.content_hash = long.content_hash = "f" * 64

    cs.mark_duplicates([long, short])

    assert short.duplicate_of == 2


def test_distinct_chapters_are_not_marked_duplicates():
    one = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0, pages=50)
    two = ChapterRecord(ordinal=2, slug="b", label="Chapter 2", number=2.0, pages=51)
    one.content_hash, two.content_hash = "0" * 64, "f" * 64
    cs.mark_duplicates([one, two])
    assert one.duplicate_of is None and two.duplicate_of is None


def test_chapters_that_could_not_be_hashed_are_never_duplicates():
    one = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0, pages=10)
    two = ChapterRecord(ordinal=2, slug="b", label="Chapter 2", number=2.0, pages=10)
    one.content_hash = two.content_hash = ""

    cs.mark_duplicates([one, two])

    assert one.duplicate_of is None and two.duplicate_of is None


def test_a_page_count_tie_is_won_by_the_lower_ordinal_whatever_the_list_order():
    first = ChapterRecord(ordinal=1, slug="a", label="Chapter 144", number=144.0, pages=80)
    second = ChapterRecord(ordinal=2, slug="b", label="Chapter 144.0", number=144.0, pages=80)
    first.content_hash = second.content_hash = "f" * 64

    cs.mark_duplicates([second, first])

    assert first.duplicate_of is None
    assert second.duplicate_of == 1


def test_rescanning_clears_a_stale_duplicate_flag():
    kept = ChapterRecord(ordinal=1, slug="a", label="Chapter 144", number=144.0, pages=80)
    stale = ChapterRecord(ordinal=2, slug="b", label="Chapter 144.0", number=144.0,
                          pages=80, duplicate_of=1)
    kept.content_hash = "f" * 64
    # This time the dupe's page 1 could not be hashed, so it is no longer known
    # to be the same chapter and must not stay marked as one.
    stale.content_hash = ""

    cs.mark_duplicates([kept, stale])

    assert stale.duplicate_of is None


def test_a_scan_fills_the_page_count_and_hash(monkeypatch):
    page = "https://c4.comic-site.test/x/chapter_9/ch_9_1.jpg"
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, _page_html(page, page + "?2")))
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (200, PAGE_IMAGE))
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)

    scanned = cs.scan_chapter(SERIES, record)

    assert scanned.pages == 2
    assert scanned.content_hash == average_hash(PAGE_IMAGE)


def test_a_scan_writes_no_pages(tmp_path, monkeypatch):
    page = "https://c4.comic-site.test/x/chapter_9/ch_9_1.jpg"
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, _page_html(page)))
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (200, PAGE_IMAGE))
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)

    cs.scan_chapter(SERIES, record)

    assert list(tmp_path.iterdir()) == []


def test_a_scan_that_cannot_hash_page_one_leaves_the_hash_empty(monkeypatch):
    page = "https://c4.comic-site.test/x/chapter_9/ch_9_1.jpg"
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, _page_html(page)))
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (404, b""))
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)

    scanned = cs.scan_chapter(SERIES, record)

    assert scanned.pages == 1
    assert scanned.content_hash == ""


def test_a_truncated_page_one_is_unhashable_not_a_crash(monkeypatch):
    # JPEG carries the magic bytes but is not a decodable picture, which is what
    # a response cut off mid-transfer looks like. It must read as "could not
    # hash", not abort the scan of the whole series.
    page = "https://c4.comic-site.test/x/chapter_9/ch_9_1.jpg"
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, _page_html(page)))
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (200, JPEG))
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)

    scanned = cs.scan_chapter(SERIES, record)

    assert scanned.pages == 1
    assert scanned.content_hash == ""


def test_a_legacy_page_file_blocks_the_run(tmp_path, capsys):
    (tmp_path / "c_1_1.jpg").write_bytes(JPEG)
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)
    assert cs.has_legacy_files(tmp_path)
    assert cs.run(SERIES, tmp_path, Manifest("T", SERIES, [record]), [record]) == 1
    assert "unused/2026-11-01-pre-ordinal" in capsys.readouterr().out


def test_ordinal_named_files_are_not_legacy(tmp_path):
    (tmp_path / "c_000001_001.jpg").write_bytes(JPEG)
    assert not cs.has_legacy_files(tmp_path)


def test_a_missing_directory_has_no_legacy_files(tmp_path):
    assert not cs.has_legacy_files(tmp_path / "nope")


def test_a_duplicate_is_never_downloaded(tmp_path, monkeypatch):
    page = "https://c4.comic-site.test/x/chapter_9/ch_9_1.jpg"
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, _page_html(page)))
    seen = []
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (seen.append(u), (200, JPEG))[1])
    kept = ChapterRecord(ordinal=1, slug="a", label="Chapter 144", number=144.0, pages=81)
    dupe = ChapterRecord(ordinal=2, slug="b", label="Chapter 144.0", number=144.0,
                         pages=80, duplicate_of=1)
    manifest = Manifest("T", SERIES, [kept, dupe])

    code = cs.run(SERIES, tmp_path, manifest, [kept, dupe], delay=0)

    assert code == 0
    assert list(tmp_path.glob("c_000002_*")) == []
    assert seen.count(page) == 1


def test_a_run_that_saves_nothing_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, "<html></html>"))
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)
    assert cs.run(SERIES, tmp_path, Manifest("T", SERIES, [record]), [record], delay=0) == 1


def test_a_fully_resumed_run_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, "<html></html>"))
    (tmp_path / "c_000001_001.jpg").write_bytes(JPEG)
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0, pages=1)
    assert cs.run(SERIES, tmp_path, Manifest("T", SERIES, [record]), [record], delay=0) == 0


def test_the_manifest_is_written_after_each_chapter(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, "<html></html>"))
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)
    cs.run(SERIES, tmp_path, Manifest("T", SERIES, [record]), [record], delay=0)
    assert (tmp_path / "chapters.json").is_file()


def test_the_manifest_is_written_even_when_the_chapter_has_no_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, "<html></html>"))
    record = ChapterRecord(ordinal=7, slug="a", label="Chapter 7", number=7.0)
    cs.run(SERIES, tmp_path, Manifest("T", SERIES, [record]), [record], delay=0)

    saved = Manifest.from_dict(json.loads((tmp_path / "chapters.json").read_text("utf-8")))

    assert [c.ordinal for c in saved.chapters] == [7]


def test_a_refused_chapter_page_is_recorded_and_the_run_continues(
    tmp_path, monkeypatch, capsys
):
    def page(url, headers, cookies):
        if "chapter-1" in url:
            return 200, CHALLENGE_HTML
        return 200, "<html></html>"

    monkeypatch.setattr(cs, "default_transport", page)
    blocked = ChapterRecord(ordinal=1, slug="chapter-1", label="Chapter 1", number=1.0)
    other = ChapterRecord(ordinal=2, slug="chapter-2", label="Chapter 2", number=2.0)
    manifest = Manifest("T", SERIES, [blocked, other])

    code = cs.run(SERIES, tmp_path, manifest, [blocked, other], delay=0)

    # The refusal fails the run, but chapter 2 was still attempted and both
    # ordinals are in the manifest, so a retry resumes without renumbering.
    assert code == 1
    saved = Manifest.from_dict(json.loads((tmp_path / "chapters.json").read_text("utf-8")))
    assert [c.ordinal for c in saved.chapters] == [1, 2]
    out = capsys.readouterr().out
    assert "FAILED" in out and "Chapter 1" in out


def test_a_dry_run_writes_no_pages_but_still_saves_the_manifest(
    tmp_path, monkeypatch, capsys
):
    page = "https://c4.comic-site.test/x/chapter_9/ch_9_1.jpg"
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, _page_html(page)))
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (200, JPEG))
    record = ChapterRecord(ordinal=3, slug="a", label="Chapter 3", number=3.0)

    code = cs.run(
        SERIES, tmp_path, Manifest("T", SERIES, [record]), [record],
        delay=0, dry_run=True,
    )

    assert code == 0
    assert list(tmp_path.glob("c_*.jpg")) == []
    assert (tmp_path / "chapters.json").is_file()
    out = capsys.readouterr().out
    assert "3" in out and "Chapter 3" in out
