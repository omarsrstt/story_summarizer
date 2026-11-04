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


def test_a_broken_config_does_not_make_a_url_check_raise(monkeypatch):
    # A config typo must not surface as a JSONDecodeError from inside
    # page_urls/scan_chapter/download_chapter, none of which expect it.
    import json

    monkeypatch.setattr(cs, "_SITE_HOST", None)
    monkeypatch.setattr(cs, "load_config", lambda path=None: json.loads("{not json"))

    assert cs.safe_image_url("https://c1.comic-site.test/a/c_1_1.jpg")
    assert not cs.safe_image_url("https://evil.example/a.jpg")


def test_a_missing_config_falls_back_to_the_default_host(monkeypatch):
    monkeypatch.setattr(cs, "_SITE_HOST", None)
    monkeypatch.setattr(cs, "load_config", lambda path=None: {})

    assert cs.safe_image_url("https://c1.comic-site.test/a/c_1_1.jpg")


def test_the_config_is_read_once_per_call_not_once_per_image(monkeypatch):
    # 40 images on a page used to mean 40 reads and parses of the same file.
    reads = []
    monkeypatch.setattr(cs, "_SITE_HOST", None)
    monkeypatch.setattr(
        cs, "load_config", lambda path=None: (reads.append(1), {"comic_site_host": "comic-site.test"})[1]
    )
    html = _page_html(
        *[f"https://c4.comic-site.test/x/chapter_9/ch_9_{i}.jpg" for i in range(1, 41)]
    )

    assert len(cs.page_urls(html)) == 40
    assert len(reads) == 1


def test_the_host_cache_is_reset_by_the_module_global(monkeypatch):
    # The reset mechanism the tests rely on, asserted directly: a cached host
    # left over from an earlier test would silently decide later answers.
    monkeypatch.setattr(cs, "_SITE_HOST", "other.example")
    assert cs.safe_image_url("https://c1.other.example/a/c_1_1.jpg")
    monkeypatch.setattr(cs, "_SITE_HOST", None)
    monkeypatch.setattr(cs, "load_config", lambda path=None: {"comic_site_host": "comic-site.test"})
    assert cs.safe_image_url("https://c1.comic-site.test/a/c_1_1.jpg")


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


def test_an_interrupted_write_leaves_no_file_that_counts_as_a_page(tmp_path, monkeypatch):
    # A write interrupted mid-call (Ctrl+C, power loss, a full disk) must not
    # leave a truncated page that every later run treats as finished: the
    # existence check would skip it and ``pages_on_disk`` would count it, so the
    # chapter could exit 0 holding a page nothing can read. The bytes go to a
    # temporary name first, so a kill between the write and the rename leaves a
    # ``.part`` file -- not a page name, and not counted.
    page = "https://c4.comic-site.test/x/chapter_9/ch_9_1.jpg"
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, _page_html(page)))
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (200, PAGE_IMAGE))
    record = ChapterRecord(ordinal=9, slug="chapter-9", label="Chapter 9", number=9.0)

    def half_write(self, payload):
        # A real interrupted write: half the bytes reach the file, then the
        # machine goes away. This is what distinguishes the two implementations
        # -- writing straight to the page name leaves a readable-looking stub.
        with open(self, "wb") as handle:
            handle.write(payload[: len(payload) // 2])
        raise KeyboardInterrupt("power loss")

    monkeypatch.setattr(type(tmp_path / "x"), "write_bytes", half_write)

    try:
        cs.download_chapter(SERIES, record, tmp_path, delay=0)
    except KeyboardInterrupt:
        pass

    assert cs.pages_on_disk(tmp_path, 9) == 0
    # The partial bytes may survive under the temporary name. What must not
    # survive is a file wearing a page's name -- that is the one the existence
    # check and the reader would both accept.
    assert [p.name for p in tmp_path.iterdir() if p.suffix in {".jpg", ".jpeg", ".png", ".webp"}] == []
    assert list(tmp_path.glob("*.part"))


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
import io  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import comic_images  # noqa: E402
from comic_images import average_hash, chapter_fingerprint  # noqa: E402
from comic_manifest import Manifest  # noqa: E402


def _png(pixels) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.array(pixels, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _quad(top_left, top_right, bottom_left, bottom_right) -> bytes:
    """A 32x32 page whose four quadrants are separately black or white.

    Flat fills all hash to the same value (a mean with nothing above it), so the
    pages a fingerprint test needs must have structure. This is the smallest
    picture that reliably produces distinct hashes.
    """
    top = np.hstack([np.full((16, 16), top_left), np.full((16, 16), top_right)])
    bottom = np.hstack(
        [np.full((16, 16), bottom_left), np.full((16, 16), bottom_right)]
    )
    return _png(np.vstack([top, bottom]))


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

#: Three distinct decodable pages, so a fingerprint of pages 2-4 is 3 hashes.
PAGE_B = _quad(255, 0, 0, 255)
PAGE_C = _quad(0, 255, 255, 0)
PAGE_D = _quad(255, 255, 0, 0)


def _flipped(fingerprint: str, count: int, *, start: int = 0) -> str:
    """``fingerprint`` with exactly ``count`` of its bits inverted from ``start``.

    The measured numbers this file encodes are in bits -- 0 for a true duplicate,
    337/768 for two chapters sharing an opener -- so the fixtures are built to a
    bit distance rather than by hand-picking images and hoping. ``start`` lets two
    fixtures differ from the original by a similar amount while differing from
    each other by a lot, which is what keeps them out of one another's chain.
    """
    bits = bin(int(fingerprint, 16))[2:].zfill(len(fingerprint) * 4)
    inverted = "".join(
        ("0" if bit == "1" else "1") if start <= index < start + count else bit
        for index, bit in enumerate(bits)
    )
    return f"{int(inverted, 2):0{len(fingerprint)}x}"


#: What ``scan_chapter`` really stores: three page hashes, 768 bits.
FINGERPRINT = chapter_fingerprint([PAGE_IMAGE, PAGE_B, PAGE_C, PAGE_D])


def test_the_fixture_fingerprint_is_three_page_hashes():
    assert len(FINGERPRINT) == 192
    assert FINGERPRINT == average_hash(PAGE_B) + average_hash(PAGE_C) + average_hash(PAGE_D)


def test_the_chapter_with_more_pages_wins_a_duplicate_pair():
    short = ChapterRecord(ordinal=1, slug="a", label="Chapter 144.0", number=144.0, pages=80)
    long = ChapterRecord(ordinal=2, slug="b", label="Chapter 144", number=144.0, pages=81)
    short.content_hash = long.content_hash = FINGERPRINT

    cs.mark_duplicates([short, long])

    assert short.duplicate_of == 2
    assert long.duplicate_of is None


def test_a_true_duplicate_pair_is_marked():
    # Measured on the live site: 144/144.0 and 150/150.0 are 0 bits apart out of
    # 768 once pages 2-4 are compared.
    kept = ChapterRecord(ordinal=1, slug="a", label="Chapter 144", number=144.0, pages=81)
    dupe = ChapterRecord(ordinal=2, slug="b", label="Chapter 144.0", number=144.0, pages=81)
    kept.content_hash = dupe.content_hash = FINGERPRINT

    cs.mark_duplicates([kept, dupe])

    assert dupe.duplicate_of == 1
    assert kept.duplicate_of is None


def test_a_true_duplicate_with_a_re_encoded_page_is_still_marked():
    # Slack for a re-encode is what an average hash exists to provide. Equality
    # on the string was too brittle for a 768-bit fingerprint.
    kept = ChapterRecord(ordinal=1, slug="a", label="Chapter 150", number=150.0, pages=81)
    dupe = ChapterRecord(ordinal=2, slug="b", label="Chapter 150.0", number=150.0, pages=81)
    kept.content_hash = FINGERPRINT
    dupe.content_hash = _flipped(FINGERPRINT, 20)

    cs.mark_duplicates([kept, dupe])

    assert dupe.duplicate_of == 1


def test_two_chapters_sharing_an_opener_are_not_duplicates():
    # THE REGRESSION TEST for the live bug. 175.1 and 175.2 share a byte-identical
    # page 1 (0 bits apart there) and are 337 bits apart out of 768 on pages 2-4.
    # The old page-1 fingerprint called them equal and dropped a real chapter.
    first = ChapterRecord(ordinal=1, slug="a", label="Chapter 175.1", number=175.1, pages=4)
    second = ChapterRecord(ordinal=2, slug="b", label="Chapter 175.2", number=175.2, pages=4)
    first.content_hash = FINGERPRINT
    second.content_hash = _flipped(FINGERPRINT, 337)

    cs.mark_duplicates([first, second])

    assert first.duplicate_of is None
    assert second.duplicate_of is None


def test_the_threshold_scales_with_the_number_of_pages_hashed():
    # 10 bits of slack per page hash: 30 out of 768 for a three-page fingerprint.
    # The two neighbours flip disjoint bit ranges, otherwise they would also be
    # within the threshold of each other and group together as one chapter.
    kept = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0, pages=10)
    within = ChapterRecord(ordinal=2, slug="b", label="Chapter 2", number=2.0, pages=10)
    beyond = ChapterRecord(ordinal=3, slug="c", label="Chapter 3", number=3.0, pages=10)
    kept.content_hash = FINGERPRINT
    within.content_hash = _flipped(FINGERPRINT, 30)
    beyond.content_hash = _flipped(FINGERPRINT, 31, start=400)

    cs.mark_duplicates([kept, within, beyond])

    assert within.duplicate_of == 1
    assert beyond.duplicate_of is None


def test_a_duplicate_pair_is_decided_regardless_of_reading_order():
    short = ChapterRecord(ordinal=1, slug="a", label="Chapter 144.0", number=144.0, pages=80)
    long = ChapterRecord(ordinal=2, slug="b", label="Chapter 144", number=144.0, pages=81)
    short.content_hash = long.content_hash = FINGERPRINT

    cs.mark_duplicates([long, short])

    assert short.duplicate_of == 2


def test_distinct_chapters_are_not_marked_duplicates():
    one = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0, pages=50)
    two = ChapterRecord(ordinal=2, slug="b", label="Chapter 2", number=2.0, pages=51)
    one.content_hash, two.content_hash = _flipped(FINGERPRINT, 400), FINGERPRINT
    cs.mark_duplicates([one, two])
    assert one.duplicate_of is None and two.duplicate_of is None


def test_chapters_that_could_not_be_hashed_are_never_duplicates():
    one = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0, pages=10)
    two = ChapterRecord(ordinal=2, slug="b", label="Chapter 2", number=2.0, pages=10)
    one.content_hash = two.content_hash = ""

    cs.mark_duplicates([one, two])

    assert one.duplicate_of is None and two.duplicate_of is None


def test_fingerprints_of_different_lengths_are_never_duplicates():
    # A one-page fingerprint and a three-page one have no meaningful distance.
    # Erring towards "not a duplicate" is deliberate: a false duplicate silently
    # drops a real chapter, a missed duplicate only wastes bandwidth.
    three_pages = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0, pages=4)
    one_page = ChapterRecord(ordinal=2, slug="b", label="Chapter 2", number=2.0, pages=2)
    three_pages.content_hash = FINGERPRINT
    one_page.content_hash = FINGERPRINT[:64]

    cs.mark_duplicates([three_pages, one_page])

    assert three_pages.duplicate_of is None
    assert one_page.duplicate_of is None


def test_a_legacy_sixty_four_char_hash_only_compares_at_sixty_four_chars():
    # Records the limitation rather than pretending it does not exist: a hash
    # written by the old page-1 code is 64 characters, and the length guard in
    # ``_is_duplicate`` is a length test, not a format test. So a legacy value
    # is never compared against a current 192-character fingerprint -- but it IS
    # compared against any other 64-character value, whether that is another
    # legacy hash or a current one-page fingerprint from a 2-page chapter
    # (``len // 64`` = 1). Two distinct chapters that share a cover could
    # therefore still group if both carried a legacy hash.
    #
    # This is pinned deliberately, and the route is closed at the caller instead
    # of here: ``execute`` rescans every chapter in the plan and clears the hash
    # of any record whose scan failed, so no stale value reaches
    # ``mark_duplicates``. Removing that rescan-and-clear would re-open it, and
    # this test says so out loud rather than letting the comment claim a
    # protection the code does not provide.
    legacy = FINGERPRINT[:64]
    other_legacy = _flipped(legacy, 4)
    current = FINGERPRINT

    # Different lengths never compare, legacy or not.
    assert not cs._is_duplicate(legacy, current)

    # Two 64-character values within the one-page threshold of 10 bits DO
    # compare, which is exactly the latent risk described above.
    assert cs._is_duplicate(legacy, other_legacy)


def test_a_page_count_tie_is_won_by_the_lower_ordinal_whatever_the_list_order():
    first = ChapterRecord(ordinal=1, slug="a", label="Chapter 144", number=144.0, pages=80)
    second = ChapterRecord(ordinal=2, slug="b", label="Chapter 144.0", number=144.0, pages=80)
    first.content_hash = second.content_hash = FINGERPRINT

    cs.mark_duplicates([second, first])

    assert first.duplicate_of is None
    assert second.duplicate_of == 1


def test_a_chain_of_near_matches_keeps_one_survivor():
    # A matches B and B matches C, so all three are the same chapter even though
    # the ends are further apart than the threshold.
    first = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0, pages=10)
    second = ChapterRecord(ordinal=2, slug="b", label="Chapter 2", number=2.0, pages=10)
    third = ChapterRecord(ordinal=3, slug="c", label="Chapter 3", number=3.0, pages=10)
    first.content_hash = FINGERPRINT
    second.content_hash = _flipped(FINGERPRINT, 20)
    third.content_hash = _flipped(FINGERPRINT, 40)

    cs.mark_duplicates([first, second, third])

    assert second.duplicate_of == 1
    assert third.duplicate_of == 1


def test_rescanning_clears_a_stale_duplicate_flag():
    kept = ChapterRecord(ordinal=1, slug="a", label="Chapter 144", number=144.0, pages=80)
    stale = ChapterRecord(ordinal=2, slug="b", label="Chapter 144.0", number=144.0,
                          pages=80, duplicate_of=1)
    kept.content_hash = FINGERPRINT
    # This time the dupe's pages could not be hashed, so it is no longer known
    # to be the same chapter and must not stay marked as one.
    stale.content_hash = ""

    cs.mark_duplicates([kept, stale])

    assert stale.duplicate_of is None


def test_a_scan_fingerprints_pages_two_to_four(monkeypatch):
    # Four pages listed, so the fingerprint covers pages 2, 3 and 4. Each is
    # served a different picture so the result can be checked position by
    # position -- and so that a fingerprint that accidentally included page 1
    # would be visibly wrong rather than coincidentally right.
    pages = {
        "ch_9_1.jpg": PAGE_IMAGE,
        "ch_9_2.jpg": PAGE_B,
        "ch_9_3.jpg": PAGE_C,
        "ch_9_4.jpg": PAGE_D,
    }
    html = _page_html(*(f"https://c4.comic-site.test/x/chapter_9/{name}" for name in pages))
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, html))
    monkeypatch.setattr(
        cs, "bytes_transport", lambda u, h, c: (200, pages[u.rsplit("/", 1)[-1]])
    )
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)

    scanned = cs.scan_chapter(SERIES, record)

    assert scanned.pages == 4
    assert scanned.content_hash == FINGERPRINT
    # Page 1 is not in it at all.
    assert average_hash(PAGE_IMAGE) not in scanned.content_hash


def test_a_scan_fetches_only_the_fingerprint_pages(monkeypatch):
    # Five pages listed, and only pages 2-4 are fetched: three images, because
    # page 1 is not evidence of anything and is never requested. That is one
    # request per chapter fewer than fetching it just to skip it -- ~300 saved on
    # a full series, against a site that bot-checks.
    names = [f"ch_9_{index}.jpg" for index in range(1, 6)]
    html = _page_html(*(f"https://c4.comic-site.test/x/chapter_9/{name}" for name in names))
    fetched = []
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, html))
    monkeypatch.setattr(
        cs, "bytes_transport", lambda u, h, c: (fetched.append(u.rsplit("/", 1)[-1]), (200, PAGE_B))[1]
    )
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)

    cs.scan_chapter(SERIES, record)

    assert fetched == names[1:4]
    assert names[0] not in fetched
    assert len(record.content_hash) == 192


def test_a_scan_writes_no_pages(tmp_path, monkeypatch):
    page = "https://c4.comic-site.test/x/chapter_9/ch_9_1.jpg"
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, _page_html(page)))
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (200, PAGE_IMAGE))
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)

    cs.scan_chapter(SERIES, record)

    assert list(tmp_path.iterdir()) == []


def test_a_scan_that_cannot_fetch_any_fingerprint_page_leaves_it_empty(monkeypatch):
    # Two pages, so page 2 is inside the window and is the only page that could
    # contribute a hash. It 404s, so there is no evidence at all.
    html = _page_html(
        "https://c4.comic-site.test/x/chapter_9/ch_9_1.jpg",
        "https://c4.comic-site.test/x/chapter_9/ch_9_2.jpg",
    )
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, html))
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (404, b""))
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)

    scanned = cs.scan_chapter(SERIES, record)

    assert scanned.pages == 2
    assert scanned.content_hash == ""


def test_a_truncated_page_is_unhashable_not_a_crash(monkeypatch):
    # JPEG carries the magic bytes but is not a decodable picture, which is what
    # a response cut off mid-transfer looks like. It must read as "could not
    # hash", not abort the scan of the whole series. Two pages, so the truncated
    # one is the fingerprint page -- page 1 is not fetched at all.
    html = _page_html(
        "https://c4.comic-site.test/x/chapter_9/ch_9_1.jpg",
        "https://c4.comic-site.test/x/chapter_9/ch_9_2.jpg",
    )
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, html))
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (200, JPEG))
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)

    scanned = cs.scan_chapter(SERIES, record)

    assert scanned.pages == 2
    assert scanned.content_hash == ""


def test_a_failed_fetch_costs_its_slot_and_never_refills_the_window(monkeypatch):
    # Page 2 -- the first page of the fingerprint window -- fails to fetch. The
    # window must stay pages 2-4: no later page may be pulled in to stand in for
    # the one that failed, because the scan would then describe pages 3-5 and
    # never match a clean scan of the same artwork. The assertion on ``fetched``
    # is the one with teeth here -- a drop-without-refill implementation happens
    # to produce the same string, because an undecodable payload is skipped
    # anyway (see ``comic_images.chapter_fingerprint``); what must not happen is
    # the window sliding forward.
    names = [f"ch_9_{index}.jpg" for index in range(1, 6)]
    html = _page_html(*(f"https://c4.comic-site.test/x/chapter_9/{name}" for name in names))
    served = {names[2]: PAGE_C, names[3]: PAGE_D, names[4]: PAGE_B}
    fetched = []

    def image(url, headers, cookies):
        name = url.rsplit("/", 1)[-1]
        fetched.append(name)
        if name == names[1]:
            raise RuntimeError("connection reset")
        return 200, served[name]

    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, html))
    monkeypatch.setattr(cs, "bytes_transport", image)
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)

    scanned = cs.scan_chapter(SERIES, record)

    # Exactly the window: page 1 was never requested, and page 5 was not pulled
    # in to replace the page that failed.
    assert fetched == names[1:4]
    assert scanned.content_hash == average_hash(PAGE_C) + average_hash(PAGE_D)


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


def test_a_failed_scan_cannot_lend_its_stale_hash_to_a_live_chapter(
    tmp_path, monkeypatch
):
    # The manifest restores last run's fingerprint onto each record. If a scan
    # fails and the old value survives, a *different* chapter that hashes to that
    # value this run is marked as its duplicate -- one real chapter skipped on the
    # strength of a scan that could not run.
    stale = ChapterRecord(
        ordinal=1, slug="chapter-1", label="Chapter 1", number=1.0,
        pages=9, content_hash="f" * 192,
    )
    live = ChapterRecord(ordinal=2, slug="chapter-2", label="Chapter 2", number=2.0)
    manifest = Manifest("T", SERIES, [stale, live])
    cs.save_manifest(tmp_path / "chapters.json", manifest)

    def page(url, headers, cookies):
        if "chapter-1" in url:
            return 500, ""
        # Four pages, so the scan produces a full three-hash fingerprint.
        return 200, _page_html(
            *(f"https://c4.comic-site.test/x/chapter_2/ch_2_{i}.jpg" for i in range(1, 5))
        )

    monkeypatch.setattr(cs, "default_transport", page)
    # Chapter 2's fingerprint comes out exactly the value chapter 1 carries from
    # last run. ``scan_chapter`` imports ``average_hash`` inside the module it
    # lives in, so patch it at the source.
    monkeypatch.setattr(comic_images, "average_hash", lambda payload: "f" * 64)
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (200, PAGE_IMAGE))

    setup = cs.Setup(
        SERIES, tmp_path, tmp_path / "chapters.json",
        cs.load_manifest(tmp_path / "chapters.json"),
        [stale, live],
    )
    args = cs.parse_args(["-n", "T", "-c", "1-2"])

    cs.execute(setup, args)

    saved = Manifest.from_dict(
        json.loads((tmp_path / "chapters.json").read_text("utf-8"))
    )
    assert saved.chapters[0].content_hash == ""
    assert saved.chapters[0].pages == 0
    assert saved.chapters[1].duplicate_of is None
    assert len(saved.chapters[1].content_hash) == 192


# -- the plan is refused before it is fetched --------------------------------


def test_a_name_that_escapes_the_comics_directory_is_refused():
    # -n becomes a path component directly, so ../../escape would write outside
    # comics/. comic_reader.comic_slug already refuses this exact shape.
    for name in ("../escape", "a/b", "a\\b", "..", "."):
        with pytest.raises(SystemExit):
            cs.parse_args(["-n", name])


def test_a_name_that_cannot_be_a_folder_is_a_clean_message(capsys):
    with pytest.raises(SystemExit):
        cs.parse_args(["-n", "../escape"])
    err = capsys.readouterr().err
    assert "-n:" in err and "escape" in err
    assert "Traceback" not in err


def test_an_empty_manifest_cannot_be_bounded():
    # --from/--to default an omitted bound to the first/last chapter, which an
    # empty manifest does not have; indexing it raised IndexError.
    empty = Manifest("T", SERIES, [])
    with pytest.raises(cs.ScrapeError, match="empty"):
        cs._bounds_selection(empty, None, None)
    with pytest.raises(cs.ScrapeError, match="empty"):
        cs._bounds_selection(empty, "1", "2")


def test_a_half_typed_label_is_refused_before_the_chapter_list_is_fetched(monkeypatch):
    # parse_args must not let -c chapter through as "all" and then spend the
    # chapter-list request to find out.
    def boom(*args, **kwargs):
        raise AssertionError("the chapter list must not be fetched")

    monkeypatch.setattr(cs, "post_transport", boom)
    with pytest.raises(SystemExit):
        cs.main(["-n", "T", "-c", "chapter"])
