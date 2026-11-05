# Comic Download Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken `src/comic_scraper.py` with a download stage that finds every chapter the site has, saves its pages under reading-order ordinal filenames, and can never report success while having saved nothing.

**Architecture:** One AJAX request yields the full chapter list (newest-first). A manifest (`chapters.json`) pins each chapter's reading-order ordinal to its site slug so ordinals never shift. A scan pass fetches each chapter's page list and the hash of its first page, which settles duplicates *before* any bytes are written. Pages are then written as `c_<ordinal:06d>_<page:03d>.<ext>`, a shape `comic_reader.group_chapters` already parses, so the reader and summariser need no changes.

**Tech Stack:** Python 3.11, `curl_cffi` (TLS impersonation), BeautifulSoup, PIL + numpy (average hashing), pytest.

**Spec:** `docs/superpowers/specs/2026-11-01-comic-download-design.md`

## Global Constraints

- Target a series' pages **only** on `comic-site.test` and its subdomains, over HTTP(S) only. An off-site image URL is refused, never followed.
- No JavaScript is ever executed and no browser is opened by the download stage.
- `comic_reader.py` and `summarize.py` are **not modified** by this plan.
- Ordinals are 1-based, assigned from the **reversed** AJAX list, and never reassigned once written.
- Image filenames match `comic_reader.CHAPTER_PATTERNS[0]`: `c_<digits>_<digits>.<jpg|jpeg|png|webp>`.
- The download stage requires `uv sync --extra comic --extra scraper` and must say so on a missing import.
- The run must never overwrite or delete the owner's existing data; legacy trees are archived to `unused/2026-11-01-pre-ordinal/` by hand (Task 8).
- Duplicate resolution is "the chapter with more pages wins", decided in the scan pass before any page file is written.
- All commands run through `uv run`.

## Review Focus

- **A chapter page that returns HTTP 200 but no images** (what the site does above chapter 280, and what a Cloudflare interstitial does). Expectation: an explicit selection reports that chapter as having no pages and exits non-zero; `--all` does not treat one empty chapter as the end of the series when the list said otherwise.
- **A chapter whose image list is empty after filtering** (every URL off-site). Expectation: reported as a failure, non-zero exit, never a clean "0 to download".
- **A run interrupted halfway.** Expectation: the manifest and every written page survive; the next run resumes without renumbering.
- **A fully resumed run** (everything already on disk). Expectation: exit 0, nothing re-downloaded, no "nothing was saved" failure.
- **A fractional or duplicated label** (`175.1`, `144.0`). Expectation: `-c 175.1` selects it, and `144.0` is recognized as a re-upload of `144` rather than a second chapter of the story.

---

### Task 1: One page-naming rule, owned by the reader

**Files:**
- Modify: `src/comic_reader.py:34-40` (below the `CHAPTER_PATTERNS` block)
- Test: `tests/test_comic_naming.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `comic_reader.CHAPTER_PATTERNS: tuple[re.Pattern, ...]` — unchanged, and now the single source of truth for page filenames.
  - `comic_reader.PAGE_EXTENSIONS: frozenset[str]` → `{".jpg", ".jpeg", ".png", ".webp"}`.
  - `comic_reader.page_filename(ordinal: int, page: int, ext: str) -> str` → `"c_000012_003.jpg"`; raises `ValueError` on an unsupported extension.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_comic_naming.py
import pytest

from comic_reader import page_filename


def test_an_ordinal_and_page_are_zero_padded():
    assert page_filename(12, 3, ".jpg") == "c_000012_003.jpg"


def test_lexical_order_equals_reading_order():
    names = [page_filename(n, 1, ".jpg") for n in (2, 10, 100)]
    assert sorted(names) == names


def test_a_webp_page_keeps_its_extension():
    assert page_filename(7, 2, ".webp") == "c_000007_002.webp"


def test_the_reader_accepts_the_name_the_writer_produces():
    from comic_reader import CHAPTER_PATTERNS

    # The pattern is greedy over \d+, so it captures "000175", not "175". What
    # matters is that the reader still recovers the ordinal it was written with.
    name = page_filename(175, 12, ".png")
    match = CHAPTER_PATTERNS[0].match(name)
    assert match is not None
    assert (int(match.group(1)), int(match.group(2))) == (175, 12)


def test_an_unknown_extension_is_refused():
    with pytest.raises(ValueError):
        page_filename(1, 1, ".gif")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_comic_naming.py -v`
Expected: FAIL with `ImportError: cannot import name 'page_filename'`

- [ ] **Step 3: Implement in `src/comic_reader.py`**

Leave `CHAPTER_PATTERNS` untouched (it already accepts any digit run, which is why ordinals need no reader change) and add directly below it:

```python
PAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def page_filename(ordinal: int, page: int, ext: str) -> str:
    """The file name for one page of chapter ``ordinal``.

    Zero-padding both numbers keeps lexical order equal to reading order, which
    is what lets ``group_chapters`` sort by name alone.
    """
```

Normalise `ext` to lowercase and to a leading dot; raise `ValueError` if the result is not in `PAGE_EXTENSIONS`; otherwise return `f"c_{ordinal:06d}_{page:03d}{normalised}"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_comic_naming.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/comic_reader.py tests/test_comic_naming.py
git commit -m "feat(comic): one shared page-naming rule for scraper and reader"
```

---

### Task 2: The chapter manifest

**Files:**
- Create: `src/comic_manifest.py`
- Test: `tests/test_comic_manifest.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass ChapterRecord(ordinal: int, slug: str, label: str, number: float, pages: int = 0, content_hash: str = "", duplicate_of: int | None = None)`
  - `@dataclass Manifest(series: str, site: str, chapters: list[ChapterRecord])`
  - `load_manifest(path: Path) -> Manifest` — a missing file yields an empty manifest.
  - `save_manifest(path: Path, manifest: Manifest) -> None` — atomic: write `<path>.tmp`, then `os.replace`.
  - `assign(manifest: Manifest, listing: list[tuple[str, str]]) -> Manifest` — `listing` is `(slug, label)` in the site's **raw newest-first** order; returns a manifest whose `chapters` are in ascending ordinal order, and prints one line naming any chapter it could not place in its list position.
  - `ordinal_of_label(manifest: Manifest, label: str) -> int | None`
  - `parse_label_number(label: str) -> float` — `"Chapter 175.10"` → `175.10`, `"Chapter 280 - 237"` → `280.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_comic_manifest.py
from pathlib import Path

from comic_manifest import (
    ChapterRecord,
    Manifest,
    assign,
    load_manifest,
    ordinal_of_label,
    parse_label_number,
    save_manifest,
)

# Raw AJAX order is newest-first, exactly as the site returns it.
NEWEST_FIRST = [
    ("chapter-177", "Chapter 177"),
    ("chapter-175-19", "Chapter 175.19"),
    ("chapter-175", "Chapter 175"),
    ("chapter-174", "Chapter 174"),
]


def _empty():
    return Manifest(series="TBATE", site="s", chapters=[])


def test_the_newest_first_list_is_reversed_into_reading_order():
    manifest = assign(_empty(), NEWEST_FIRST)
    assert {c.ordinal: c.label for c in manifest.chapters} == {
        1: "Chapter 174",
        2: "Chapter 175",
        3: "Chapter 175.19",
        4: "Chapter 177",
    }


def test_list_order_beats_numeric_order_for_inserted_chapters():
    # 175.19 was inserted after 175; sorting by number would put it before 175.
    manifest = assign(_empty(), NEWEST_FIRST)
    assert ordinal_of_label(manifest, "Chapter 175.19") > ordinal_of_label(
        manifest, "Chapter 175"
    )


def test_existing_ordinals_survive_a_retroactive_insertion():
    first = assign(_empty(), NEWEST_FIRST)
    second = assign(first, NEWEST_FIRST + [("chapter-173", "Chapter 173")])
    for label in ("Chapter 177", "Chapter 175.19", "Chapter 175", "Chapter 174"):
        assert ordinal_of_label(second, label) == ordinal_of_label(first, label)


def test_a_new_chapter_never_takes_a_used_ordinal():
    first = assign(_empty(), NEWEST_FIRST)
    second = assign(first, NEWEST_FIRST + [("chapter-174-5", "Chapter 174.5")])
    ordinals = [c.ordinal for c in second.chapters]
    assert len(ordinals) == len(set(ordinals))


def test_a_fractional_label_parses_exactly():
    assert parse_label_number("Chapter 175.10") == 175.10
    assert parse_label_number("Chapter 175.1") == 175.1
    assert parse_label_number("Chapter 280 - 237") == 280.0


def test_records_added_fields_are_preserved_across_a_reassign():
    manifest = assign(_empty(), NEWEST_FIRST)
    manifest.chapters[0].pages = 7
    again = assign(manifest, NEWEST_FIRST)
    assert again.chapters[0].pages == 7


def test_a_manifest_round_trips_through_disk(tmp_path: Path):
    path = tmp_path / "chapters.json"
    original = assign(_empty(), NEWEST_FIRST)
    original.chapters[0].duplicate_of = None
    original.chapters[0].content_hash = "abc"
    save_manifest(path, original)
    loaded = load_manifest(path)
    assert [
        (c.ordinal, c.slug, c.label, c.pages, c.content_hash, c.duplicate_of)
        for c in loaded.chapters
    ] == [
        (c.ordinal, c.slug, c.label, c.pages, c.content_hash, c.duplicate_of)
        for c in original.chapters
    ]


def test_a_missing_manifest_is_empty(tmp_path: Path):
    assert load_manifest(tmp_path / "nope.json").chapters == []


def test_an_empty_listing_keeps_what_is_already_known():
    first = assign(_empty(), NEWEST_FIRST)
    assert assign(first, []).chapters == first.chapters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_comic_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'comic_manifest'`

- [ ] **Step 3: Implement `src/comic_manifest.py`**

Model the atomic write on `scraper_store.py`. `ChapterRecord` converts to and from JSON with `dataclasses.asdict` / `ChapterRecord(**fields)`; `Manifest` serialises `{"series", "site", "chapters"}`.

`assign` is the only piece with a shape worth spelling out:

1. Group the existing records by `slug`.
2. Reverse `listing` so it is oldest-first (reading order).
3. Walk it. A known slug keeps its record unchanged. An unknown slug gets `max(ordinal of every chapter already in the manifest) + 1`.
4. Records for slugs no longer in the listing are kept unchanged, so a site that temporarily drops a chapter does not lose its ordinal.
5. Return `chapters` sorted by ordinal.

An unknown slug can only ever take a number strictly greater than every ordinal already assigned, so **no existing chapter's ordinal can ever move** — that is the design's binding constraint. The price is deliberate: a chapter the site inserts retroactively sits at the end of reading order rather than in its true position, because the integer slot between two assigned chapters does not exist and renumbering is forbidden. `assign` therefore prints one line for each such chapter (`note: <label> is listed before <label> but was appended as chapter <N>`) so the anomaly is visible rather than silent.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_comic_manifest.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/comic_manifest.py tests/test_comic_manifest.py
git commit -m "feat(comic): a manifest that pins ordinals to slugs"
```

---

### Task 3: One retry policy, and reading the chapter list in one request

**Files:**
- Modify: `src/scraper_fast.py:52-96` (extract the retry loop, add `post_transport`)
- Modify: `src/comic_scraper.py` (rewrite the parsing half; delete `next_chapter`, `Run`, `ChapterPage`, `ChapterPlan`, `plan_chapters`, `image_filename`, `scrape_chapters`)
- Test: `tests/test_scraper_fast.py` (append), `tests/test_comic_scraper.py` (replace the file: the walk/resume tests exercise deleted functions)

**Interfaces:**
- Consumes: `scraper_fast._MAX_ATTEMPTS`, `_BACKOFF_BASE`, `_RATE_LIMIT_MAX_ATTEMPTS`, `_RATE_LIMIT_BASE`; `scraper_parse.looks_like_challenge`; `comic_reader.load_config`.
- Produces:
  - `scraper_fast._with_retries(fetch: Callable[[], tuple[int, object]]) -> tuple[int, object]` — the existing policy, applied to any single-request callable.
  - `scraper_fast.post_transport(url: str, headers: dict, cookies: dict) -> tuple[int, str]` — same policy, HTTP POST.
  - `comic_scraper.ScrapeError(Exception)` — kept.
  - `comic_scraper.fetch_chapter_list(series_url: str) -> list[tuple[str, str]]` — `(slug, label)` in the site's own newest-first order; raises `ScrapeError` naming "challenge" on a non-200 or an interstitial.
  - `comic_scraper.chapter_url(series_url: str, slug: str) -> str`
  - `comic_scraper.page_urls(html: str) -> list[str]` — image URLs in page order, off-site URLs dropped.
  - `comic_scraper.safe_image_url(url: str) -> bool` — carried over from the replaced file unchanged.
  - `comic_scraper.image_ext(url: str) -> str` — `.jpg`/`.jpeg`/`.png`/`.webp`; unknown or missing extensions raise `ScrapeError`.
  - `comic_scraper._with_retries`, `post_transport` imported into the module namespace so tests can monkeypatch them.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scraper_fast.py  (append)
import scraper_fast as sf


def test_a_post_goes_through_the_same_retry_policy(monkeypatch):
    calls = []

    def flaky(url, headers, cookies):
        calls.append(url)
        return (503, "") if len(calls) < 3 else (200, "ok")

    monkeypatch.setattr(sf, "_raw_post_transport", flaky)
    monkeypatch.setattr(sf.time, "sleep", lambda _s: None)
    assert sf.post_transport("https://x/y", {}, {}) == (200, "ok")
    assert len(calls) == 3


def test_a_get_still_uses_the_shared_policy(monkeypatch):
    monkeypatch.setattr(sf, "_raw_transport", lambda u, h, c: (200, "body"))
    assert sf.default_transport("https://x/y", {}, {}) == (200, "body")
```

```python
# tests/test_comic_scraper.py  (replaces the file; these two blocks are the
# ones worth keeping from the version it replaces, and carry over verbatim)
import pytest

import comic_scraper as cs

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
    html = "".join(
        f'<img class="wp-manga-chapter-img" data-src="{u}">'
        for u in (
            "https://x.comic-site.test/a/c_1_1.jpg",
            "https://x.comic-site.test/a/ch_1_2.webp",
            "https://x.comic-site.test/a/67.webp",
            "https://x.comic-site.test/a/1-(2).jpeg",
            "https://x.comic-site.test/a/ch-259-img-3.jpg",
            "https://x.comic-site.test/a/9.png",
        )
    )
    assert len(cs.page_urls(html)) == 6


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scraper_fast.py tests/test_comic_scraper.py -v`
Expected: FAIL — `post_transport`, `fetch_chapter_list`, `page_urls`, `chapter_url`, `image_ext` do not exist.

- [ ] **Step 3: Implement**

In `src/scraper_fast.py`: move the body of `default_transport`'s loop into `_with_retries(fetch)` unchanged, and make `default_transport` a one-liner around it — `return _with_retries(lambda: _raw_transport(url, headers, cookies))`. Add `_raw_post_transport` mirroring `_raw_transport` but using `cffi.post`, and `post_transport` as `_with_retries(lambda: _raw_post_transport(url, headers, cookies))`.

In `src/comic_scraper.py`: `fetch_chapter_list` POSTs to `series_url.rstrip("/") + "/ajax/chapters/"` with `{"X-Requested-With": "XMLHttpRequest"}`, raises `ScrapeError` (mentioning "challenge" when `looks_like_challenge(body)`) if the status is not 200, then selects `a[href*=chapter]` with BeautifulSoup and returns `(href.rstrip("/").rsplit("/", 1)[-1], a.get_text(" ", strip=True))`. `page_urls` selects `img.wp-manga-chapter-img`, prefers `data-src` over `src`, and keeps only URLs passing `safe_image_url`. `image_ext` takes `urlparse(url).path`'s suffix, lowercases it, folds `.jpeg` to `.jpg`, and raises `ScrapeError` for anything else — a page whose extension cannot be determined must fail loudly rather than be saved under a guessed name.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scraper_fast.py tests/test_comic_scraper.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/scraper_fast.py src/comic_scraper.py tests/test_scraper_fast.py tests/test_comic_scraper.py
git commit -m "feat(comic): read the chapter list in one request, share the retry policy"
```

---

### Task 4: Average hash, so a re-upload can be recognized

**Files:**
- Create: `src/comic_images.py`
- Test: `tests/test_comic_images.py`

**Interfaces:**
- Consumes: `comic_reader.PAGE_EXTENSIONS`.
- Produces:
  - `looks_like_image(payload: bytes) -> bool` — magic-byte check.
  - `average_hash(payload: bytes) -> str` — 16×16 greyscale average hash as 64 hex characters; raises `ValueError` when the payload is not an image.
  - `hamming(a: str, b: str) -> int`
  - `DUPLICATE_DISTANCE: int = 10`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_comic_images.py
import io

import numpy as np
import pytest
from PIL import Image

from comic_images import (
    DUPLICATE_DISTANCE,
    average_hash,
    hamming,
    looks_like_image,
)


def _png(pixels) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.array(pixels, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_the_same_picture_hashes_identically():
    payload = _png(np.full((32, 32), 200))
    assert hamming(average_hash(payload), average_hash(payload)) == 0


def test_a_different_picture_hashes_further_than_the_threshold():
    white = average_hash(_png(np.full((32, 32), 255)))
    split = average_hash(_png(np.hstack([np.zeros((32, 16)), np.full((32, 16), 255)])))
    assert hamming(white, split) > DUPLICATE_DISTANCE


def test_a_rescale_of_the_same_picture_still_matches():
    plain = average_hash(_png(np.full((32, 32), 200)))
    bigger = average_hash(_png(np.full((128, 128), 200)))
    assert hamming(plain, bigger) <= DUPLICATE_DISTANCE


def test_an_html_error_page_is_not_an_image():
    assert not looks_like_image(b"<html>error</html>")


def test_each_real_format_is_recognised():
    for magic in (b"\xff\xd8\xff\xe0", b"\x89PNG\r\n\x1a\n", b"RIFF0000WEBP", b"GIF89a"):
        assert looks_like_image(magic + b"0" * 32)


def test_hashing_something_that_is_not_an_image_is_refused():
    with pytest.raises(ValueError):
        average_hash(b"<html>error</html>")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_comic_images.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'comic_images'`

- [ ] **Step 3: Implement `src/comic_images.py`**

`average_hash`: `Image.open(io.BytesIO(payload)).convert("L").resize((16, 16))`, threshold each pixel against the mean, pack the 256 booleans into an int, return `f"{value:064x}"`; wrap `Image.open`/`resize` failures in `ValueError`. `hamming`: `bin(int(a, 16) ^ int(b, 16)).count("1")`. `looks_like_image` reuses the magic-byte tuple already in `comic_scraper.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_comic_images.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/comic_images.py tests/test_comic_images.py
git commit -m "feat(comic): recognize a re-uploaded chapter by average hash"
```

---

### Task 5: Download one chapter, with an honest failure count

**Files:**
- Modify: `src/scraper_fast.py` (add `bytes_transport` beside `post_transport`)
- Modify: `src/comic_scraper.py`
- Test: `tests/test_comic_scraper.py` (append)

**Interfaces:**
- Consumes: `scraper_fast._with_retries`; `comic_images.average_hash`, `looks_like_image`; `comic_reader.page_filename`; `comic_manifest.ChapterRecord`.
- Produces:
  - `scraper_fast.bytes_transport(url: str, headers: dict, cookies: dict) -> tuple[int, bytes]` — same retry policy, raw body (needed because `default_transport` decodes with `resp.text`, which corrupts image bytes).
  - `comic_scraper.ChapterResult(record: ChapterRecord, saved: int, failed: int, on_disk: int)`
  - `comic_scraper.download_chapter(series_url: str, record: ChapterRecord, out_dir: Path, *, overwrite: bool = False, delay: float = 0.4) -> ChapterResult`
  - `comic_scraper.pages_on_disk(out_dir: Path, ordinal: int) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_comic_scraper.py  (append)

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_comic_scraper.py -v`
Expected: FAIL — `ChapterRecord` not imported, `download_chapter` missing.

- [ ] **Step 3: Implement**

`bytes_transport` mirrors `_raw_transport` but returns `resp.content`, going through `_with_retries`.

`download_chapter` fetches the chapter page with `default_transport` (raising `ScrapeError` on a challenge), calls `page_urls`, and for each URL at 1-based index `i` computes `page_filename(record.ordinal, i, image_ext(url))`. A page already on disk is skipped unless `overwrite`. Otherwise fetch with `bytes_transport` under `{"Referer": series_url}`, reject anything failing `looks_like_image`, and write the file. `time.sleep(delay)` between requests. Return counts of pages saved, pages failed, and total pages on disk for that ordinal. Set `record.pages` to the number of URLs the page listed.

Do **not** set `content_hash` here: the hash belongs to the scan pass (Task 6), which decides duplicates before any byte is written. Two writers for one field is how they drift.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_comic_scraper.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/scraper_fast.py src/comic_scraper.py tests/test_comic_scraper.py
git commit -m "feat(comic): download a chapter under its ordinal and count failures"
```

---

### Task 6: Scan for duplicates before writing, then run

**Files:**
- Modify: `src/comic_scraper.py`
- Test: `tests/test_comic_scraper.py` (append)

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `comic_scraper.scan_chapter(series_url: str, record: ChapterRecord) -> ChapterRecord` — fills `pages` and `content_hash` from page 1; raises `ScrapeError` on a challenge.
  - `comic_scraper.mark_duplicates(records: list[ChapterRecord]) -> None` — in place; for each identical hash the record with the most pages survives and every other gets `duplicate_of` set to its ordinal (first in reading order wins a tie).
  - `comic_scraper.has_legacy_files(out_dir: Path) -> bool`
  - `comic_scraper.run(series_url: str, out_dir: Path, manifest: Manifest, plan: list[ChapterRecord], *, overwrite: bool = False, delay: float = 0.4, dry_run: bool = False) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_comic_scraper.py  (append)
from comic_images import average_hash
from comic_manifest import Manifest


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


def test_a_scan_fills_the_page_count_and_hash(monkeypatch):
    page = "https://c4.comic-site.test/x/chapter_9/ch_9_1.jpg"
    monkeypatch.setattr(cs, "default_transport", lambda u, h, c: (200, _page_html(page, page + "?2")))
    monkeypatch.setattr(cs, "bytes_transport", lambda u, h, c: (200, JPEG))
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)

    scanned = cs.scan_chapter(SERIES, record)

    assert scanned.pages == 2
    assert scanned.content_hash == average_hash(JPEG)


def test_a_legacy_page_file_blocks_the_run(tmp_path, capsys):
    (tmp_path / "c_1_1.jpg").write_bytes(JPEG)
    record = ChapterRecord(ordinal=1, slug="a", label="Chapter 1", number=1.0)
    assert cs.has_legacy_files(tmp_path)
    assert cs.run(SERIES, tmp_path, Manifest("T", SERIES, [record]), [record]) == 1
    assert "unused/2026-11-01-pre-ordinal" in capsys.readouterr().out


def test_ordinal_named_files_are_not_legacy(tmp_path):
    (tmp_path / "c_000001_001.jpg").write_bytes(JPEG)
    assert not cs.has_legacy_files(tmp_path)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_comic_scraper.py -v`
Expected: FAIL — `mark_duplicates`, `scan_chapter`, `has_legacy_files`, `run` missing.

- [ ] **Step 3: Implement**

`scan_chapter` fetches the chapter page and page 1 only: it sets `record.pages = len(urls)` and `record.content_hash` from page 1's bytes (empty when page 1 cannot be fetched).

`mark_duplicates` groups records by non-empty `content_hash`, and within each group keeps a single winner (most pages, first in reading order on a tie) and sets `duplicate_of` to the winner's ordinal on every other member.

`has_legacy_files` is true when any file in `out_dir` matches `CHAPTER_PATTERNS` but its chapter digits are not exactly six characters long — that is the `c_7_1.jpg` shape the old script wrote, which the reader would merge into the new numbering.

`run` returns `1` immediately when `has_legacy_files`, printing the archive command for `unused/2026-11-01-pre-ordinal/`. Otherwise, for each record in `plan`: skip a record with `duplicate_of` set (printing the reason once), otherwise `download_chapter` and `save_manifest` after each chapter even when it failed, so an interrupted run keeps its ordinals. In `dry_run`, scan and report but write no pages. Finally print a per-chapter failure summary and return `1` if any non-duplicate chapter has `failed > 0` or ends with zero pages on disk, `0` otherwise.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_comic_scraper.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/comic_scraper.py tests/test_comic_scraper.py
git commit -m "feat(comic): settle duplicates before writing, resume without renumbering"
```

---

### Task 7: CLI, fractional label selection, and docs

**Files:**
- Modify: `src/comic_scraper.py` (`main`, argument parsing)
- Modify: `src/comic_config.json` (add `comic_series_url` and `comic_site_host`)
  > **Superseded:** the site keys do **not** belong in the tracked config. They live in the
  > gitignored `src/comic_site.json`, with `src/comic_site.example.json` tracked; the tracked
  > `comic_config.json` must never name a site. See `AGENTS.md` → "Config: the site is never
  > tracked". This was found only after the work shipped, when the repo turned out to be public.
- Modify: `README.md` (the "Downloading the pages" section)
- Test: `tests/test_comic_selection.py` (create)

**Interfaces:**
- Consumes: `scraper.parse_chapter_input` for ordinal selections; `comic_manifest.assign`, `load_manifest`, `save_manifest`.
- Produces:
  - `comic_scraper.parse_label_selection(text: str) -> list[float]` — `"175.1"`, `"175.1-175.19"`, `"144,150"`; raises `ValueError` on anything else.
  - `comic_scraper.normalise_label(text: str) -> str` — case-folded, `chapter` prefix stripped, whitespace collapsed to one space.
  - `comic_scraper.select_plan(manifest: Manifest, text: str | None) -> list[ChapterRecord]` — `None`/`"all"` for everything; site labels otherwise.
  - CLI: `-n/--name` (output folder), `--slug` (site slug, defaults to `--name`), `-c/--chapters` (site labels), `--ordinals` (manifest positions, parsed by `scraper.parse_chapter_input`), `--from/--to` (label bounds), `--all`, `--overwrite`, `--dry-run`, `--delay`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_comic_selection.py
import pytest

from comic_manifest import ChapterRecord, Manifest
from comic_scraper import parse_label_selection, select_plan


def test_a_fractional_label_is_accepted():
    assert parse_label_selection("175.1") == [175.1]


def test_a_fractional_range_is_accepted():
    assert parse_label_selection("175.1-175.5") == [175.1, 175.2, 175.3, 175.4, 175.5]


def test_a_plain_number_still_works():
    assert parse_label_selection("144,150") == [144.0, 150.0]


def test_a_whole_number_range_steps_by_one():
    assert parse_label_selection("174-176") == [174.0, 175.0, 176.0]


def test_garbage_is_refused():
    with pytest.raises(ValueError):
        parse_label_selection("chapter one")


def test_a_label_selects_its_chapter():
    manifest = Manifest(
        "T",
        "s",
        [
            ChapterRecord(1, "chapter-144", "Chapter 144", 144.0),
            ChapterRecord(2, "chapter-144.0", "Chapter 144.0", 144.0),
            ChapterRecord(3, "chapter-175-1", "Chapter 175.1", 175.1),
        ],
    )
    assert [r.ordinal for r in select_plan(manifest, "175.1")] == [3]
    assert [r.ordinal for r in select_plan(manifest, "144")] == [1]


def test_a_label_match_is_not_ambiguous_for_a_re_upload():
    # 144 and 144.0 both publish the number 144.0, so matching on the number
    # would select both. The label is the site's own identifier and is unique.
    manifest = Manifest(
        "T",
        "s",
        [
            ChapterRecord(1, "chapter-144", "Chapter 144", 144.0),
            ChapterRecord(2, "chapter-144.0", "Chapter 144.0", 144.0),
        ],
    )
    assert [r.ordinal for r in select_plan(manifest, "144.0")] == [2]


def test_a_selection_may_leave_out_the_chapter_word():
    manifest = Manifest("T", "s", [ChapterRecord(1, "a", "Chapter 175.1", 175.1)])
    assert [r.ordinal for r in select_plan(manifest, "175.1")] == [1]


def test_no_selection_means_everything():
    manifest = Manifest("T", "s", [ChapterRecord(1, "a", "Chapter 1", 1.0)])
    assert select_plan(manifest, None) == manifest.chapters
    assert select_plan(manifest, "all") == manifest.chapters


def test_a_label_that_is_not_on_the_site_is_refused_loudly():
    manifest = Manifest("T", "s", [ChapterRecord(1, "a", "Chapter 1", 1.0)])
    with pytest.raises(ValueError, match="not in the chapter list"):
        select_plan(manifest, "999")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_comic_selection.py -v`
Expected: FAIL with `ImportError: cannot import name 'parse_label_selection'`

- [ ] **Step 3: Implement and update docs**

`parse_label_selection` splits on commas; a part containing `-` is a range whose step is 0.1 when either end has a fractional part and 1.0 otherwise; expand inclusively and round to 3 decimals so float drift cannot produce `175.29999999999998`. `normalise_label` case-folds, strips a leading `chapter`, and collapses whitespace, so `"Chapter 175.1"` and `"175.1"` compare equal. `select_plan` matches a selection against `normalise_label(record.label)` first; only if nothing matches does it fall back to `record.number` within a tolerance of 0.05, and it raises `ValueError` naming the missing label when neither matches. Label-first is what makes `-c 144` and `-c 144.0` select different chapters.

`main` loads the config, resolves the series URL from `--slug` and `comic_series_url`, fetches the list, `assign`s and saves the manifest, prints the plan, then calls `run`. `--dry-run` scans and prints without writing pages.

~~`comic_config.json` gains `"comic_series_url": "https://comic-site.test/manga/{slug}/"` and `"comic_site_host": "comic-site.test"`, and `safe_image_url` reads the host from the config instead of hardcoding it.~~

**Superseded:** the site keys went to the gitignored `src/comic_site.json` instead, merged over the
tracked `comic_config.json` by `comic_reader.load_config`. `safe_image_url` still reads the host
from the config rather than hardcoding it, which remains the point; only *which* file holds it
changed. See `AGENTS.md` → "Config: the site is never tracked".

README replaces the walk-based description with: the one-request chapter list, the manifest, ordinal filenames, the archive step for legacy trees, the duplicate rule, and the fact that `-c` takes site labels while `summarize.py -c` takes ordinals.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest tests -v`
Expected: PASS, whole suite green.

- [ ] **Step 5: Commit**

```bash
git add src/comic_scraper.py src/comic_config.json README.md tests/test_comic_selection.py
git commit -m "feat(comic): site-label selection for the download stage, and document it"
```

---

### Task 8: Migrate the series and prove it end to end

**Files:**
- Modify: `AGENTS.md` (the pipeline section, which currently does not mention comics at all)
- No new code.

**Interfaces:**
- Consumes: everything above.
- Produces: nothing importable; this task produces *verified data* and accurate documentation.

- [ ] **Step 1: Archive the legacy trees**

Run:
```bash
mkdir -p unused/2026-11-01-pre-ordinal
mv comics/TBATE novels/comic-TBATE transcripts/comic-TBATE summaries/comic-TBATE unused/2026-11-01-pre-ordinal/
```
Expected: `comics/`, `novels/`, `transcripts/` and `summaries/` no longer contain a `TBATE`/`comic-TBATE` entry; `unused/` does. All four are gitignored, so `git status` stays clean.

- [ ] **Step 2: Download chapters 1–3 and the 175 block as a smoke test**

Run: `uv run python src/comic_scraper.py -n TBATE --slug the-beginning-after-the-end-series -c 1-3,175-175.2 --dry-run`
Expected: four chapters listed with their site labels and page counts matching the site (1 page for chapter 1, 54 for chapter 2, 32 for chapter 3); no page files written; exit 0.

Then run the same command without `--dry-run`.
Expected: `c_000001_001.jpg` … and no filename outside the `c_NNNNNN_NNN.ext` shape.

- [ ] **Step 3: Prove the reader groups what the scraper wrote**

Run: `uv run python src/comic_reader.py --series TBATE --comics comics/TBATE --only 1-2`
Expected: `novels/comic-TBATE/chapter_00001.txt` and `chapter_00002.txt`, the first beginning `# TBATE chapter 1`, and the second containing real dialogue rather than an empty body.

- [ ] **Step 4: Prove the summariser sees them in order**

Run: `uv run python src/summarize.py -n comic-TBATE --dry-run`
Expected: chapters 1 and 2 listed in ascending order with no gaps and no duplicates.

- [ ] **Step 5: Document the stage in `AGENTS.md`**

Add a third pipeline entry describing `src/comic_scraper.py` → `src/comic_reader.py` → `summarize.py`, the ordinal-vs-label distinction, the manifest, and the archive path. State that `comics/` uses `c_<ordinal:06d>_<page:03d>.<ext>` and that legacy `c_<n>_<p>` files must be archived before a run.

- [ ] **Step 6: Commit**

```bash
git add AGENTS.md
git commit -m "docs(comic): describe the download, read and summarise pipeline"
```

The full-series download is deliberately **not** part of this task: it is a multi-hour run the owner starts separately once the smoke test is green. Task 8 proves the shapes, the numbering and both consumers agree.
