"""The catalogue lister: what a site publishes, read without downloading it.

The failures these tests exist for are all of the "looks like success" kind,
because a listing has no downstream stage to disagree with it:

* A Cloudflare interstitial or a 5xx read as "no more pages" would truncate the
  catalogue to whatever was fetched before the block, and the file would still
  parse and carry a plausible count.
* Counting ``/manga/`` links instead of grid cards over-reports by five a page
  (17 links against 12 cards, measured), so the count would be wrong while every
  element looked right.
* A capped ``--pages`` run written as complete is a trial that later reads as the
  whole site.
* A failed walk overwriting the previous good catalogue destroys the only copy
  of a list that cost ~950 requests to build.
"""

import json

import pytest

import comic_catalog as cc
import comic_scraper as cs


SITE = cs.Site(
    series_url="https://comic-site.test/manga/{slug}/",
    host="comic-site.test",
    name="alpha",
)


def page_html(cards, sidebar=()):
    """A catalogue page shaped like the theme's: cards, plus stray manga links."""

    card_html = "".join(
        f'<div class="page-item-detail"><h3 class="h5">'
        f'<a href="https://comic-site.test/manga/{slug}/">{title}</a>'
        f"</h3></div>"
        for slug, title in cards
    )
    sidebar_html = "".join(
        f'<a href="https://comic-site.test/manga/{slug}/">{slug}</a>'
        for slug in sidebar
    )
    return f'<html><body><div class="grid">{card_html}</div>{sidebar_html}</body></html>'


# -- what one page yields --------------------------------------------------


def test_a_page_yields_the_slug_and_title_of_each_card():
    html = page_html([("mood-disorder", "Mood Disorder"),
                      ("celeb-lady", "Celeb Lady")])

    assert cc.parse_catalogue_page(html, SITE) == [
        ("mood-disorder", "Mood Disorder"),
        ("celeb-lady", "Celeb Lady"),
    ]


def test_sidebar_links_are_not_counted_as_series():
    """A page has 17 ``/manga/`` links but 12 cards.

    Counting links would over-report by five on every page -- measured on the
    real site, so this is the exact shape rather than a guess.
    """

    html = page_html(
        [("mood-disorder", "Mood Disorder")],
        sidebar=["popular-one", "popular-two", "popular-three"],
    )

    assert cc.parse_catalogue_page(html, SITE) == [("mood-disorder", "Mood Disorder")]


def test_the_slug_comes_from_the_href_not_the_title():
    """Titles are not unique; the slug is what the downloader needs.

    A series and its sequel can both render as the same title, so a slug derived
    from the title would point at the wrong page.
    """

    html = page_html([("series-season-2", "Series")])

    assert cc.parse_catalogue_page(html, SITE) == [("series-season-2", "Series")]


def test_a_card_without_an_href_is_skipped_rather_than_invented():
    """A made-up slug lists a series that cannot then be downloaded."""

    html = (
        '<div class="page-item-detail"><h3 class="h5"><a>No Link</a></h3></div>'
        '<div class="page-item-detail"><h3 class="h5">'
        '<a href="https://comic-site.test/manga/real/">Real</a></h3></div>'
    )

    assert cc.parse_catalogue_page(html, SITE) == [("real", "Real")]


# -- walking ---------------------------------------------------------------


class Recorder:
    """A transport stand-in that answers per page and counts the asks."""

    def __init__(self, pages):
        self.pages = pages
        self.asked = []
        self.fail_at = None
        self.challenge_at = None

    def __call__(self, url, headers, cookies):
        # The catalogue URL carries a query string (?m_orderby=title), so the page
        # number comes from the path, not from the end of the whole URL.
        path = url.split("?", 1)[0].rstrip("/")
        page = int(path.rsplit("/", 1)[-1])
        self.asked.append(page)
        if page == self.challenge_at:
            # The genuine interstitial: ``looks_like_challenge`` keys on these
            # scripts, deliberately NOT on the visible phrase "Just a moment...",
            # which real chapter prose also contains.
            return 200, (
                '<html><head><script>window._cf_chl_opt = {}</script></head>'
                "<body>Just a moment...</body></html>"
            )
        if page == self.fail_at:
            return 503, ""
        return self.pages.get(page, (404, ""))


def site_transport(monkeypatch, recorder):
    monkeypatch.setattr(cc, "default_transport", recorder)


@pytest.fixture(autouse=True)
def local_site(monkeypatch):
    """Keep every test off the real ``comic_site.json``.

    Without this, a test that goes through ``main`` reads the developer's actual
    site -- and prints its host, which is the value this repo must not publish.
    The fixture makes that impossible rather than relying on each test to
    remember.
    """

    monkeypatch.setattr(cc.cs, "load_config", lambda path=None: {
        "sites": {"alpha": {
            # Not the placeholder: ``ensure_configured`` refuses that by design,
            # so a fixture using it would make every test take the refusal path.
            "comic_series_url": "https://alpha.test/manga/{slug}/",
            "comic_site_host": "alpha.test",
        }},
        "default_site": "alpha",
    })
    monkeypatch.setattr(cc.cs, "_ACTIVE_SITE", None)


def test_the_walk_stops_at_the_first_empty_page(monkeypatch):
    recorder = Recorder({
        1: (200, page_html([("a", "A"), ("b", "B")])),
        2: (200, page_html([("c", "C")])),
        3: (404, ""),
    })
    site_transport(monkeypatch, recorder)

    result = cc.walk(SITE, delay=0)

    assert [s.slug for s in result.series] == ["a", "b", "c"]
    assert result.last_page == 2, "the last page with content, not the page that ended it"
    assert result.stalled_pages == []
    assert recorder.asked[:3] == [1, 2, 3]


def test_a_duplicate_across_pages_is_listed_once(monkeypatch):
    """4 of 90 sampled cards repeated across pages on the real site."""

    recorder = Recorder({
        1: (200, page_html([("a", "A"), ("b", "B")])),
        2: (200, page_html([("b", "B"), ("c", "C")])),
        3: (404, ""),
    })
    site_transport(monkeypatch, recorder)

    result = cc.walk(SITE, delay=0)

    assert [s.slug for s in result.series] == ["a", "b", "c"]
    assert result.stalled_pages == []


def test_a_fully_repeated_page_is_retried_and_the_new_series_kept(monkeypatch):
    """Measured: pages 61-63 of a real walk served 12/12 duplicates, then changed.

    Without the retry, ~36 series vanish and the count still looks plausible.
    """

    stale = page_html([("a", "A"), ("b", "B")])
    recorder = Recorder({
        1: (200, page_html([("a", "A"), ("b", "B")])),
        # First ask for page 2 repeats page 1 entirely; the retry is real content.
        2: (200, page_html([("c", "C"), ("d", "D")])),
        3: (404, ""),
    })
    calls = {"n": 0}
    original = recorder.__call__

    def flaky(url, headers, cookies):
        if url.split("?", 1)[0].endswith("/page/2/"):
            calls["n"] += 1
            if calls["n"] == 1:
                recorder.asked.append(2)
                return 200, stale
        return original(url, headers, cookies)

    monkeypatch.setattr(cc, "default_transport", flaky)

    result = cc.walk(SITE, delay=0)

    assert [s.slug for s in result.series] == ["a", "b", "c", "d"]
    assert result.stalled_pages == [], "the retry recovered the page"
    assert calls["n"] == 2, "the stalled page must actually be re-fetched"


def test_a_page_that_stays_stalled_is_recorded_as_a_hole(monkeypatch):
    """If the retry also repeats, the series are genuinely missing.

    The walk continues (later pages were observed to resume normally) but the
    hole is reported, so the file is not written as complete.
    """

    stale = page_html([("a", "A")])
    recorder = Recorder({
        1: (200, stale),
        2: (200, page_html([("b", "B")])),
        3: (404, ""),
    })
    original = recorder.__call__

    def always_stale_on_2(url, headers, cookies):
        if url.split("?", 1)[0].endswith("/page/2/"):
            recorder.asked.append(2)
            return 200, stale
        return original(url, headers, cookies)

    monkeypatch.setattr(cc, "default_transport", always_stale_on_2)

    result = cc.walk(SITE, delay=0)

    assert result.stalled_pages == [2]
    assert [s.slug for s in result.series] == ["a"], "page 3 held nothing new either"


def test_a_cloudflare_challenge_raises_instead_of_ending_the_walk(monkeypatch):
    """The one failure that reads as success: a block is not the end of the list.

    Without this, a run blocked on page 2 writes a two-series catalogue and
    reports it as complete.
    """

    recorder = Recorder({
        1: (200, page_html([("a", "A")])),
        2: (200, page_html([("b", "B")])),
    })
    recorder.challenge_at = 2
    site_transport(monkeypatch, recorder)

    with pytest.raises(cs.ScrapeError, match="Cloudflare challenge"):
        cc.walk(SITE, delay=0)


def test_a_server_error_raises_instead_of_ending_the_walk(monkeypatch):
    recorder = Recorder({1: (200, page_html([("a", "A")]))})
    recorder.fail_at = 2
    site_transport(monkeypatch, recorder)

    with pytest.raises(cs.ScrapeError, match="HTTP 503"):
        cc.walk(SITE, delay=0)


def test_a_capped_walk_stops_after_that_many_pages(monkeypatch):
    recorder = Recorder({
        p: (200, page_html([(f"s{p}", f"S{p}")])) for p in range(1, 10)
    })
    site_transport(monkeypatch, recorder)

    result = cc.walk(SITE, delay=0, max_pages=3)

    assert recorder.asked == [1, 2, 3], "a cap must not read the next page"
    assert result.last_page == 3
    assert len(result.series) == 3


def test_a_capped_walk_does_not_probe_for_the_end(monkeypatch):
    """Otherwise the cap costs a request and reports a page it did not read."""

    recorder = Recorder({p: (200, page_html([(f"s{p}", f"S{p}")]))
                         for p in range(1, 10)})
    site_transport(monkeypatch, recorder)

    result = cc.walk(SITE, delay=0, max_pages=1)

    assert recorder.asked == [1]
    assert result.last_page == 1


def test_a_stalled_page_makes_the_run_incomplete(monkeypatch, tmp_path, capsys):
    """A hole must not be written as a complete catalogue.

    Measured: three consecutive stalled pages cost ~36 series while the run
    still reported a clean finish.
    """

    stale = page_html([("a", "A")])
    recorder = Recorder({1: (200, stale), 2: (200, page_html([("b", "B")])),
                         3: (404, "")})
    original = recorder.__call__

    def stale_on_2(url, headers, cookies):
        if url.split("?", 1)[0].endswith("/page/2/"):
            recorder.asked.append(2)
            return 200, stale
        return original(url, headers, cookies)

    monkeypatch.setattr(cc, "default_transport", stale_on_2)
    out = tmp_path / "cat.json"

    cc.main(["--out", str(out), "--delay", "0"])

    payload = cc.load(out)
    assert payload["complete"] is False
    assert payload["stalled_pages"] == [2]
    err = capsys.readouterr().err.lower()
    assert "warning" in err, "a hole must not be reported as a clean run"
    assert "no new series" in err and "incomplete" in err


def test_an_unstalled_run_lists_no_stalled_pages(monkeypatch, tmp_path):
    """Empty is written explicitly, so a reader can tell it from an old file."""

    recorder = Recorder({1: (200, page_html([("a", "A")])), 2: (404, "")})
    site_transport(monkeypatch, recorder)
    out = tmp_path / "cat.json"

    cc.main(["--out", str(out), "--delay", "0"])

    payload = cc.load(out)
    assert payload["complete"] is True
    assert payload["stalled_pages"] == []


def test_a_walk_that_starts_late_is_not_complete(monkeypatch, tmp_path):
    """``--first-page 100`` alone lists pages 100+ -- not the catalogue.

    Measured on the real site, the tail of one page repeats as the head of the
    next, so a page is not a fixed stride through the list. Starting mid-way is
    therefore a partial listing, and writing it as complete would make a
    fragment of the site's catalogue look like all of it.
    """

    recorder = Recorder({
        100: (200, page_html([("a", "A")])),
        101: (404, ""),
    })
    site_transport(monkeypatch, recorder)
    out = tmp_path / "cat.json"

    cc.main(["--first-page", "100", "--out", str(out), "--delay", "0"])

    assert cc.load(out)["complete"] is False


def test_the_walk_carries_the_series_url_from_the_config(monkeypatch):
    """One source of truth for the URL shape, so a site differing on it works."""

    recorder = Recorder({1: (200, page_html([("a", "A")])), 2: (404, "")})
    site_transport(monkeypatch, recorder)

    result = cc.walk(SITE, delay=0)

    assert result.series[0].url == "https://comic-site.test/manga/a/"


def test_a_checkpoint_is_written_during_a_long_walk(monkeypatch, tmp_path):
    """A crash at page 900 must not discard the other 899."""

    recorder = Recorder({p: (200, page_html([(f"s{p}", f"S{p}")]))
                         for p in range(1, 60)})
    site_transport(monkeypatch, recorder)
    seen = []

    cc.walk(SITE, delay=0, checkpoint=lambda series, page: seen.append((len(series), page)))

    assert seen, "no checkpoint during a walk longer than the interval"
    assert seen[0][1] == cc.CHECKPOINT_EVERY


# -- the URL the walk builds ----------------------------------------------


def test_the_catalogue_url_is_derived_from_the_series_url():
    """Sites differ on their path prefix, so it is stripped, not hardcoded."""

    site = cs.Site(series_url="https://other.test/comics/{slug}/",
                   host="other.test")

    assert site.series_base() == "https://other.test/comics/"
    assert site.catalogue_url(3) == (
        "https://other.test/comics/page/3/?m_orderby=title"
    )


def test_the_default_catalogue_url_matches_the_theme_measured():
    """Guards the string that was checked against the live site.

    The ``?m_orderby=title`` is load-bearing: the default order is "latest
    update", so a series bumped mid-walk shifts every later page and the walk
    collects duplicates instead of series.
    """

    assert SITE.catalogue_url(1) == (
        "https://comic-site.test/manga/page/1/?m_orderby=title"
    )


def test_the_catalogue_url_can_be_overridden_for_a_site_with_no_ordering():
    """A site that does not understand ``m_orderby`` must be able to say so."""

    site = cs._site_from_entry(
        {
            "comic_series_url": "https://plain.test/manga/{slug}/",
            "comic_site_host": "plain.test",
            "series_list_url": "{series_base}page/{page}/",
        },
        "plain",
    )

    assert site.catalogue_url(2) == "https://plain.test/manga/page/2/"


# -- writing ---------------------------------------------------------------


def test_a_capped_run_is_written_as_incomplete(monkeypatch, tmp_path):
    """A trial written as complete is the whole site's list as far as a reader knows."""

    recorder = Recorder({1: (200, page_html([("a", "A")])), 2: (404, "")})
    site_transport(monkeypatch, recorder)
    out = tmp_path / "cat.json"

    cc.main(["--pages", "1", "--out", str(out), "--delay", "0"])

    assert cc.load(out)["complete"] is False


def test_an_uncapped_run_is_written_as_complete(monkeypatch, tmp_path):
    recorder = Recorder({1: (200, page_html([("a", "A")])), 2: (404, "")})
    site_transport(monkeypatch, recorder)
    out = tmp_path / "cat.json"

    cc.main(["--out", str(out), "--delay", "0"])

    payload = cc.load(out)
    assert payload["complete"] is True
    assert payload["count"] == 1
    assert payload["pages"] == 1


def test_a_failed_walk_leaves_the_previous_catalogue_intact(monkeypatch, tmp_path):
    """A failure before the first checkpoint must not destroy the old file.

    The old file cost ~950 requests to build, so a run that dies early has to
    leave it alone -- and, when it dies *after* a checkpoint, has to say so
    instead of claiming the file is untouched.
    """

    out = tmp_path / "cat.json"
    out.write_text(json.dumps({"count": 7, "series": ["previous"]}), encoding="utf-8")

    recorder = Recorder({1: (200, page_html([("a", "A")]))})
    recorder.fail_at = 2
    site_transport(monkeypatch, recorder)

    code = cc.main(["--out", str(out), "--delay", "0"])

    assert code == 1
    assert cc.load(out)["count"] == 7, "the previous catalogue was overwritten"


def test_a_write_that_dies_midway_leaves_the_old_file_intact(monkeypatch, tmp_path):
    """The rename is what makes the write atomic, not the writing.

    A crash between "serialise" and "publish" must not leave a half-written
    catalogue where the previous good one was: the old file cost ~950 requests,
    and a truncated JSON that does not parse is unrecoverable.
    """

    out = tmp_path / "cat.json"
    out.write_text(json.dumps({"count": 7}), encoding="utf-8")

    def explode(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(cc.os, "replace", explode)

    with pytest.raises(OSError):
        cc.save(out, SITE, [cc.Series("a", "A", "https://alpha.test/manga/a/")],
                1, complete=True)

    assert cc.load(out)["count"] == 7, "the old file was truncated in place"


def test_the_temporary_file_is_not_left_behind(monkeypatch, tmp_path):
    """A stray ``.tmp`` beside the catalogue looks like a second catalogue."""

    out = tmp_path / "cat.json"

    cc.save(out, SITE, [cc.Series("a", "A", "https://alpha.test/manga/a/")],
            1, complete=True)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["cat.json"]


def test_a_failure_after_a_checkpoint_admits_the_file_is_partial(
        monkeypatch, tmp_path, capsys):
    """The checkpoint has already replaced the file; say so rather than lie.

    Silence would leave a partial catalogue on disk with the user believing the
    previous complete one is still there.
    """

    out = tmp_path / "cat.json"
    out.write_text(json.dumps({"count": 7, "series": ["previous"]}), encoding="utf-8")

    recorder = Recorder({p: (200, page_html([(f"s{p}", f"S{p}")]))
                         for p in range(1, 60)})
    recorder.fail_at = 55
    site_transport(monkeypatch, recorder)

    code = cc.main(["--out", str(out), "--delay", "0"])

    captured = capsys.readouterr()
    assert code == 1
    assert "PARTIAL" in captured.err
    assert "left as it was" not in captured.err
    payload = cc.load(out)
    assert payload["checkpoint"] is True
    assert payload["count"] < 54, "the file should hold only the checkpointed pages"


def test_an_unconfigured_site_is_refused_before_any_request(monkeypatch, tmp_path):
    """Same refusal as the downloader, not an ``HTTP 0`` from a placeholder host."""

    def explode(*args, **kwargs):
        raise AssertionError("a request was made for an unconfigured site")

    monkeypatch.setattr(cc, "default_transport", explode)
    monkeypatch.setattr(cc.cs, "load_config", lambda path=None: {})

    code = cc.main(["--out", str(tmp_path / "cat.json")])

    assert code == 1


# -- naming the output file -----------------------------------------------


def test_a_hostile_site_name_cannot_choose_the_write_location(tmp_path):
    """A config is a file a user edits; ``../`` must not escape the directory."""

    path = cc.catalog_path("../../etc/passwd")

    assert ".." not in path.name
    assert path.parent.name == "_catalog"


def test_the_site_name_is_reported_instead_of_the_host(monkeypatch, tmp_path):
    """The host is the value this repo must never publish, stdout included."""

    recorder = Recorder({1: (200, page_html([("a", "A")])), 2: (404, "")})
    site_transport(monkeypatch, recorder)
    out = tmp_path / "cat.json"

    cc.main(["--out", str(out), "--delay", "0"])

    payload = cc.load(out)
    assert payload["site"] == "alpha"
    # The URL is in the file on purpose (it is gitignored), but the run's own
    # reporting names the site, so a pasted transcript carries no host.
    assert payload["series"][0]["url"].startswith("https://alpha.test/")


def test_no_host_reaches_stdout(monkeypatch, tmp_path, capsys):
    """AGENTS.md: the site must not reach tracked content or a pasted transcript.

    Terminal output is the easiest place for it to leak, because the run prints
    the URL it is fetching. Progress therefore reports the site name and the
    *path*, never the host.
    """

    recorder = Recorder({1: (200, page_html([("a", "A")])), 2: (404, "")})
    site_transport(monkeypatch, recorder)

    cc.main(["--out", str(tmp_path / "cat.json"), "--delay", "0"])

    captured = capsys.readouterr()
    assert "alpha.test" not in captured.out
    assert "alpha.test" not in captured.err
    assert "alpha" in captured.out, "the site must still be identifiable by name"
    assert "/manga/page/1/" in captured.out, "the path is what progress reports"


def test_no_host_reaches_stdout_on_the_refusal_path(monkeypatch, tmp_path, capsys):
    """A refusal is printed too -- an error message may not name the host either."""

    monkeypatch.setattr(cc.cs, "load_config", lambda path=None: {})

    cc.main(["--out", str(tmp_path / "cat.json")])

    captured = capsys.readouterr()
    assert "alpha.test" not in captured.out + captured.err

