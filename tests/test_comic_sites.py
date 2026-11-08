"""Several sites in one config, and per-series glossaries.

The point of these tests is that a *second* site and a *second* series are
addable without touching the first: the existing flat ``comic_site.json`` and the
flat ``glossary`` list must keep working exactly as they did.

Each test names the failure it prevents. The ones that matter most:

* A flat config resolving to a different site than before would silently
  download into a folder the user already has, from the wrong site.
* An ambiguous multi-site config picking one quietly would do the same.
* A series inheriting another series' glossary would make the validator accept
  the wrong names as known, which is a gate that reports success and is wrong.
"""

import json

import pytest

import comic_check
import comic_scraper as cs
from comic_text import resolve_glossary


# -- the flat, single-site config ------------------------------------------


def test_a_flat_config_still_resolves_to_one_unnamed_site(monkeypatch):
    """The original two-key file must behave identically.

    This is the compatibility contract: an existing install edits nothing.
    """

    monkeypatch.setattr(cs, "_ACTIVE_SITE", None)
    config = {
        "comic_series_url": "https://comic-site.test/manga/{slug}/",
        "comic_site_host": "comic-site.test",
    }
    site = cs.resolve_site(config)

    assert site.name is None, "a flat config has no site name to report"
    assert site.host == "comic-site.test"
    assert site.series_url == "https://comic-site.test/manga/{slug}/"


def test_a_flat_config_uses_todays_shape_defaults(monkeypatch):
    """The Madara defaults are what a site with no keys gets.

    If this drifts, every unconfigured site silently changes shape.
    """

    monkeypatch.setattr(cs, "_ACTIVE_SITE", None)
    site = cs.resolve_site({"comic_series_url": "https://a.test/{slug}/",
                            "comic_site_host": "a.test"})

    assert site.chapter_list_url == "{series_url}ajax/chapters/"
    assert site.chapter_list_method == "POST"
    assert site.chapter_list_headers == {"X-Requested-With": "XMLHttpRequest"}
    assert site.chapter_link_selector == "a[href*=chapter]"
    assert site.image_selector == "img.wp-manga-chapter-img"
    assert site.image_attribute == "data-src"


def test_the_default_list_url_is_the_one_that_used_to_be_hardcoded():
    """Guards the exact string the old code built by hand.

    ``fetch_chapter_list`` used to concatenate ``"/ajax/chapters/"``; a default
    that misses the trailing slash asks a different URL and 404s.
    """

    site = cs._site_from_entry({}, None)
    assert site.list_url("https://a.test/manga/x/") == (
        "https://a.test/manga/x/ajax/chapters/"
    )


# -- naming a site ---------------------------------------------------------


def test_site_name_selects_between_two_configured_sites(monkeypatch):
    """Two sites coexist; --site picks one.

    Before this, both keys were single-valued, so a second site had nowhere to
    go and keeping the first meant swapping files by hand.
    """

    monkeypatch.setattr(cs, "_ACTIVE_SITE", None)
    config = {
        "sites": {
            "alpha": {"comic_series_url": "https://alpha.test/{slug}/",
                      "comic_site_host": "alpha.test"},
            "beta": {"comic_series_url": "https://beta.test/{slug}/",
                     "comic_site_host": "beta.test"},
        },
        "default_site": "alpha",
    }

    assert cs.resolve_site(config).name == "alpha"
    assert cs.resolve_site(config, "beta").name == "beta"
    assert cs.resolve_site(config, "beta").host == "beta.test"


def test_a_single_configured_site_needs_no_default(monkeypatch):
    """One entry in ``sites`` is unambiguous, so it needs no ceremony."""

    monkeypatch.setattr(cs, "_ACTIVE_SITE", None)
    config = {"sites": {"only": {"comic_series_url": "https://o.test/{slug}/",
                                 "comic_site_host": "o.test"}}}
    assert cs.resolve_site(config).name == "only"


def test_an_ambiguous_config_is_refused_not_guessed(monkeypatch):
    """Picking one of several sites silently would fetch the wrong comic.

    The pages would look plausible and land in the folder the user named, so
    there is no later signal that it went wrong.
    """

    monkeypatch.setattr(cs, "_ACTIVE_SITE", None)
    config = {"sites": {name: {} for name in ("x", "y", "z")}}
    with pytest.raises(cs.ScrapeError, match="several sites are configured"):
        cs.resolve_site(config)


def test_an_unknown_site_name_names_the_ones_that_exist(monkeypatch):
    monkeypatch.setattr(cs, "_ACTIVE_SITE", None)
    config = {"sites": {"alpha": {}, "beta": {}}}
    with pytest.raises(cs.ScrapeError, match="alpha, beta"):
        cs.resolve_site(config, "gamma")


def test_a_site_name_against_a_flat_config_is_refused(monkeypatch):
    """A named site with no map would silently fetch from the flat site.

    The user believes they chose a site; the run uses another. The pages land in
    the folder they named and look plausible, so nothing later reveals it.
    """

    monkeypatch.setattr(cs, "_ACTIVE_SITE", None)
    config = {"comic_series_url": "https://flat.test/{slug}/",
              "comic_site_host": "flat.test"}
    with pytest.raises(cs.ScrapeError, match="no 'sites' map"):
        cs.resolve_site(config, "beta")


def test_shape_keys_override_only_what_they_set(monkeypatch):
    """A site that only differs on the image attribute keeps the rest.

    All-or-nothing would force every site to restate the Madara defaults.
    """

    monkeypatch.setattr(cs, "_ACTIVE_SITE", None)
    config = {"sites": {"plain": {
        "comic_series_url": "https://p.test/{slug}/",
        "comic_site_host": "p.test",
        "image_attribute": "src",
    }}}
    site = cs.resolve_site(config, "plain")

    assert site.image_attribute == "src"
    assert site.image_selector == "img.wp-manga-chapter-img"
    assert site.chapter_list_method == "POST"


def test_a_sites_map_with_no_entries_falls_back_to_the_flat_keys(monkeypatch):
    """An empty ``sites`` object must not shadow a working flat config."""

    monkeypatch.setattr(cs, "_ACTIVE_SITE", None)
    config = {"sites": {},
              "comic_series_url": "https://flat.test/{slug}/",
              "comic_site_host": "flat.test"}
    site = cs.resolve_site(config)

    assert site.name is None
    assert site.host == "flat.test"


# -- the shape drives the fetch -------------------------------------------


def test_a_get_site_fetches_the_list_without_ajax_headers(monkeypatch):
    """A non-Madara site lists chapters over a plain GET.

    The old code could only POST, so such a site returned ``0`` and read as an
    empty series rather than a wrong request.
    """

    monkeypatch.setattr(cs, "_ACTIVE_SITE", None)
    site = cs._site_from_entry(
        {"comic_series_url": "https://g.test/{slug}/",
         "comic_site_host": "g.test",
         "chapter_list_method": "GET"}, "g")

    calls = []

    def fake_get(url, headers, cookies):
        calls.append(("GET", url, headers))
        return 200, '<a href="/c/chapter-1/">Chapter 1</a>'

    def fake_post(url, headers, cookies):
        raise AssertionError("a GET site must not be POSTed to")

    monkeypatch.setattr(cs, "default_transport", fake_get)
    monkeypatch.setattr(cs, "post_transport", fake_post)

    assert cs.fetch_chapter_list("https://g.test/series/", site) == [
        ("chapter-1", "Chapter 1")
    ]
    assert calls[0][0] == "GET"


def test_a_custom_list_url_is_the_one_actually_fetched(monkeypatch):
    """A site whose list lives elsewhere must be fetched from there.

    Asserting the *default* URL would be tautological: hardcoding the Madara
    path produces the same string. This drives a non-default template through
    ``fetch_chapter_list`` and checks the URL that was requested.
    """

    monkeypatch.setattr(cs, "_ACTIVE_SITE", None)
    site = cs._site_from_entry(
        {"comic_series_url": "https://c.test/comic/{slug}/",
         "comic_site_host": "c.test",
         "chapter_list_url": "{series_url}chapter-list.json"}, "c")

    requested = []

    def fake_post(url, headers, cookies):
        requested.append(url)
        return 200, '<a href="/c/chapter-1/">One</a>'

    monkeypatch.setattr(cs, "post_transport", fake_post)

    cs.fetch_chapter_list("https://c.test/comic/some-series/", site)

    assert requested == ["https://c.test/comic/some-series/chapter-list.json"], (
        "the configured chapter_list_url was ignored"
    )


def test_the_chapter_link_selector_comes_from_the_site(monkeypatch):
    """A site whose list uses a different anchor is config, not code.

    The decoy matters: its href *contains* "chapter", so the Madara default
    selector would pick it up. That is what makes this test able to tell the two
    selectors apart instead of passing either way.
    """

    monkeypatch.setattr(cs, "_ACTIVE_SITE", None)
    site = cs._site_from_entry(
        {"comic_series_url": "https://l.test/{slug}/",
         "comic_site_host": "l.test",
         "chapter_link_selector": "li.chapter a"}, "l")

    body = (
        '<li class="chapter"><a href="/c/chapter-2/">Two</a></li>'
        '<a href="/announcements/chapter-delays/">Delays</a>'
    )
    monkeypatch.setattr(cs, "post_transport", lambda u, h, c: (200, body))

    assert cs.fetch_chapter_list("https://l.test/s/", site) == [("chapter-2", "Two")]


def test_the_image_selector_and_attribute_come_from_the_site(monkeypatch):
    """A plain ``<img src>`` site yields its pages instead of nothing."""

    monkeypatch.setattr(cs, "_ACTIVE_SITE", None)
    monkeypatch.setattr(cs, "_SITE_HOST", None)
    site = cs._site_from_entry(
        {"comic_series_url": "https://s.test/{slug}/",
         "comic_site_host": "s.test",
         "image_selector": "div.pages img",
         "image_attribute": "src"}, "s")
    cs.use_site(site)

    html = (
        '<div class="pages"><img src="https://s.test/p/1.jpg"></div>'
        '<img class="wp-manga-chapter-img" data-src="https://s.test/p/2.jpg">'
    )
    # The second image matches the *default* selector, not this site's, so it is
    # correctly ignored: the site said where its pages live.
    assert cs.page_urls(html, site) == ["https://s.test/p/1.jpg"]


# -- the host guard follows the selected site -----------------------------


def test_the_host_guard_narrows_to_the_selected_site(monkeypatch):
    """With two sites configured, the active one's host is the only one allowed.

    A guard that accepted either host would let site A's page fetch site B's
    images, which is how an ad host gets in.
    """

    monkeypatch.setattr(cs, "_ACTIVE_SITE", None)
    monkeypatch.setattr(cs, "_SITE_HOST", None)
    cs.use_site(cs._site_from_entry(
        {"comic_series_url": "https://a.test/{slug}/", "comic_site_host": "a.test"},
        "a"))

    assert cs.safe_image_url("https://cdn.a.test/p/1.jpg")
    assert not cs.safe_image_url("https://cdn.b.test/p/1.jpg")


# -- per-series glossaries -------------------------------------------------


def test_a_per_series_glossary_is_selected_by_name():
    """Two series keep separate names in one config."""

    config = {
        "glossary": ["FLAT"],
        "glossaries": {"TBATE": ["ARTHUR"], "OTHER": ["KAI"]},
    }
    assert resolve_glossary(config, "TBATE") == ["ARTHUR"]
    assert resolve_glossary(config, "OTHER") == ["KAI"]


def test_the_series_name_matches_with_or_without_the_comic_prefix():
    """``--series TBATE`` and the folder ``comic-TBATE`` must both find it.

    The reader takes a bare name while the transcript directory carries the
    prefix; requiring one spelling would be a silent miss.
    """

    config = {"glossaries": {"TBATE": ["ARTHUR"]}}
    assert resolve_glossary(config, "TBATE") == ["ARTHUR"]
    assert resolve_glossary(config, "comic-TBATE") == ["ARTHUR"]
    assert resolve_glossary(config, "tbate") == ["ARTHUR"]


def test_a_series_with_no_entry_falls_back_to_the_flat_list():
    """This is what makes the change backwards compatible."""

    config = {"glossary": ["ARTHUR"], "glossaries": {"OTHER": ["KAI"]}}
    assert resolve_glossary(config, "BRAND-NEW") == ["ARTHUR"]


def test_no_series_at_all_reads_only_the_flat_list():
    """A caller that does not know the series must not get another's names."""

    config = {"glossary": ["ARTHUR"], "glossaries": {"OTHER": ["KAI"]}}
    assert resolve_glossary(config) == ["ARTHUR"]


def test_a_default_glossary_is_shared_and_overridden_per_series():
    """Shared terms serve every series that has no list of its own."""

    config = {"glossaries": {"default": ["SHARED"], "TBATE": ["ARTHUR"]}}
    assert resolve_glossary(config, "TBATE") == ["ARTHUR"]
    assert resolve_glossary(config, "NEW") == ["SHARED"]


def test_a_null_glossary_entry_is_dropped_not_read_as_a_name():
    """``null`` must not become the protected term ``"None"``.

    ``str(None)`` is ``"None"``, so filtering on the stringified value invents a
    glossary term that then suppresses a real misreading.
    """

    assert resolve_glossary({"glossary": ["A", None, "", "  ", "B"]}) == ["A", "B"]


def test_a_malformed_glossaries_value_does_not_abort_the_run():
    """A typo costs one series its names, not the whole transcript."""

    assert resolve_glossary({"glossary": ["A"], "glossaries": "nope"}, "TBATE") == ["A"]


def test_the_validator_and_the_reader_resolve_the_same_glossary(tmp_path):
    """They must agree, or a protected name is reported as noise forever.

    The reader protects a term while segmenting; the validator then checks the
    transcript for terms it does not know. One definition serves both.
    """

    config = tmp_path / "comic_config.json"
    config.write_text(json.dumps({
        "glossary": ["FLAT"],
        "glossaries": {"TBATE": ["ARTHUR"]},
    }), encoding="utf-8")

    assert comic_check.load_glossary(config, "comic-TBATE") == ["ARTHUR"]
    assert comic_check.load_glossary(config) == ["FLAT"]
