"""Tests for the paginated HTTP chapter-list harvest mode."""

import pytest

from scraper_parse import ChapterRef
from scraper_session import ChallengeError

from scraper_fast import make_session


def _site(**overrides):
    site = {
        "name": "Example Site",
        "base_url": "https://example.com/",
        "harvest": "paginated",
        "fast_chapter_url": "https://example.com/book/{novel_slug}/{slug}",
        "chapter_list_page_url": "https://example.com/book/{novel_slug}/chapters",
        "chapter_list_link_selector": ".chapter-list a",
        "chapter_list_next_selector": "a[rel=next]",
        "content_selector": "#content",
        "chapter_title_selector": ".chapter-title",
    }
    site.update(overrides)
    return site


_P1 = """
<html><body>
<ul class="chapter-list">
  <li><a href="/book/novel/chapter-1" title="c-1: Prologue: Start">c-1: Prologue: Start</a></li>
  <li><a href="/book/novel/chapter-2" title="Chapter 1: The Light"></a></li>
</ul>
<a rel="next" href="https://example.com/book/novel/chapters?page=2">Next</a>
</body></html>
"""

_P2 = """
<html><body>
<ul class="chapter-list">
  <li><a href="/book/novel/chapter-3">Chapter 2: Next</a></li>
</ul>
</body></html>
"""


def _transport_for(pages):
    calls = []

    def transport(url, headers, cookies):
        calls.append(url)
        return 200, pages.get(url, "<html></html>")

    return transport, calls


def test_paginated_harvest_follows_next_links(tmp_path):
    """The list harvest walks every page, collecting (slug, title) pairs."""
    from scraper_session import harvest_paginated

    pages = {
        "https://example.com/book/novel/chapters": _P1,
        "https://example.com/book/novel/chapters?page=2": _P2,
    }
    transport, calls = _transport_for(pages)

    data = harvest_paginated(_site(), "novel", transport)

    assert calls == [
        "https://example.com/book/novel/chapters",
        "https://example.com/book/novel/chapters?page=2",
    ]
    assert data.chapter_map == [
        ("chapter-1", "c-1: Prologue: Start"),
        ("chapter-2", "Chapter 1: The Light"),
        ("chapter-3", "Chapter 2: Next"),
    ]
    assert data.user_agent == ""
    assert data.cookies == {}


def test_paginated_harvest_uses_title_attribute_over_text(tmp_path):
    """Anchors with an explicit title attribute win over (possibly empty) text."""
    from scraper_session import harvest_paginated

    transport, _ = _transport_for({
        "https://example.com/book/novel/chapters": """
        <html><body><ul class="chapter-list">
          <li><a href="/book/novel/chapter-1" title="c-1: Prologue: Start"></a></li>
          <li><a href="/book/novel/chapter-2">Chapter 1: The Light</a></li>
        </ul></body></html>
        """
    })

    data = harvest_paginated(_site(), "novel", transport)
    # title attribute used when present, text when absent — both included
    assert data.chapter_map == [
        ("chapter-1", "c-1: Prologue: Start"),
        ("chapter-2", "Chapter 1: The Light"),
    ]


def test_paginated_harvest_skips_anchors_without_usable_title(tmp_path):
    """An anchor with neither title attribute nor text is dropped, not crashed on."""
    from scraper_session import harvest_paginated

    transport, _ = _transport_for({
        "https://example.com/book/novel/chapters": """
        <html><body><ul class="chapter-list">
          <li><a href="/book/novel/chapter-1"></a></li>
          <li><a href="/book/novel/chapter-2">Chapter 1: The Light</a></li>
        </ul></body></html>
        """
    })

    data = harvest_paginated(_site(), "novel", transport)
    assert data.chapter_map == [("chapter-2", "Chapter 1: The Light")]


def test_paginated_harvest_raises_on_challenge(tmp_path):
    """A Cloudflare challenge while listing fails with an actionable message
    (Option A: explicit failure, no silent fallback to the browser)."""
    from scraper_session import harvest_paginated

    transport = lambda url, headers, cookies: (
        403,
        "<html><title>Just a moment...</title>"
        "<script>window._cf_chl_opt={}</script></html>",
    )

    with pytest.raises(ChallengeError) as exc:
        harvest_paginated(_site(), "novel", transport)

    assert "browser" in str(exc.value)


def test_paginated_harvest_loop_guard(tmp_path):
    """A next link pointing at an already-seen page stops the walk."""
    from scraper_session import harvest_paginated

    pages = {
        "https://example.com/book/novel/chapters": _P1,
        "https://example.com/book/novel/chapters?page=2": _P2.replace(
            "</body></html>",
            '<a rel="next" href="https://example.com/book/novel/chapters"></a></body></html>',
        ),
    }
    transport, calls = _transport_for(pages)

    data = harvest_paginated(_site(), "novel", transport)

    assert calls == [
        "https://example.com/book/novel/chapters",
        "https://example.com/book/novel/chapters?page=2",
    ]
    assert len(data.chapter_map) == 3


def test_paginated_harvest_no_pages_error(tmp_path):
    """A site whose first list page has no chapter links raises, not silently
    returns empty (distinct from the challenge case)."""
    from scraper_session import harvest_paginated

    transport, _ = _transport_for({"https://example.com/book/novel/chapters": "<html></html>"})

    with pytest.raises(ChallengeError):
        harvest_paginated(_site(), "novel", transport)


def _numbered_site(**overrides):
    """A site with page numbers in the URL and no rel=next link."""

    return _site(
        chapter_list_page_url="https://example.com/novel/{novel_slug}/chapters?page={page}",
        **overrides,
    )


def test_paginated_harvest_walks_numbered_pages_until_no_new_links(tmp_path):
    """Without a rel=next link, the walk increments the {page} placeholder.

    Sites that silently repeat the last page for out-of-range numbers are
    stopped by the "no new slugs" rule rather than by an absent next link.
    """
    from scraper_session import harvest_paginated

    p1 = '<html><body><ul class="chapter-list">' + (
        '<li><a href="/novel/novel/chapter-1">Chapter 1: A</a></li>'
        '<li><a href="/novel/novel/chapter-2">Chapter 2: B</a></li>'
    ) + "</ul></body></html>"
    p2 = '<html><body><ul class="chapter-list">' + (
        '<li><a href="/novel/novel/chapter-3">Chapter 3: C</a></li>'
    ) + "</ul></body></html>"

    pages = {
        "https://example.com/novel/novel/chapters?page=1": p1,
        "https://example.com/novel/novel/chapters?page=2": p2,
        # out of range: the site repeats page 2 verbatim
        "https://example.com/novel/novel/chapters?page=3": p2,
    }
    transport, calls = _transport_for(pages)

    data = harvest_paginated(_numbered_site(), "novel", transport)

    assert calls == [
        "https://example.com/novel/novel/chapters?page=1",
        "https://example.com/novel/novel/chapters?page=2",
        "https://example.com/novel/novel/chapters?page=3",
    ]
    assert data.chapter_map == [
        ("chapter-1", "Chapter 1: A"),
        ("chapter-2", "Chapter 2: B"),
        ("chapter-3", "Chapter 3: C"),
    ]


def test_paginated_harvest_numbered_pages_stop_when_a_page_is_empty(tmp_path):
    """An empty page ends a numbered walk (the site has no more chapters)."""
    from scraper_session import harvest_paginated

    pages = {
        "https://example.com/novel/novel/chapters?page=1": _P1.replace(
            'href="/book/novel/', 'href="/novel/novel/'
        ).replace(
            '<a rel="next" href="https://example.com/book/novel/chapters?page=2">Next</a>',
            "",
        ),
        "https://example.com/novel/novel/chapters?page=2": "<html></html>",
    }
    transport, calls = _transport_for(pages)

    data = harvest_paginated(_numbered_site(), "novel", transport)

    assert calls == [
        "https://example.com/novel/novel/chapters?page=1",
        "https://example.com/novel/novel/chapters?page=2",
    ]
    assert len(data.chapter_map) == 2


def test_paginated_harvest_does_not_require_a_next_selector(tmp_path):
    """A numbered-page site needs no `chapter_list_next_selector` at all."""
    from scraper_session import harvest_paginated

    site = _numbered_site()
    del site["chapter_list_next_selector"]

    pages = {
        "https://example.com/novel/novel/chapters?page=1": '<html><body><ul class="chapter-list">'
        '<li><a href="/novel/novel/chapter-2">Chapter 1: A</a></li>'
        "</ul></body></html>",
    }
    transport, _ = _transport_for(pages)

    data = harvest_paginated(site, "novel", transport)
    assert data.chapter_map == [("chapter-2", "Chapter 1: A")]


def test_paginated_harvest_raises_on_rate_limit_instead_of_truncating(tmp_path):
    """A 429 part-way through the walk must NOT be mistaken for the end of the
    list: silently returning a partial map hides hundreds of chapters."""
    from scraper_session import harvest_paginated

    p1 = '<html><body><ul class="chapter-list">' + (
        '<li><a href="/novel/novel/chapter-1">Chapter 1: A</a></li>'
        '<li><a href="/novel/novel/chapter-2">Chapter 2: B</a></li>'
    ) + "</ul></body></html>"

    pages = {
        "https://example.com/novel/novel/chapters?page=1": p1,
        "https://example.com/novel/novel/chapters?page=2": (
            "<html><head><title>Access denied</title></head><body>"
            "Access denied | used Cloudflare to restrict access</body></html>"
        ),
    }

    def transport(url, headers, cookies):
        return (429, pages[url])

    with pytest.raises(ChallengeError) as exc:
        harvest_paginated(_numbered_site(), "novel", transport)

    assert "429" in str(exc.value) or "rate" in str(exc.value).lower()


def test_paginated_harvest_raises_on_cloudflare_access_denied(tmp_path):
    """Cloudflare's "Access denied" page is a block, not an empty chapter list."""
    from scraper_session import harvest_paginated

    denied = (
        "<html><head><title>Access denied</title></head><body>"
        "Access denied | used Cloudflare to restrict access</body></html>"
    )

    transport, _ = _transport_for(
        {"https://example.com/novel/novel/chapters?page=1": denied}
    )

    with pytest.raises(ChallengeError):
        harvest_paginated(_numbered_site(), "novel", transport)


def test_make_session_paginated_mode(tmp_path, monkeypatch):
    """make_session wires the paginated walker in as the FastSession harvest."""
    import scraper_session
    import scraper_fast

    captured = {}

    def fake_harvest_paginated(site, novel_slug, transport):
        captured["novel_slug"] = novel_slug
        captured["site"] = site
        return scraper_session.SessionData("", {}, [("chapter-2", "Chapter 1: X")])

    monkeypatch.setattr(scraper_fast, "harvest_paginated", fake_harvest_paginated)

    s = make_session(site=_site(), novel_slug="novel", cache_path=tmp_path / "c.json")

    map_ = s.chapter_map()  # triggers the lazy harvest

    assert captured["novel_slug"] == "novel"
    assert captured["site"]["name"] == "Example Site"
    assert map_ == {1: ChapterRef(1, "chapter-2", "Chapter 1: X")}


def test_make_session_browser_mode_still_used_when_configured(tmp_path, monkeypatch):
    """Sites with `harvest: browser` keep the browser harvest untouched."""
    import scraper_session
    import scraper_fast

    calls = {"n": 0}

    def fake_browser_harvest(site, novel_slug, chrome_major=None, driver_factory=None):
        calls["n"] += 1
        return scraper_session.SessionData("UA", {"cf_clearance": "v"}, [("chapter-2", "Chapter 1: X")])

    monkeypatch.setattr(scraper_fast, "harvest_session", fake_browser_harvest)

    s = make_session(
        site=_site(harvest="browser"), novel_slug="novel", cache_path=tmp_path / "c.json"
    )
    assert s.chapter_map() == {1: ChapterRef(1, "chapter-2", "Chapter 1: X")}
    assert calls["n"] == 1


def test_make_session_browser_mode_is_default(tmp_path, monkeypatch):
    """An absent `harvest` key means browser mode (back-compat with older configs)."""
    import scraper_session
    import scraper_fast

    calls = {"n": 0}

    def fake_browser_harvest(site, novel_slug, chrome_major=None, driver_factory=None):
        calls["n"] += 1
        return scraper_session.SessionData("UA", {}, [("chapter-2", "Chapter 1: X")])

    monkeypatch.setattr(scraper_fast, "harvest_session", fake_browser_harvest)

    site = _site()
    del site["harvest"]

    s = make_session(site=site, novel_slug="novel", cache_path=tmp_path / "c.json")
    assert s.chapter_map() == {1: ChapterRef(1, "chapter-2", "Chapter 1: X")}
