from scraper_session import SessionCache, SessionData


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
