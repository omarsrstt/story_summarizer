from scraper_parse import ChapterRef, build_chapter_map, parse_chapter_number


def test_parse_number_from_titled_chapter():
    assert parse_chapter_number("Chapter 1: The Light At The End of The Tunnel") == 1


def test_parse_number_from_bare_chapter():
    assert parse_chapter_number("Chapter 98") == 98


def test_parse_number_ignores_prologue_and_junk():
    assert parse_chapter_number("c-1: Prologue: Start") is None
    assert parse_chapter_number("") is None


def test_build_map_uses_title_number_and_keeps_slug():
    options = [
        ("chapter-1", "c-1: Prologue: Start"),
        ("chapter-2", "Chapter 1: The Light At The End of The Tunnel"),
        ("chapter-530-end-story", "Chapter 530: End Story"),
    ]
    result = build_chapter_map(options)
    assert set(result) == {1, 530}
    assert result[1] == ChapterRef(
        1, "chapter-2", "Chapter 1: The Light At The End of The Tunnel"
    )
    assert result[530].slug == "chapter-530-end-story"
