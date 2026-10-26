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


from scraper_parse import extract_chapter, looks_like_chapter

SAMPLE = """
<html><body>
  <div class="top"><span class="chapter">Chapter 4: My Life Now</span></div>
  <div class="txt"><p>Line one.</p><p>Line two.</p></div>
</body></html>
"""


def test_extract_returns_title_and_paragraph_text():
    title, text = extract_chapter(SAMPLE, "div.txt", "div.top span.chapter")
    assert title == "Chapter 4: My Life Now"
    assert "Line one." in text and "Line two." in text


def test_extract_missing_nodes_returns_empty():
    assert extract_chapter("<html></html>", "div.txt", "span.chapter") == ("", "")


def test_plausibility_rejects_short_or_challenge_text():
    assert looks_like_chapter("x" * 500) is True
    assert looks_like_chapter("Just a moment...") is False
    assert looks_like_chapter("") is False


NESTED = (
    '<div class="txt">'
    "<div><script>ad('" + ("x" * 500) + "')</script></div>"
    "<p><p>First para.</p><p>Second para.</p></p>"
    "</div>"
)


def test_extract_does_not_double_count_p_nested_in_p():
    # html.parser nests these <p>s; a browser (and the site's page_source) would
    # have auto-closed the outer <p>, so we must not emit the paragraphs twice.
    _, text = extract_chapter(NESTED, "div.txt", "span.chapter")
    assert text == "First para.\nSecond para."


def test_extract_excludes_scripts_and_ad_divs():
    _, text = extract_chapter(NESTED, "div.txt", "span.chapter")
    assert "x" * 50 not in text


INLINE = (
    '<div class="top"><span class="chapter">Chapter 9: Inline</span></div>'
    "<div class=\"txt\"><p>Hello <b>bolded</b> world.</p></div>"
)


def test_extract_matches_legacy_for_inline_markup():
    # Legacy uses p.get_text().strip(); get_text(strip=True) would glue words to
    # adjacent inline tags ("Helloboldedworld."), diverging from the corpus.
    _, text = extract_chapter(INLINE, "div.txt", "div.top span.chapter")
    assert text == "Hello bolded world."


