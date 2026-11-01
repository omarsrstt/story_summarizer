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
