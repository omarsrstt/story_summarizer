"""Tests for comic page slicing, bubble filtering and transcript rendering."""

from PIL import Image

from comic_pages import (
    dedupe_seam_blocks,
    format_transcript,
    is_bubble_like,
    reading_order,
    render_block_text,
    slice_scroll,
)


def _blank(width=100, height=2000, colour="white"):
    return Image.new("RGB", (width, height), colour)


# -- slicing a continuous scroll -------------------------------------------


def test_a_short_image_is_one_tile_at_offset_zero():
    image = _blank(height=400)
    tiles = slice_scroll(image, tile_height=1200)
    assert len(tiles) == 1
    assert tiles[0][0] == 0


def test_a_tall_scroll_is_sliced_into_overlapping_tiles():
    tiles = slice_scroll(_blank(height=5000), tile_height=1200, overlap=120)
    assert len(tiles) > 1
    for _, tile in tiles:
        assert tile.size[1] <= 1200


def test_tiles_cover_the_whole_scroll():
    tiles = slice_scroll(_blank(height=5000), tile_height=1200, overlap=120)
    top, last = tiles[-1]
    assert top + last.size[1] >= 5000


def test_tile_offsets_are_the_absolute_pixel_row():
    tiles = slice_scroll(_blank(height=5000), tile_height=1200, overlap=120)
    assert [top for top, _ in tiles] == [0, 1080, 2160, 3240, 4320]


def test_no_empty_trailing_tile_is_produced():
    tiles = slice_scroll(_blank(height=1300), tile_height=1200, overlap=120)
    assert all(tile.size[1] > 0 for _, tile in tiles)


# -- duplicate detection across a seam -------------------------------------


def test_the_same_text_twice_at_one_place_is_one_block():
    # A bubble on a seam is read by both tiles, a tile-height apart in the
    # crop but at the same absolute row.
    blocks = [
        {"page": 1, "box": (10, 900, 100, 940), "text": "hello"},
        {"page": 1, "box": (10, 902, 100, 942), "text": "hello"},
    ]
    assert len(dedupe_seam_blocks(blocks)) == 1


def test_the_same_words_scroll_twice_at_different_places_are_kept():
    # "NO!" repeated down a page is content, not a seam artefact.
    blocks = [
        {"page": 1, "box": (10, 100, 100, 140), "text": "NO!"},
        {"page": 1, "box": (10, 1500, 100, 1540), "text": "NO!"},
    ]
    assert len(dedupe_seam_blocks(blocks)) == 2


def test_the_same_text_on_different_pages_is_kept():
    blocks = [
        {"page": 1, "box": (10, 900, 100, 940), "text": "END"},
        {"page": 2, "box": (10, 900, 100, 940), "text": "END"},
    ]
    assert len(dedupe_seam_blocks(blocks)) == 2


# -- bubble detection ------------------------------------------------------


def test_pale_region_full_of_dark_lettering_is_bubble_like():
    image = Image.new("RGB", (200, 120), "white")
    for y in range(30, 90, 10):
        for x in range(20, 180):
            image.putpixel((x, y), (0, 0, 0))
    assert is_bubble_like(image, (0, 0, 200, 120))


def test_busy_dark_region_is_not_bubble_like():
    image = Image.new("RGB", (200, 120), (30, 30, 40))
    assert not is_bubble_like(image, (0, 0, 200, 120))


def test_tiny_region_is_not_bubble_like():
    image = Image.new("RGB", (200, 120), "white")
    assert not is_bubble_like(image, (0, 0, 20, 10))


# -- ordering and rendering ------------------------------------------------


def test_blocks_are_ordered_top_to_bottom():
    blocks = [
        {"box": (10, 500, 100, 550), "text": "third", "page": 2},
        {"box": (10, 50, 100, 90), "text": "first", "page": 1},
        {"box": (10, 300, 100, 340), "text": "second", "page": 1},
    ]
    assert [b["text"] for b in reading_order(blocks)] == ["first", "second", "third"]


def test_blocks_on_the_same_line_go_left_to_right():
    # Same line: their vertical extents overlap, so x decides.
    blocks = [
        {"box": (200, 50, 300, 95), "text": "right", "page": 1},
        {"box": (10, 52, 120, 92), "text": "left", "page": 1},
    ]
    assert [b["text"] for b in reading_order(blocks)] == ["left", "right"]


def test_ordering_never_returns_to_an_earlier_page():
    blocks = [
        {"box": (10, 50, 100, 90), "text": "page two", "page": 2},
        {"box": (10, 900, 100, 940), "text": "page one", "page": 1},
    ]
    assert [b["text"] for b in reading_order(blocks)] == ["page one", "page two"]


def test_render_collapses_internal_whitespace():
    assert render_block_text("  he   doesn't\nhit ") == "he doesn't hit"


def test_transcript_starts_then_marks_each_page_once():
    blocks = [
        {"box": (10, 50, 100, 90), "text": "first line", "page": 1},
        {"box": (10, 200, 100, 240), "text": "second line", "page": 1},
        {"box": (10, 50, 100, 90), "text": "third line", "page": 2},
    ]
    out = format_transcript("Chapter 1", blocks)
    assert out.startswith("# Chapter 1")
    assert out.count("--- page 1 ---") == 1
    assert out.count("--- page 2 ---") == 1
    assert out.index("first line") < out.index("second line") < out.index("third line")


def test_transcript_skips_empty_blocks():
    blocks = [
        {"box": (10, 50, 100, 90), "text": "real", "page": 1},
        {"box": (10, 200, 100, 240), "text": "   ", "page": 1},
    ]
    out = format_transcript("Chapter 1", blocks)
    assert "real" in out
    assert out.count("\n") <= 4
