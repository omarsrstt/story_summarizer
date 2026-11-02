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
