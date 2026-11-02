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


# -- a selection that is not empty, but names nothing ----------------------


def test_the_word_chapter_alone_does_not_mean_everything():
    # It folds to "", which used to mean "all". A half-typed label must not
    # download the entire series.
    manifest = Manifest("T", "s", [ChapterRecord(1, "a", "Chapter 1", 1.0)])
    with pytest.raises(ValueError, match="names no chapter"):
        select_plan(manifest, "chapter")


def test_chapter_all_does_not_mean_everything():
    manifest = Manifest("T", "s", [ChapterRecord(1, "a", "Chapter 1", 1.0)])
    with pytest.raises(ValueError, match="only on its own"):
        select_plan(manifest, "Chapter all")


def test_the_omitted_selection_still_means_everything():
    # The distinction the fix rests on: an omitted argument is "everything", a
    # mistyped one is an error.
    manifest = Manifest("T", "s", [ChapterRecord(1, "a", "Chapter 1", 1.0)])
    assert select_plan(manifest, None) == manifest.chapters
    assert select_plan(manifest, "") == manifest.chapters


# -- a label need not be a number ------------------------------------------


def test_a_non_numeric_label_is_selectable_by_its_exact_text():
    # The label is present in the manifest, so refusing it because it will not
    # parse as a float would make label-first matching a lie.
    manifest = Manifest(
        "T",
        "s",
        [
            ChapterRecord(1, "chapter-175-5a", "Chapter 175.5a", 175.5),
            ChapterRecord(2, "chapter-176", "Chapter 176", 176.0),
        ],
    )
    assert [r.ordinal for r in select_plan(manifest, "175.5a")] == [1]


def test_a_site_slash_label_is_selectable_by_its_exact_text():
    # ``comic_manifest.parse_label_number`` supports "Chapter 280 - 237" and
    # tests it, so a selection must be able to name it too.
    manifest = Manifest(
        "T", "s", [ChapterRecord(1, "chapter-280-237", "Chapter 280 - 237", 280.0)]
    )
    assert [r.ordinal for r in select_plan(manifest, "Chapter 280 - 237")] == [1]


# -- a range must not silently omit a chapter the site lists inside it ------


def test_a_range_that_omits_a_retroactively_inserted_chapter_warns(capsys):
    # The site inserted Chapter 5.1 between 5 and 6 after the manifest was
    # built, so ``assign`` appended it at max + 1: ordinal 7, after chapter 6.
    # ``-c 1-6`` returns 1..6 by position and would drop 5.1 without a word.
    chapters = [
        ChapterRecord(i, f"c{i}", f"Chapter {i}", float(i)) for i in range(1, 7)
    ]
    chapters.append(ChapterRecord(7, "c5-1", "Chapter 5.1", 5.1))
    manifest = Manifest("T", "s", chapters)

    plan = select_plan(manifest, "1-6")

    assert [r.ordinal for r in plan] == [1, 2, 3, 4, 5, 6]
    err = capsys.readouterr().err
    assert "warning" in err
    assert "Chapter 5.1" in err


def test_a_contiguous_range_says_nothing(capsys):
    manifest = Manifest(
        "T", "s", [ChapterRecord(i, f"c{i}", f"Chapter {i}", float(i)) for i in range(1, 7)]
    )
    assert [r.ordinal for r in select_plan(manifest, "1-6")] == [1, 2, 3, 4, 5, 6]
    assert capsys.readouterr().err == ""


def test_the_strictly_earlier_fractional_neighbour_is_not_warned_about(capsys):
    # This site lists 175.1 *before* 175. A range 175-176 therefore omits 175.1
    # numerically, but it was pinned before the range and is genuinely outside
    # it -- warning here would be noise on every such run.
    manifest = Manifest(
        "T",
        "s",
        [
            ChapterRecord(1, "c175-1", "Chapter 175.1", 175.1),
            ChapterRecord(2, "c175", "Chapter 175", 175.0),
            ChapterRecord(3, "c176", "Chapter 176", 176.0),
        ],
    )
    assert [r.ordinal for r in select_plan(manifest, "175-176")] == [2, 3]
    assert capsys.readouterr().err == ""
