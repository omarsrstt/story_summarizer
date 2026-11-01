"""Tests for the comic -> chapter-text driver's filename grouping and slug."""

from pathlib import Path

import pytest

from comic_reader import comic_slug, group_chapters, parse_only


# -- keeping comics off the novels' namespaces ----------------------------


def test_a_series_becomes_a_prefixed_slug():
    assert comic_slug("TBATE") == "comic-TBATE"
    assert comic_slug("  TBATE  ") == "comic-TBATE"


def test_prefixing_is_idempotent():
    # Re-running with the slug the script itself printed must not stack prefixes.
    assert comic_slug("comic-TBATE") == "comic-TBATE"
    assert comic_slug("COMIC-TBATE") == "comic-TBATE"


def test_the_slug_can_never_collide_with_a_novel():
    # A comic called "Tensei Slime" must not land on novels/tensei-slime, or
    # it would overwrite a novel of that name -- the whole point of the prefix.
    assert comic_slug("the-beginning-after-the-end") == (
        "comic-the-beginning-after-the-end"
    )


def test_a_slug_that_would_escape_the_output_tree_is_rejected():
    for bad in ("..", "../etc", "a/b", "a\\b", ""):
        with pytest.raises(ValueError):
            comic_slug(bad)


def _touch(directory: Path, name: str) -> None:
    (directory / name).write_bytes(b"x")


def test_paginated_pages_are_grouped_and_ordered(tmp_path):
    for name in ("c_1_10.jpg", "c_1_2.jpg", "c_1_1.jpg"):
        _touch(tmp_path, name)
    chapters = group_chapters(tmp_path)
    assert list(chapters) == [1]
    assert [p.name for p in chapters[1]] == ["c_1_1.jpg", "c_1_2.jpg", "c_1_10.jpg"]


def test_a_single_scroll_is_its_own_chapter(tmp_path):
    _touch(tmp_path, "c_0_1.jpg")
    chapters = group_chapters(tmp_path)
    assert list(chapters) == [0]
    assert [p.name for p in chapters[0]] == ["c_0_1.jpg"]


def test_both_layouts_in_one_directory(tmp_path):
    _touch(tmp_path, "c_0_1.jpg")
    _touch(tmp_path, "c_1_1.jpg")
    _touch(tmp_path, "c_1_2.jpg")
    chapters = group_chapters(tmp_path)
    assert list(chapters) == [0, 1]
    assert len(chapters[0]) == 1
    assert len(chapters[1]) == 2


def test_unrelated_files_are_ignored(tmp_path):
    _touch(tmp_path, "c_1_1.jpg")
    _touch(tmp_path, "cover.jpg")
    _touch(tmp_path, "notes.txt")
    chapters = group_chapters(tmp_path)
    assert list(chapters) == [1]
    assert len(chapters[1]) == 1


def test_an_empty_directory_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        group_chapters(tmp_path)


# -- selection -------------------------------------------------------------


def test_only_all_selects_everything():
    assert parse_only("all", [0, 1, 2]) == [0, 1, 2]
    assert parse_only(None, [0, 1, 2]) == [0, 1, 2]


def test_only_single_and_range_and_list():
    assert parse_only("1", [0, 1, 2]) == [1]
    assert parse_only("1-2", [0, 1, 2]) == [1, 2]
    assert parse_only("0,2", [0, 1, 2]) == [0, 2]


def test_only_ignores_numbers_not_present():
    assert parse_only("5", [0, 1]) == []
