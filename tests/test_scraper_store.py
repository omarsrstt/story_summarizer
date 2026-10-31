from pathlib import Path

from scraper_store import chapter_filename, existing_chapter, write_chapter


def test_filename_uses_five_digit_padding():
    assert chapter_filename(7) == "chapter_00007.txt"
    assert chapter_filename(1) == "chapter_00001.txt"
    assert chapter_filename(530) == "chapter_00530.txt"


def test_existing_chapter_matches_all_forms(tmp_path):
    d = tmp_path / "novel"
    d.mkdir()
    assert existing_chapter(tmp_path, "novel", 7) is None

    # Legacy four-digit padding and the original unpadded form are still
    # recognised so an existing chapter is never re-scraped as a duplicate.
    (d / "chapter_0007.txt").write_text("legacy", encoding="utf-8")
    assert existing_chapter(tmp_path, "novel", 7) is not None
    (d / "chapter_0007.txt").unlink()
    (d / "chapter_7.txt").write_text("older", encoding="utf-8")
    assert existing_chapter(tmp_path, "novel", 7) is not None


def test_write_then_skip_without_overwrite(tmp_path):
    path = write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "body")
    assert path is not None
    assert Path(path).read_text(encoding="utf-8").startswith("Chapter 7: X")
    assert write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "new") is None


def test_skip_when_deprecated_unpadded_file_exists(tmp_path):
    d = tmp_path / "novel"
    d.mkdir()
    (d / "chapter_7.txt").write_text("old", encoding="utf-8")
    assert write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "new") is None


def test_skip_when_deprecated_four_digit_file_exists(tmp_path):
    d = tmp_path / "novel"
    d.mkdir()
    (d / "chapter_0007.txt").write_text("old", encoding="utf-8")
    assert write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "new") is None


def test_overwrite_replaces_content(tmp_path):
    write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "old")
    write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "new", overwrite=True)
    assert (tmp_path / "novel" / "chapter_00007.txt").read_text(
        encoding="utf-8"
    ).endswith("new")


def test_overwrite_deprecated_file_in_place(tmp_path):
    # Overwriting a chapter that exists under an old name must not leave a
    # second copy under the new five-digit name.
    d = tmp_path / "novel"
    d.mkdir()
    (d / "chapter_0007.txt").write_text("old", encoding="utf-8")
    path = write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "new", overwrite=True)
    assert Path(path).name == "chapter_0007.txt"
    assert (d / "chapter_0007.txt").read_text(encoding="utf-8").endswith("new")
    assert not (d / "chapter_00007.txt").exists()
