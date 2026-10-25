from pathlib import Path

from scraper_store import chapter_filename, existing_chapter, write_chapter


def test_filename_is_unpadded():
    assert chapter_filename(7) == "chapter_7.txt"


def test_existing_chapter_matches_both_forms(tmp_path):
    d = tmp_path / "novel"
    d.mkdir()
    assert existing_chapter(tmp_path, "novel", 7) is None
    (d / "chapter_0007.txt").write_text("legacy", encoding="utf-8")
    assert existing_chapter(tmp_path, "novel", 7) is not None


def test_write_then_skip_without_overwrite(tmp_path):
    path = write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "body")
    assert path is not None
    assert Path(path).read_text(encoding="utf-8").startswith("Chapter 7: X")
    assert write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "new") is None


def test_skip_when_legacy_padded_file_exists(tmp_path):
    d = tmp_path / "novel"
    d.mkdir()
    (d / "chapter_0007.txt").write_text("old", encoding="utf-8")
    assert write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "new") is None


def test_overwrite_replaces_content(tmp_path):
    write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "old")
    write_chapter(tmp_path, "novel", 7, "Chapter 7: X", "new", overwrite=True)
    assert (tmp_path / "novel" / "chapter_7.txt").read_text(encoding="utf-8").endswith(
        "new"
    )
