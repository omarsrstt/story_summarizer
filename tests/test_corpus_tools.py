"""Tests for the read-only-by-default corpus maintenance tools."""

from corpus_tools import plan_padding_rename, strip_recurring_trailers

import corpus_tools


def _chapter(directory, name, *lines, encoding="utf-8"):
    path = directory / name
    path.write_text("\n".join(lines) + "\n", encoding=encoding)
    return path


# --- padding rename -------------------------------------------------------


def test_plan_rename_targets_five_digit_names(tmp_path):
    novel = tmp_path / "novel"
    novel.mkdir()
    _chapter(novel, "chapter_1.txt", "Chapter 1: A", "body")
    _chapter(novel, "chapter_0513.txt", "Chapter 513: B", "body")

    moves = plan_padding_rename(novel)

    assert {src.name: dst.name for src, dst in moves} == {
        "chapter_1.txt": "chapter_00001.txt",
        "chapter_0513.txt": "chapter_00513.txt",
    }


def test_plan_rename_skips_files_already_padded(tmp_path):
    novel = tmp_path / "novel"
    novel.mkdir()
    _chapter(novel, "chapter_00001.txt", "Chapter 1: A", "body")

    assert plan_padding_rename(novel) == []


def test_plan_rename_ignores_non_chapter_files(tmp_path):
    novel = tmp_path / "novel"
    novel.mkdir()
    _chapter(novel, "notes.txt", "not a chapter")
    _chapter(novel, "chapter_1.txt", "Chapter 1: A", "body")

    moves = plan_padding_rename(novel)
    assert [src.name for src, _ in moves] == ["chapter_1.txt"]


# --- recurring-trailer detection -----------------------------------------


def test_strip_removes_a_trailer_that_repeats_across_chapters(tmp_path):
    """The ad line is learned from the data, not hardcoded, so it works on any
    site without naming a single string."""
    novel = tmp_path / "novel"
    novel.mkdir()
    for n in range(1, 11):
        _chapter(
            novel,
            f"chapter_{n}.txt",
            f"Chapter {n}: Title",
            f"Prose for chapter {n}.",
            "Read more stories at example-dot-com!",
        )

    report = strip_recurring_trailers(novel)

    assert report.lines_removed == 10
    for n in range(1, 11):
        text = (novel / f"chapter_{n}.txt").read_text(encoding="utf-8")
        assert "example-dot-com" not in text
        assert f"Prose for chapter {n}." in text


def test_strip_leaves_unique_final_lines_alone(tmp_path):
    """A distinctive last line is prose, not boilerplate, and must survive."""
    novel = tmp_path / "novel"
    novel.mkdir()
    _chapter(novel, "chapter_1.txt", "Chapter 1: A", "The end was quiet.")
    _chapter(novel, "chapter_2.txt", "Chapter 2: B", "She turned away.")

    report = strip_recurring_trailers(novel)

    assert report.files_changed == 0
    assert (novel / "chapter_1.txt").read_text(encoding="utf-8").strip().endswith(
        "The end was quiet."
    )


def test_strip_handles_a_two_line_trailer(tmp_path):
    """Some chapters carry the ad as two separate lines; both are removed."""
    novel = tmp_path / "novel"
    novel.mkdir()
    for n in range(1, 11):
        _chapter(
            novel,
            f"chapter_{n}.txt",
            f"Chapter {n}: Title",
            f"Prose for chapter {n}.",
            "Read more at example-dot-com.",
            "Thank you so much!",
        )

    strip_recurring_trailers(novel)

    for n in range(1, 11):
        lines = (novel / f"chapter_{n}.txt").read_text(encoding="utf-8").splitlines()
        assert lines[-1] == f"Prose for chapter {n}."


def test_strip_does_not_touch_a_matching_line_mid_chapter(tmp_path):
    """Dialogue can quote the ad phrase; only trailing lines may be removed."""
    novel = tmp_path / "novel"
    novel.mkdir()
    for n in range(1, 11):
        _chapter(
            novel,
            f"chapter_{n}.txt",
            f"Chapter {n}: Title",
            "Thank you so much!",
            f"Prose for chapter {n}.",
            "Read more at example-dot-com.",
        )

    strip_recurring_trailers(novel)

    lines = (novel / "chapter_1.txt").read_text(encoding="utf-8").splitlines()
    assert "Thank you so much!" in lines  # the mid-chapter quote survives
    assert lines[-1] == "Prose for chapter 1."


def test_strip_dry_run_reports_without_writing(tmp_path):
    novel = tmp_path / "novel"
    novel.mkdir()
    for n in range(1, 11):
        _chapter(
            novel,
            f"chapter_{n}.txt",
            f"Chapter {n}: Title",
            f"Prose for chapter {n}.",
            "Read more at example-dot-com!",
        )

    report = strip_recurring_trailers(novel, apply=False)

    assert report.lines_removed == 10
    assert "example-dot-com" in (novel / "chapter_1.txt").read_text(encoding="utf-8")


def test_strip_never_empties_a_chapter(tmp_path):
    """If a whole file looks like boilerplate, keep it rather than write nothing."""
    novel = tmp_path / "novel"
    novel.mkdir()
    for n in range(1, 11):
        _chapter(novel, f"chapter_{n}.txt", "Read more at example-dot-com!")

    strip_recurring_trailers(novel)

    text = (novel / "chapter_1.txt").read_text(encoding="utf-8")
    assert text.strip() == "Read more at example-dot-com!"


def test_strip_needs_a_minimum_number_of_chapters(tmp_path):
    """One or two chapters give no statistical signal; refuse to guess."""
    novel = tmp_path / "novel"
    novel.mkdir()
    _chapter(novel, "chapter_1.txt", "Chapter 1: A", "Prose.", "Read more at x!")
    _chapter(novel, "chapter_2.txt", "Chapter 2: B", "Prose.", "Read more at x!")

    report = strip_recurring_trailers(novel)

    assert report.files_changed == 0

# --- command line ---------------------------------------------------------


def test_cli_all_apply_writes_a_single_backup(tmp_path):
    """`all --apply` must not leave two backup copies, one per sub-step."""
    novel = tmp_path / "novel"
    novel.mkdir()
    for n in range(1, 11):
        _chapter(
            novel,
            f"chapter_{n}.txt",
            f"Chapter {n}: Title",
            f"Prose for chapter {n}.",
            "Read more at example-dot-com!",
        )

    assert corpus_tools.main(["all", str(novel), "--apply"]) == 0

    assert (novel / "chapter_00001.txt").exists()
    backups = sorted(p.name for p in tmp_path.iterdir() if ".bak" in p.name)
    assert backups == ["novel.bak"]


def test_cli_is_a_dry_run_without_apply(tmp_path):
    novel = tmp_path / "novel"
    novel.mkdir()
    for n in range(1, 11):
        _chapter(
            novel,
            f"chapter_{n}.txt",
            f"Chapter {n}: Title",
            f"Prose for chapter {n}.",
            "Read more at example-dot-com!",
        )

    assert corpus_tools.main(["all", str(novel)]) == 0

    assert (novel / "chapter_1.txt").exists()
    assert not (novel / "chapter_00001.txt").exists()
    assert "example-dot-com" in (novel / "chapter_1.txt").read_text(encoding="utf-8")
    assert not any(".bak" in p.name for p in tmp_path.iterdir())


def test_cli_rejects_a_missing_directory(tmp_path):
    assert corpus_tools.main(["all", str(tmp_path / "nope")]) == 2

def test_strip_preserves_crlf_line_endings(tmp_path):
    """A corpus scraped on Windows uses CRLF; rewriting must not silently
    convert it to LF (or vice versa) just because of the host platform."""
    novel = tmp_path / "novel"
    novel.mkdir()
    for n in range(1, 11):
        raw = (
            f"Chapter {n}: Title\r\nProse for chapter {n}.\r\n"
            "Read more at example-dot-com!\r\n"
        )
        (novel / f"chapter_{n}.txt").write_bytes(raw.encode("utf-8"))

    strip_recurring_trailers(novel)

    data = (novel / "chapter_1.txt").read_bytes()
    assert data == "Chapter 1: Title\r\nProse for chapter 1.\r\n".encode("utf-8")


def test_strip_preserves_lf_line_endings(tmp_path):
    novel = tmp_path / "novel"
    novel.mkdir()
    for n in range(1, 11):
        raw = (
            f"Chapter {n}: Title\nProse for chapter {n}.\n"
            "Read more at example-dot-com!\n"
        )
        (novel / f"chapter_{n}.txt").write_bytes(raw.encode("utf-8"))

    strip_recurring_trailers(novel)

    data = (novel / "chapter_1.txt").read_bytes()
    assert data == "Chapter 1: Title\nProse for chapter 1.\n".encode("utf-8")
