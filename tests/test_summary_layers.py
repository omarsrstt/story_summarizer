"""Tests for the wider-artifact layer: batching, grouping, and digest shape.

These are the decisions that turn 516 per-chapter recaps into a 5-chapter
summary, a skim digest, or a whole-novel summary. They are pure arithmetic over
chapter numbers, so they are cheap to pin down -- and worth pinning down,
because an off-by-one here silently drops chapters from every wider artifact
built on top of the recaps.
"""

import pytest

from summary_layers import (
    MergeError,
    Piece,
    batch_label,
    group_pieces,
    pack_groups,
    partial_outputs,
    rounds_needed,
    summary_filename,
)


def pieces(*numbers):
    return [Piece(n, n, f"recap {n}") for n in numbers]


# --------------------------------------------------------------------------
# labels and filenames
# --------------------------------------------------------------------------

def test_batch_label_singular_for_one_chapter():
    assert batch_label(7, 7) == "Chapter 7"


def test_batch_label_plural_for_a_range():
    assert batch_label(1, 5) == "Chapters 1-5"


def test_summary_filename_pads_both_ends():
    assert summary_filename(1, 5) == "summary_chapters_00001-00005.txt"


def test_summary_filename_sorts_lexicographically_by_chapter():
    names = [summary_filename(n, n + 4) for n in (1, 10, 100)]
    assert sorted(names) == names


# --------------------------------------------------------------------------
# fixed-width batching
# --------------------------------------------------------------------------

def test_partial_outputs_covers_everything_when_it_divides_evenly():
    ranges = partial_outputs(pieces(*range(1, 11)), unit=5)
    assert ranges == [(1, 5), (6, 10)]


def test_partial_outputs_keeps_a_short_final_batch():
    ranges = partial_outputs(pieces(*range(1, 13)), unit=5)
    assert ranges == [(1, 5), (6, 10), (11, 12)]


def test_partial_outputs_keeps_a_lone_final_chapter():
    ranges = partial_outputs(pieces(*range(1, 7)), unit=5)
    assert ranges == [(1, 5), (6, 6)]


def test_partial_outputs_rejects_a_zero_width():
    with pytest.raises(MergeError):
        partial_outputs(pieces(1, 2), unit=0)


# --------------------------------------------------------------------------
# grouping for digest batches and merge rounds
# --------------------------------------------------------------------------

def test_group_pieces_chops_at_the_requested_size():
    assert [len(g) for g in group_pieces(pieces(*range(1, 6)), 2)] == [2, 2, 1]


def test_group_pieces_rejects_a_zero_width():
    with pytest.raises(MergeError):
        group_pieces(pieces(1), 0)


def test_pack_groups_respects_fan_in_with_a_roomy_budget():
    groups = pack_groups(pieces(*range(1, 8)), fan_in=3, fits=lambda _: True)
    assert [len(g) for g in groups] == [3, 3, 1]


def test_pack_groups_splits_earlier_when_the_text_does_not_fit():
    # Refuse anything beyond three pieces, then beyond two.
    fits = lambda group: len(group) <= (3 if len(group) <= 3 else 2)
    groups = pack_groups(pieces(*range(1, 7)), fan_in=10, fits=fits)
    assert [len(g) for g in groups] == [3, 3]


def test_pack_groups_keeps_order():
    groups = pack_groups(pieces(*range(1, 8)), fan_in=3, fits=lambda _: True)
    flattened = [p.start for group in groups for p in group]
    assert flattened == list(range(1, 8))


def test_pack_groups_never_drops_a_piece_even_alone():
    groups = pack_groups(pieces(1, 2, 3), fan_in=5, fits=lambda group: len(group) == 1)
    assert [[p.start for p in g] for g in groups] == [[1], [2], [3]]


# --------------------------------------------------------------------------
# round estimation
# --------------------------------------------------------------------------

def test_rounds_needed_is_zero_for_a_single_piece():
    assert rounds_needed(1, 20) == 0


def test_rounds_needed_collapses_in_one_round_when_fan_in_covers_all():
    assert rounds_needed(5, 20) == 1


def test_rounds_needed_counts_the_tree_height():
    # 516 -> 26 -> 2 -> 1
    assert rounds_needed(516, 20) == 3


def test_rounds_needed_for_two_pieces_with_narrow_fan_in():
    assert rounds_needed(2, 2) == 1


def test_rounds_needed_rejects_a_fan_in_that_cannot_reduce():
    with pytest.raises(MergeError):
        rounds_needed(10, 1)


# --------------------------------------------------------------------------
# digest file shape
# --------------------------------------------------------------------------

def test_render_digest_is_ordered_by_chapter_not_insertion():
    from summarize import render_digest

    text = render_digest({10: "Tenth.", 2: "Second."})
    assert text.index("Chapter 2:") < text.index("Chapter 10:")
    assert text.endswith("\n")


def test_digest_line_parses_the_plain_form():
    from summarize import parse_digest_line

    assert parse_digest_line("Chapter 7: The gate opens.") == (7, "The gate opens.")


def test_digest_line_parses_markdown_emphasis_and_dashes():
    from summarize import parse_digest_line

    assert parse_digest_line("**Chapter 12:** A dragon hatches.") == (
        12, "A dragon hatches.")
    assert parse_digest_line("Chapter 3 - They flee.") == (3, "They flee.")


def test_digest_line_ignores_text_without_a_chapter_number():
    from summarize import parse_digest_line

    assert parse_digest_line("Here is a digest of the chapter.") is None


def test_digest_line_ignores_an_empty_body():
    from summarize import parse_digest_line

    assert parse_digest_line("Chapter 4:") is None


# --------------------------------------------------------------------------
# master file shape
# --------------------------------------------------------------------------

def test_render_master_is_clean_once_it_collapses_to_one_piece():
    from summarize import render_master

    text = render_master([Piece(1, 516, "The whole story.")])
    assert text.startswith("# Novel summary")
    assert "Partial" not in text
    assert "The whole story." in text


def test_render_master_marks_a_partial_tree():
    from summarize import render_master

    text = render_master([Piece(1, 100, "First half."), Piece(101, 200, "Second half.")])
    assert "Partial" in text
    assert "First half." in text and "Second half." in text
