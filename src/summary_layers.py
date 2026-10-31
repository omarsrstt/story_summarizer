"""Geometry for merging per-chapter recaps into wider summaries.

A "batch summary" of n chapters and a whole-novel summary are the same
operation at different scales: take the recaps of consecutive chapters and
merge them into one piece of prose. The only real difference is whether the
input fits in the model's context window.

So this module holds only the arithmetic -- which chapters form a group, what
the group is called, what the output file is called. Measuring how much text
actually fits requires a tokenizer, so that lives in ``summarize.py``.

Merging reads recaps, never source chapters, which is what lets a batch
summary be regenerated at a different width without re-reading the novel.
"""

from __future__ import annotations

from dataclasses import dataclass


class MergeError(RuntimeError):
    """The requested width cannot be produced."""


@dataclass
class Piece:
    """One unit of text to merge: a chapter recap, or a previous round's output."""

    #: Inclusive chapter range this text covers, e.g. ``start=1, end=5``.
    start: int
    end: int
    text: str

    @property
    def span(self) -> str:
        return f"{self.start}-{self.end}"


def batch_label(start: int, end: int) -> str:
    """Human-facing label for a chapter range."""

    return f"Chapter {start}" if start == end else f"Chapters {start}-{end}"


def summary_filename(start: int, end: int) -> str:
    """Filename for a batch summary covering ``start``..``end`` inclusive."""

    return f"summary_chapters_{start:05d}-{end:05d}.txt"


def group_pieces(pieces: list, size: int) -> list[list]:
    """Chop an ordered list of pieces into groups of at most ``size``.

    The final group may be short: a range that does not divide evenly must
    still cover every chapter rather than dropping the remainder.
    """

    if size < 1:
        raise MergeError(f"Group size must be at least 1, got {size}")
    return [pieces[i:i + size] for i in range(0, len(pieces), size)]


def partial_outputs(pieces: list, unit: int) -> list[tuple[int, int]]:
    """Chapter ranges of fixed-width batches covering ``pieces``.

    A short final batch is kept rather than dropped, and a lone leftover
    chapter still gets its own summary.
    """

    if unit < 1:
        raise MergeError(f"Batch width must be at least 1, got {unit}")
    ranges = []
    for i in range(0, len(pieces), unit):
        chunk = pieces[i:i + unit]
        ranges.append((chunk[0].start, chunk[-1].end))
    return ranges


def pack_groups(pieces: list, *, fan_in: int, fits) -> list[list]:
    """Group consecutive pieces so each group fits one prompt, respecting ``fan_in``.

    ``fits`` answers "would these pieces fit in one prompt?"; it is supplied by
    the caller because answering it needs a tokenizer. Pieces are added to the
    current group for as long as both the token budget and ``fan_in`` allow,
    which keeps the tree as shallow as the window permits instead of forcing a
    fixed shape.
    """

    if fan_in < 1:
        raise MergeError(f"fan_in must be at least 1, got {fan_in}")
    groups: list[list] = []
    current: list = []
    for piece in pieces:
        if current and (len(current) >= fan_in or not fits([*current, piece])):
            groups.append(current)
            current = [piece]
        else:
            current.append(piece)
    if current:
        groups.append(current)
    return groups


def rounds_needed(count: int, fan_in: int) -> int:
    """How many merge rounds it takes to collapse ``count`` pieces into one.

    Assumes every round merges at most ``fan_in`` pieces. Used only to describe
    the plan up front (``--dry-run`` and the progress line); the real run
    decides each round from measured token counts, because recap lengths vary
    far too much for a chapter count to predict them.
    """

    if fan_in < 2:
        raise MergeError(f"fan_in must be at least 2, got {fan_in}")
    rounds = 0
    while count > 1:
        count = -(-count // fan_in)  # ceiling division
        rounds += 1
    return rounds
