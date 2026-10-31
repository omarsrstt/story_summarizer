"""Corpus maintenance for scraped chapter directories.

Two jobs, both **read-only unless told otherwise**:

``rename-padding``
    Rename every ``chapter_<n>.txt`` to the five-digit ``chapter_<n:05d>.txt``.

``strip-boilerplate``
    Remove site furniture from the end of chapters.

The trailer remover deliberately contains **no advert text**. Site furniture
changes whenever the site changes, so instead of naming strings it learns them:
a line is boilerplate when the *same* line is the last line of a large share of
the chapters in the directory. Real prose essentially never repeats verbatim
across hundreds of chapters; "read more at ..." footers do. Nothing is removed
from the middle of a chapter, so dialogue that happens to quote the phrase is
safe.

Both operations print a plan and change nothing until ``apply=True``. A backup
copy is made before any write, because ``novels/`` is not tracked by git.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from scraper_store import chapter_filename

_CHAPTER_RE = re.compile(r"^chapter_(\d+)\.txt$")

# A trailing line must repeat in at least this share of the directory's chapters
# before it is treated as site furniture rather than prose.
MIN_SHARE = 0.10
MIN_CHAPTERS = 5


@dataclass
class StripReport:
    """What a trailer sweep found, and (unless dry run) removed."""

    lines_removed: int = 0
    files_changed: int = 0
    removed_texts: list[str] = field(default_factory=list)
    applied: bool = False
    reason: str = ""


def chapter_files(directory) -> list[Path]:
    """Chapter files in ``directory``, sorted by chapter number."""

    directory = Path(directory)
    files = [
        p for p in directory.glob("chapter_*.txt") if _CHAPTER_RE.match(p.name)
    ]
    return sorted(files, key=lambda p: int(_CHAPTER_RE.match(p.name).group(1)))


def plan_padding_rename(directory) -> list[tuple[Path, Path]]:
    """Return ``(source, target)`` pairs for chapters not yet five-digit padded."""

    moves: list[tuple[Path, Path]] = []
    for path in chapter_files(directory):
        number = int(_CHAPTER_RE.match(path.name).group(1))
        target = path.with_name(chapter_filename(number))
        if target != path:
            moves.append((path, target))
    return moves


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _write_lines(path: Path, lines: list[str]) -> None:
    """Write ``lines`` back using the file's own line endings.

    ``Path.write_text`` translates ``\\n`` to the platform default, which would
    silently rewrite a CRLF corpus as LF (or the reverse) on another machine.
    Reading and writing bytes keeps the prose byte-identical apart from the
    lines actually removed.
    """

    raw = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    text = newline.join(lines)
    if not raw or raw.endswith((b"\n", b"\r")):
        text += newline
    path.write_bytes(text.encode("utf-8"))


def strip_recurring_trailers(
    directory, apply: bool = True, backup: bool = True
) -> StripReport:
    """Remove trailing lines that recur across the directory's chapters.

    Walks from the end of each chapter, removing a trailing line only when that
    exact line (stripped) is also the trailing line of ``MIN_SHARE`` of the
    chapters. Repeats until no further line qualifies, so a two-line footer is
    removed one line per round.

    The CLI defaults to a dry run (``apply=False``) and prints the plan; callers
    that have already shown the plan may pass ``apply=True``.
    """

    directory = Path(directory)
    files = chapter_files(directory)
    report = StripReport(applied=apply)

    if len(files) < MIN_CHAPTERS:
        report.reason = (
            f"only {len(files)} chapter(s): too few to tell boilerplate from prose"
        )
        return report

    bodies = {path: _read_lines(path) for path in files}
    threshold = max(2, int(len(files) * MIN_SHARE))

    while True:
        tails = {}
        for path, lines in bodies.items():
            if lines and lines[-1].strip():
                tails.setdefault(lines[-1].strip(), 0)
                tails[lines[-1].strip()] += 1
        recurring = {text for text, count in tails.items() if count >= threshold}
        if not recurring:
            break

        # Never strip a chapter down to nothing: boilerplate-only files stay put.
        removable = [
            path
            for path, lines in bodies.items()
            if lines and lines[-1].strip() in recurring and _has_content_above(lines)
        ]
        if not removable:
            break

        for path in removable:
            bodies[path].pop()
            report.lines_removed += 1
        for text in recurring:
            if text not in report.removed_texts:
                report.removed_texts.append(text)

    report.files_changed = sum(
        1 for path in files if len(bodies[path]) != len(_read_lines(path))
    )

    if apply and report.files_changed:
        if backup:
            _backup(directory)
        for path in files:
            if len(bodies[path]) != len(_read_lines(path)):
                _write_lines(path, bodies[path])

    return report


def _has_content_above(lines: list[str]) -> bool:
    """True when removing the last line still leaves a non-blank line."""

    return any(line.strip() for line in lines[:-1])


def _backup(directory: Path) -> Path:
    """Copy ``directory`` beside itself before modifying it."""

    target = directory.parent / f"{directory.name}.bak"
    n = 1
    while target.exists():
        target = directory.parent / f"{directory.name}.bak{n}"
        n += 1
    shutil.copytree(directory, target)
    return target


def apply_padding_rename(
    directory, apply: bool = False, backup: bool = True
) -> list[tuple[Path, Path]]:
    """Rename chapters to five-digit padding; returns the moves performed."""

    moves = plan_padding_rename(directory)
    if not moves:
        return []
    if apply:
        if backup:
            _backup(Path(directory))
        for source, target in moves:
            source.rename(target)
    return moves


# --- command line ---------------------------------------------------------
#
# Both commands are dry runs by default: they print what would change and touch
# nothing until --apply is passed.


def _report_rename(moves) -> None:
    if not moves:
        print("Filenames already use five-digit padding; nothing to rename.")
        return
    print(f"{len(moves)} file(s) would be renamed:")
    for source, target in moves[:5]:
        print(f"  {source.name}  ->  {target.name}")
    if len(moves) > 5:
        print(f"  ... and {len(moves) - 5} more")


def _report_strip(report) -> None:
    if report.reason:
        print(f"Skipped: {report.reason}.")
        return
    if not report.files_changed:
        print("No recurring trailer lines found; nothing to remove.")
        return
    print(f"{report.lines_removed} line(s) in {report.files_changed} file(s):")
    for text in report.removed_texts:
        shown = text if len(text) <= 70 else text[:67] + "..."
        print(f"  - {shown!r}")


def _print_summary(directory: Path) -> None:
    files = chapter_files(directory)
    if not files:
        print(f"No chapter files found in {directory}.")
        return
    numbers = [int(_CHAPTER_RE.match(f.name).group(1)) for f in files]
    print(
        f"{len(files)} chapter(s): {min(numbers)}-{max(numbers)}, "
        f"{len(set(numbers))} distinct."
    )


def main(argv=None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Maintain scraped chapter directories (dry run by default)."
    )
    parser.add_argument(
        "command",
        choices=["rename-padding", "strip-boilerplate", "all"],
        help="rename to five-digit padding, strip recurring trailers, or both",
    )
    parser.add_argument("directory", help="chapter directory, e.g. novels/<slug>")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually change files (default is a dry run that only reports)",
    )
    args = parser.parse_args(argv)

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Not a directory: {directory}", file=sys.stderr)
        return 2

    _print_summary(directory)
    print()

    # Plan every requested step first, then change files at most once, under a
    # single backup. Doing the two steps independently would copy the whole
    # directory twice and leave <dir>.bak and <dir>.bak1 behind.
    moves = plan_padding_rename(directory) if args.command in (
        "rename-padding",
        "all",
    ) else []
    strip_report = (
        strip_recurring_trailers(directory, apply=False)
        if args.command in ("strip-boilerplate", "all")
        else None
    )

    if args.command in ("rename-padding", "all"):
        _report_rename(moves)
        print()

    if strip_report is not None:
        _report_strip(strip_report)
        print()

    something_to_do = bool(moves) or bool(strip_report and strip_report.files_changed)

    if not args.apply:
        if something_to_do:
            print("Dry run: nothing was changed; re-run with --apply to apply.")
        else:
            print("Dry run: nothing would change.")
        _print_summary(directory)
        return 0

    if not something_to_do:
        print("Nothing to change.")
        _print_summary(directory)
        return 0

    _backup(directory)

    if moves:
        apply_padding_rename(directory, apply=True, backup=False)
        print(f"Renamed {len(moves)} file(s).")

    if strip_report is not None and strip_report.files_changed:
        applied = strip_recurring_trailers(directory, apply=True, backup=False)
        print(f"Removed {applied.lines_removed} line(s) from {applied.files_changed} file(s).")

    print("Backup written beside the directory.")
    print()

    _print_summary(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
