"""Chapter file writing for the fast scraper.

Chapters are written as ``chapter_<n:05d>.txt`` (``chapter_00012.txt``). Two
older forms still exist in the corpus and are **deprecated but recognised**:
the zero-padded ``chapter_<n:04d>.txt`` the legacy Selenium scraper emitted, and
the original unpadded ``chapter_<n>.txt``. A chapter counts as present if any
form exists, so the fast engine never re-scrapes and duplicates a chapter that
is already on disk under an old name.

The deprecated lookups exist only as a migration window: ``corpus_tools`` can
rename everything to the five-digit form, after which they can be deleted.
"""

from __future__ import annotations

import os
from pathlib import Path

PAD = 5


def chapter_filename(number: int) -> str:
    """Return the file name the engine writes for ``number``."""

    return f"chapter_{number:0{PAD}d}.txt"


def deprecated_chapter_filenames(number: int) -> tuple[str, ...]:
    """Old name forms still accepted on read. Remove once the corpus is migrated."""

    return (f"chapter_{number:04d}.txt", f"chapter_{number}.txt")


def _novel_dir(save_root, novel_name: str) -> Path:
    return Path(save_root) / novel_name


def existing_chapter(save_root, novel_name: str, number: int) -> Path | None:
    """Return the path of an existing chapter file, checking the current name first."""

    directory = _novel_dir(save_root, novel_name)
    for name in (chapter_filename(number), *deprecated_chapter_filenames(number)):
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def write_chapter(
    save_root,
    novel_name: str,
    number: int,
    title: str,
    content: str,
    overwrite: bool = False,
) -> Path | None:
    """Write ``title`` + ``content`` atomically; skip if present unless ``overwrite``.

    Returns the path written, or ``None`` when the chapter already exists and
    ``overwrite`` is false. The file is written to a sibling temp file and
    renamed into place so a crash mid-write cannot leave a half-written chapter.
    """

    directory = _novel_dir(save_root, novel_name)
    directory.mkdir(parents=True, exist_ok=True)

    existing = existing_chapter(save_root, novel_name, number)
    if not overwrite and existing is not None:
        return None

    # Overwrite an existing old-named file in place so we do not leave a copy.
    target = existing if existing is not None else directory / chapter_filename(number)

    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(f"{title}\n\n{content}", encoding="utf-8")
    os.replace(tmp, target)
    return target
