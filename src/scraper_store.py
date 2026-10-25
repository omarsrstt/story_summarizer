"""Chapter file writing for the fast scraper.

Two filename forms exist in the corpus: the unpadded ``chapter_<n>.txt`` written
by hand and the zero-padded ``chapter_<n:04d>.txt`` the legacy Selenium scraper
emits. A chapter counts as present if either form exists, so the fast engine
never duplicates a chapter the legacy engine already saved.
"""

from __future__ import annotations

import os
from pathlib import Path


def chapter_filename(number: int) -> str:
    """Return the file name the fast engine writes for ``number``."""

    return f"chapter_{number}.txt"


def _novel_dir(save_root, novel_name: str) -> Path:
    return Path(save_root) / novel_name


def existing_chapter(save_root, novel_name: str, number: int) -> Path | None:
    """Return the path of an existing chapter file, checking both name forms."""

    directory = _novel_dir(save_root, novel_name)
    for name in (chapter_filename(number), f"chapter_{number:04d}.txt"):
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

    target = directory / chapter_filename(number)
    if not overwrite and existing_chapter(save_root, novel_name, number) is not None:
        return None

    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(f"{title}\n\n{content}", encoding="utf-8")
    os.replace(tmp, target)
    return target
