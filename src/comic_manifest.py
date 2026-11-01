"""Chapter manifest: pins each comic chapter's reading-order position to the site's slug.

This ensures that re-scraping the site months later can never renumber chapters
that have already been downloaded.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ChapterRecord:
    """A single chapter in the manifest."""

    ordinal: int
    slug: str
    label: str
    number: float
    pages: int = 0
    content_hash: str = ""
    duplicate_of: int | None = None


@dataclass
class Manifest:
    """The manifest for a series of chapters."""

    series: str
    site: str
    chapters: list[ChapterRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "series": self.series,
            "site": self.site,
            "chapters": [asdict(c) for c in self.chapters],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        """Create a Manifest from a dict (e.g., loaded from JSON)."""
        chapters = [
            ChapterRecord(**c) for c in data.get("chapters", [])
        ]
        return cls(
            series=data.get("series", ""),
            site=data.get("site", ""),
            chapters=chapters,
        )


def load_manifest(path: Path) -> Manifest:
    """Load a manifest from a JSON file.

    A missing file yields an empty manifest.
    """
    if not path.exists():
        return Manifest(series="", site="", chapters=[])

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return Manifest.from_dict(data)


def save_manifest(path: Path, manifest: Manifest) -> None:
    """Save a manifest to a JSON file atomically.

    Writes to a temp file, then uses os.replace to atomically rename it.
    """
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    os.replace(tmp, path)


def parse_label_number(label: str) -> float:
    """Parse a chapter label to extract its numeric value.

    "Chapter 175.10" -> 175.10
    "Chapter 175.1" -> 175.1
    "Chapter 280 - 237" -> 280.0 (takes the first number)
    """
    # Find all numbers in the label (integers and decimals)
    matches = re.findall(r"\d+(?:\.\d+)?", label)
    if not matches:
        return 0.0
    return float(matches[0])


def ordinal_of_label(manifest: Manifest, label: str) -> int | None:
    """Find the ordinal for a given chapter label.

    Returns the ordinal if found, None otherwise.
    """
    for chapter in manifest.chapters:
        if chapter.label == label:
            return chapter.ordinal
    return None


def assign(manifest: Manifest, listing: list[tuple[str, str]]) -> Manifest:
    """Assign ordinals to chapters based on the site's listing.

    Args:
        manifest: The existing manifest (may be empty).
        listing: A list of (slug, label) tuples in the site's newest-first order.

    Returns:
        A new Manifest with chapters sorted by ordinal.

    Unknown slugs are appended at the end with ordinal = max(existing) + 1.
    For each such retroactive insertion, prints a note to stdout.
    """
    # Group existing records by slug, preserving all fields
    existing_by_slug: dict[str, ChapterRecord] = {
        c.slug: c for c in manifest.chapters
    }

    # Get max existing ordinal (or 0 if none)
    max_ordinal = 0
    if manifest.chapters:
        max_ordinal = max(c.ordinal for c in manifest.chapters)

    # Reverse listing to oldest-first (reading order)
    reversed_listing = list(reversed(listing))

    # Track which slugs are in the new listing
    slugs_in_listing = {slug for slug, _ in listing}

    # Walk the listing in reading order (oldest first)
    # Known slugs keep their records; unknown slugs get new ordinals at the end
    new_chapters: list[ChapterRecord] = []

    for slug, label in reversed_listing:
        if slug in existing_by_slug:
            # Known chapter - keep its record unchanged
            new_chapters.append(existing_by_slug[slug])
        else:
            # Unknown chapter - assign next ordinal at the end
            max_ordinal += 1
            new_record = ChapterRecord(
                ordinal=max_ordinal,
                slug=slug,
                label=label,
                number=parse_label_number(label),
            )
            new_chapters.append(new_record)

    # Now handle retroactive insertions: if an unknown slug appears BEFORE
    # a known slug in the original (newest-first) listing, we need to report it.
    # The listing was newest-first, so we walk it and detect when an unknown slug
    # appears before a known one.
    for i, (slug, label) in enumerate(listing):
        if slug not in existing_by_slug:
            # This is a new chapter. Check if it appears before any known chapter.
            # In newest-first order, anything before this position is "newer" (higher ordinal).
            # We need to find chapters that are OLDER (lower ordinal) than this one.
            for j in range(i + 1, len(listing)):
                later_slug, later_label = listing[j]
                if later_slug in existing_by_slug:
                    # The unknown slug appears before this known slug in newest-first order,
                    # which means in reading order (oldest-first), the known slug comes first.
                    # So the unknown chapter should come AFTER the known one, but we appended it.
                    # Print the note using the newly-appended chapter's label first.
                    print(f"note: {label} is listed before {later_label} but was appended as chapter {max_ordinal}")
                    break

    # Keep records for slugs no longer in the listing (site temporarily dropped a chapter)
    for slug, record in existing_by_slug.items():
        if slug not in slugs_in_listing:
            new_chapters.append(record)

    # Sort by ordinal
    new_chapters.sort(key=lambda c: c.ordinal)

    return Manifest(
        series=manifest.series,
        site=manifest.site,
        chapters=new_chapters,
    )
