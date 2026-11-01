# tests/test_comic_manifest.py
from pathlib import Path

from comic_manifest import (
    ChapterRecord,
    Manifest,
    assign,
    load_manifest,
    ordinal_of_label,
    parse_label_number,
    save_manifest,
)

# Raw AJAX order is newest-first, exactly as the site returns it.
NEWEST_FIRST = [
    ("chapter-177", "Chapter 177"),
    ("chapter-175-19", "Chapter 175.19"),
    ("chapter-175", "Chapter 175"),
    ("chapter-174", "Chapter 174"),
]


def _empty():
    return Manifest(series="TBATE", site="s", chapters=[])


def test_the_newest_first_list_is_reversed_into_reading_order():
    manifest = assign(_empty(), NEWEST_FIRST)
    assert {c.ordinal: c.label for c in manifest.chapters} == {
        1: "Chapter 174",
        2: "Chapter 175",
        3: "Chapter 175.19",
        4: "Chapter 177",
    }


def test_list_order_beats_numeric_order_for_inserted_chapters():
    # 175.19 was inserted after 175; sorting by number would put it before 175.
    manifest = assign(_empty(), NEWEST_FIRST)
    assert ordinal_of_label(manifest, "Chapter 175.19") > ordinal_of_label(
        manifest, "Chapter 175"
    )


def test_existing_ordinals_survive_a_retroactive_insertion():
    first = assign(_empty(), NEWEST_FIRST)
    second = assign(first, NEWEST_FIRST + [("chapter-173", "Chapter 173")])
    for label in ("Chapter 177", "Chapter 175.19", "Chapter 175", "Chapter 174"):
        assert ordinal_of_label(second, label) == ordinal_of_label(first, label)


def test_a_new_chapter_never_takes_a_used_ordinal():
    first = assign(_empty(), NEWEST_FIRST)
    second = assign(first, NEWEST_FIRST + [("chapter-174-5", "Chapter 174.5")])
    ordinals = [c.ordinal for c in second.chapters]
    assert len(ordinals) == len(set(ordinals))


def test_a_fractional_label_parses_exactly():
    assert parse_label_number("Chapter 175.10") == 175.10
    assert parse_label_number("Chapter 175.1") == 175.1
    assert parse_label_number("Chapter 280 - 237") == 280.0


def test_records_added_fields_are_preserved_across_a_reassign():
    manifest = assign(_empty(), NEWEST_FIRST)
    manifest.chapters[0].pages = 7
    again = assign(manifest, NEWEST_FIRST)
    assert again.chapters[0].pages == 7


def test_a_manifest_round_trips_through_disk(tmp_path: Path):
    path = tmp_path / "chapters.json"
    original = assign(_empty(), NEWEST_FIRST)
    original.chapters[0].duplicate_of = None
    original.chapters[0].content_hash = "abc"
    save_manifest(path, original)
    loaded = load_manifest(path)
    assert [
        (c.ordinal, c.slug, c.label, c.pages, c.content_hash, c.duplicate_of)
        for c in loaded.chapters
    ] == [
        (c.ordinal, c.slug, c.label, c.pages, c.content_hash, c.duplicate_of)
        for c in original.chapters
    ]


def test_a_missing_manifest_is_empty(tmp_path: Path):
    assert load_manifest(tmp_path / "nope.json").chapters == []


def test_an_empty_listing_keeps_what_is_already_known():
    first = assign(_empty(), NEWEST_FIRST)
    assert assign(first, []).chapters == first.chapters


def test_each_retroactive_insertion_reports_its_own_ordinal(capsys):
    # 173 and 174.5 are inserted in the middle of the reading order, so both
    # land at the end. Each note must name its own ordinal, not the final max.
    first = assign(_empty(), NEWEST_FIRST)
    capsys.readouterr()  # drop anything from the setup call
    listing = [
        ("chapter-177", "Chapter 177"),
        ("chapter-175-19", "Chapter 175.19"),
        ("chapter-175", "Chapter 175"),
        ("chapter-174-5", "Chapter 174.5"),
        ("chapter-174", "Chapter 174"),
        ("chapter-173", "Chapter 173"),
    ]

    second = assign(first, listing)

    assert capsys.readouterr().out.splitlines() == [
        "note: Chapter 173 is listed before Chapter 174 but was appended as chapter 5",
        "note: Chapter 174.5 is listed before Chapter 175 but was appended as chapter 6",
    ]
    assert ordinal_of_label(second, "Chapter 173") == 5
    assert ordinal_of_label(second, "Chapter 174.5") == 6

