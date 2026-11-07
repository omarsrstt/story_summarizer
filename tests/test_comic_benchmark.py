"""Tests for the novel-backed benchmark.

The benchmark exists because the novel-free validator cannot see loss: without a
reference, a line the OCR dropped and a line the adapter cut produce identical
findings. Against the novel we can at least ask whether the words the transcript
*does* contain are spelled the way the source spells them.

These tests pin the three things that make its numbers trustworthy:

* alignment is by content, and an ambiguous chapter is reported *unmatched*
  rather than assigned to a wrong counterpart;
* a hand-checked pairing that disagrees with the measured best is a loud error,
  because a silent mismatch would blame the pipeline for a benchmark defect;
* name spelling is only reported above a sample floor, and distance-2 "loose"
  guesses never inflate the explained rate.
"""

import json
from pathlib import Path

import pytest

from comic_benchmark import (
    MIN_NAMES_FOR_FIDELITY,
    ChapterResult,
    NovelIndex,
    edit_distance,
    load_pairing,
    measure,
    normalise_apostrophes,
    words_in,
)


def _novel(tmp_path: Path, chapters: dict[int, str]) -> Path:
    directory = tmp_path / "novel"
    directory.mkdir()
    for number, text in chapters.items():
        (directory / f"chapter_{number:05d}.txt").write_text(text, encoding="utf-8")
    return directory


def _transcript(chapter: int, body: str) -> str:
    return f"# comic chapter {chapter}\n\n--- page 1 ---\n{body}\n"


# -- text normalisation ----------------------------------------------------


def test_curly_apostrophes_are_folded_so_the_novel_and_transcript_match():
    # The novel stores U+2019. Without folding, every contraction in the novel is
    # a different string from the same contraction in a transcript.
    assert normalise_apostrophes("don\u2019t") == "don't"
    assert normalise_apostrophes("\u201cdirector\u201d") == '"director"'


def test_words_in_lowercases_and_splits_on_the_transcript_style():
    assert words_in("THAT'S a MANA-core!") == ["that's", "mana", "core"]


def test_words_in_ignores_one_and_two_letter_tokens():
    # "I" and "a" carry no alignment signal, and counting them makes every
    # chapter look similar to every other.
    assert words_in("I am a man") == ["man"]


# -- edit distance ---------------------------------------------------------


def test_edit_distance_counts_single_char_mistakes():
    assert edit_distance("earthborm", "earthborn") == 1
    assert edit_distance("ashber", "ashbern") == 1
    assert edit_distance("same", "same") == 0


def test_edit_distance_gives_up_past_the_limit():
    # The cap is what makes a thousand-candidate search affordable; it must
    # report "too far" rather than a huge exact number.
    assert edit_distance("kwni", "dicathen", limit=2) == 3


# -- alignment -------------------------------------------------------------


def test_alignment_finds_the_chapter_that_shares_rare_words(tmp_path):
    directory = _novel(tmp_path, {
        1: "The merchant crossed the river. " * 30,
        2: "Xyrus Academy stood above the Elenoir forest. " * 30,
        3: "The council argued about the mountain pass. " * 30,
    })
    index = NovelIndex(directory)

    result = measure(
        index, comic_number=1,
        text=_transcript(1, "XYRUS ACADEMY stood above the ELENOIR forest."),
    )

    assert result.best == 2
    assert result.matched


def test_a_chapter_sharing_nothing_is_reported_unmatched(tmp_path):
    # This is the bonus-chapter case. Something must always be "best", so the
    # only honest answer is to say the margin is too small to mean anything.
    directory = _novel(tmp_path, {
        1: "Xyrus Academy stood above the Elenoir forest. " * 30,
        2: "The council argued about the mountain pass. " * 30,
    })
    index = NovelIndex(directory)

    result = measure(index, comic_number=1, text=_transcript(1, "zxqv wplk mmmn"))

    assert not result.matched
    assert result.margin < 0.02


def test_alignment_never_uses_the_chapter_number(tmp_path):
    # The measured offsets on this series drift, so an ordinal join would be
    # wrong on most chapters. Content must decide, even against the number.
    directory = _novel(tmp_path, {
        11: "Xyrus Academy stood above the Elenoir forest. " * 30,
        22: "The council argued about the mountain pass. " * 30,
    })
    index = NovelIndex(directory)

    result = measure(index, comic_number=22,
                     text=_transcript(22, "XYRUS ACADEMY ELENOIR forest"))

    assert result.best == 11


# -- name spelling ---------------------------------------------------------


def test_spelling_accuracy_is_none_below_the_sample_floor():
    # Comic 1 recognises three names, so its raw accuracy is noise.
    result = ChapterResult(
        comic_number=1, best=1, score=0.2, second=2, second_score=0.1,
        matched=True, name_exact=MIN_NAMES_FOR_FIDELITY - 1,
    )
    assert result.spelling_accuracy is None


def test_the_report_never_prints_accuracy_below_the_floor():
    # The floor has to reach the *output*, not just the property. Without this
    # test, setting MIN_NAMES_FOR_FIDELITY to a billion left the suite green.
    from comic_benchmark import report

    result = ChapterResult(
        comic_number=1, best=1, score=0.2, second=2, second_score=0.1,
        matched=True, name_tokens=2, name_exact=0, suspicious=0,
    )
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report([result], glossary=[])
    output = buffer.getvalue()

    assert "too few (0)" in output
    assert "0.0% of recognised" not in output


def test_spelling_accuracy_is_reported_at_the_floor():
    result = ChapterResult(
        comic_number=1, best=1, score=0.2, second=2, second_score=0.1,
        matched=True, name_exact=MIN_NAMES_FOR_FIDELITY // 2,
        name_misread=MIN_NAMES_FOR_FIDELITY // 2,
    )
    assert result.spelling_accuracy == pytest.approx(0.5)


def test_a_misread_name_is_counted_as_misread_not_exact(tmp_path):
    directory = _novel(tmp_path, {
        1: "Earthborn Dicathen Elenoir Sapin Xyrus. " * 20,
    })
    index = NovelIndex(directory)

    result = measure(index, comic_number=1,
                     text=_transcript(1, "Earthborm DICATHEN ELENOIR SAPIN XYRUS"))

    # "Earthborm" is the real defect this measures: one character from the
    # novel's "Earthborn".
    assert result.name_tokens == 5
    assert result.name_exact == 4
    assert result.name_misread == 1


def test_garbage_is_counted_unknown_and_not_as_a_spelling_failure(tmp_path):
    # "KWNI" is non-dictionary and uppercase, so it is name-shaped. Dividing
    # exact matches by every name-shaped token made chapter 2 read 53% and
    # imply a spelling problem that did not exist; it belongs in "unknown".
    names = " ".join(["DICATHEN"] * 10)
    directory = _novel(tmp_path, {1: "Dicathen. " * 20})
    index = NovelIndex(directory)

    result = measure(index, comic_number=1, text=_transcript(1, f"KWNI {names}"))

    assert result.name_tokens == 11
    assert result.name_exact == 10
    assert result.name_unknown == 1
    assert result.recognised_names == 10
    # All ten recognised names are exact: the garbage lowers no spelling score,
    # it is reported separately as unrecognised.
    assert result.spelling_accuracy == pytest.approx(1.0)


def test_the_benchmark_passes_the_real_glossary_so_counts_match_comic_check(tmp_path):
    # With an empty glossary every known name is "suspicious" and then
    # "explained" by the novel, so the benchmark's rate disagreed with the
    # validator's for the same file. The glossary must be threaded through.
    directory = _novel(tmp_path, {1: "Dicathen Elenoir. " * 20})
    index = NovelIndex(directory)
    text = _transcript(1, "DICATHEN ELENOIR")

    with_glossary = measure(index, 1, text, glossary=["DICATHEN", "ELENOIR"])
    without = measure(index, 1, text, glossary=[])

    assert with_glossary.suspicious == 0
    assert without.suspicious == 2


# -- explainability --------------------------------------------------------


def test_a_near_miss_of_a_novel_name_is_explained(tmp_path):
    directory = _novel(tmp_path, {1: "Ashbern Dicathen Elenoir. " * 20})
    index = NovelIndex(directory)

    result = measure(index, comic_number=1, text=_transcript(1, "ASHBER"))

    assert result.suspicious == 1
    assert result.explained == 1
    assert result.guessed == 0


def test_a_two_edit_match_is_a_guess_and_does_not_lift_the_score(tmp_path):
    # Distance 2 also matches unrelated words ("MBLE" -> "Myre"), so it is
    # reported separately. Counting it as explained would inflate the rate.
    directory = _novel(tmp_path, {1: "Myre Dicathen Elenoir. " * 20})
    index = NovelIndex(directory)

    result = measure(index, comic_number=1, text=_transcript(1, "MBLE"))

    assert result.explained == 0
    assert result.guessed == 1


def test_noise_is_neither_explained_nor_guessed(tmp_path):
    directory = _novel(tmp_path, {1: "Dicathen Elenoir. " * 20})
    index = NovelIndex(directory)

    result = measure(index, comic_number=1, text=_transcript(1, "dWHHL"))

    assert result.explained == 0
    assert result.guessed == 0
    assert result.unexplained == ["dWHHL"]


# -- the reliability guard -------------------------------------------------
#
# Measured on the real corpus, the unguarded matcher reported five "misread
# names" of which two were sound effects and one was the novel's own spelling.
# A named non-defect is worse than silence: "fix this name" is actionable and
# wrong.


def test_a_one_off_title_case_token_is_not_a_name(tmp_path):
    # "Agh" appears exactly once in the whole novel, beside OCR rubble
    # ("REAARLAA,P AGB AP"). Flagging "AGB -> Agh" sends someone after nothing.
    directory = _novel(tmp_path, {1: "Agh. " * 1 + "Dicathen Elenoir. " * 20})
    index = NovelIndex(directory)

    assert index.is_reliable_name("agh") is False
    assert index.nearest_name("AGB", 1) is None


def test_an_all_caps_token_is_not_treated_as_a_name(tmp_path):
    # "NOOO" recurs, but only ever shouted. A shout is not a name.
    directory = _novel(tmp_path, {1: "NOOO NOOO. " * 10})
    index = NovelIndex(directory)

    assert index.is_reliable_name("nooo") is False
    assert index.nearest_name("NOOI", 1) is None


def test_a_recurring_title_case_name_is_trusted(tmp_path):
    # The guard must not be so tight that it stops finding real misreads.
    directory = _novel(tmp_path, {1: "Earthborn Dicathen. " * 20})
    index = NovelIndex(directory)

    assert index.is_reliable_name("earthborn") is True
    assert index.nearest_name("Earthborm", 1) == ("earthborn", 1)


def test_a_name_spelled_correctly_is_exact_even_if_the_novel_uses_it_once(tmp_path):
    # is_name deliberately has no guard: if the transcript spells it the way the
    # novel does, that is correct regardless of the novel's frequency.
    directory = _novel(tmp_path, {1: "Agh. " * 1 + "Dicathen. " * 20})
    index = NovelIndex(directory)

    assert index.is_name("agh") is True
    assert index.is_reliable_name("agh") is False


def test_a_misread_of_a_noise_token_is_not_reported_as_a_misread(tmp_path):
    directory = _novel(tmp_path, {1: "Agh. " + "Dicathen Elenoir. " * 20})
    index = NovelIndex(directory)

    result = measure(index, comic_number=1, text=_transcript(1, "AGB"))

    assert result.name_misread == 0
    assert result.misread == {}
    assert result.name_unknown == 1


# -- glossary gaps ---------------------------------------------------------
#
# The glossary's stated job is to stop the segmenter splitting a name, so a name
# the glossary omits is only a *gap* when `resegment` would actually split it.
# Measured on the real corpus, dropping that condition reported 11 candidates of
# which 10 needed no entry ("Dawsid", "Eksire", "Ack", "Aww", "Here's").


def test_a_name_the_segmenter_would_split_is_reported_as_a_gap(tmp_path):
    # "Dawsid" is left whole by the splitter, "Goodsky" is not ("good sky").
    # Only the second needs a glossary entry.
    directory = _novel(tmp_path, {
        1: "Dawsid Goodsky. " * 20,
    })
    index = NovelIndex(directory)

    result = measure(
        index, comic_number=1,
        text=_transcript(1, "Dawsid Goodsky"),
        glossary=[],
    )

    assert result.glossary_gaps == ["Goodsky"]


def test_a_name_the_segmenter_leaves_whole_is_not_a_gap(tmp_path):
    directory = _novel(tmp_path, {1: "Dawsid. " * 20})
    index = NovelIndex(directory)

    result = measure(
        index, comic_number=1, text=_transcript(1, "Dawsid"), glossary=[]
    )

    assert result.glossary_gaps == []


def test_a_name_already_in_the_glossary_is_not_a_gap(tmp_path):
    directory = _novel(tmp_path, {1: "Goodsky. " * 20})
    index = NovelIndex(directory)

    result = measure(
        index, comic_number=1,
        text=_transcript(1, "Goodsky"),
        glossary=["GOODSKY"],
    )

    assert result.glossary_gaps == []


# -- the glossary is the comic's authority --------------------------------
#
# Measured trap: the glossary has ASHBERN, the novel uses "Ashber" 39 times and
# never "Ashbern" -- but the raw OCR says "Ashbern Town" at 0.96 confidence, so
# the comic spells it that way and the glossary is right. An earlier version
# called this a "glossary defect", which would have sent someone to corrupt a
# correct tracked file on the novel's authority.


def test_a_glossary_spelling_the_novel_lacks_is_correct_not_misread(tmp_path):
    directory = _novel(tmp_path, {1: "Ashber town. " * 20})
    index = NovelIndex(directory)

    result = measure(
        index, comic_number=1,
        text=_transcript(1, "ASHBERN town"),
        glossary=["ASHBERN"],
    )

    # Counted as correct spelling, not as a misread of the novel's form.
    assert result.name_misread == 0
    assert "ASHBERN" not in result.misread
    # But the divergence is still reported, so it is visible and checkable.
    assert result.spelling_differs == {"ASHBERN": "Ashber"}


def test_the_glossary_spelling_never_reports_a_gap(tmp_path):
    # A glossary term is covered, so it can never be a missing entry.
    directory = _novel(tmp_path, {1: "Ashber town. " * 20})
    index = NovelIndex(directory)

    result = measure(
        index, comic_number=1,
        text=_transcript(1, "ASHBERN"),
        glossary=["ASHBERN"],
    )

    assert result.glossary_gaps == []


def test_a_misread_reports_how_far_the_name_is_from_the_aligned_chapter(tmp_path):
    # The distance is the difference between evidence and coincidence, so it
    # must be attached to the finding rather than computed later.
    directory = _novel(tmp_path, {
        1: "Earthborn Dicathen. " * 20,
        200: "Earthborn Dicathen. " * 20,
    })
    index = NovelIndex(directory)

    result = measure(index, comic_number=1, text=_transcript(1, "Earthborm"))

    assert result.name_misread == 1
    spelling, distance = result.misread["Earthborm"]
    assert spelling == "Earthborn"
    assert distance == 0


# -- pairing ---------------------------------------------------------------


def test_pairing_reads_the_pairs_block(tmp_path):
    path = tmp_path / "pairing.json"
    path.write_text(
        json.dumps({"_comment": ["why"], "pairs": {"1": 1, "147": 100}}),
        encoding="utf-8",
    )
    assert load_pairing(path) == {1: 1, 147: 100}


def test_pairing_accepts_a_bare_mapping(tmp_path):
    # An older hand-made file has no "pairs" wrapper; it must keep working.
    path = tmp_path / "pairing.json"
    path.write_text(json.dumps({"2": 2}), encoding="utf-8")
    assert load_pairing(path) == {2: 2}


def test_no_pairing_file_means_no_expectation(tmp_path):
    assert load_pairing(None) == {}


def test_a_pairing_that_disagrees_with_content_is_flagged(tmp_path):
    # A wrong pairing must never be blamed on the pipeline, so it is recorded
    # loudly and main() exits non-zero.
    directory = _novel(tmp_path, {
        1: "Xyrus Academy Elenoir forest. " * 30,
        2: "Council mountain pass. " * 30,
    })
    index = NovelIndex(directory)

    result = measure(index, comic_number=5,
                     text=_transcript(5, "XYRUS ACADEMY ELENOIR forest"),
                     expected=2)

    assert result.best == 1
    assert result.wrong_pair


def test_a_pairing_that_agrees_is_not_flagged(tmp_path):
    directory = _novel(tmp_path, {1: "Xyrus Academy Elenoir. " * 30})
    index = NovelIndex(directory)

    result = measure(index, comic_number=1,
                     text=_transcript(1, "XYRUS ACADEMY ELENOIR"), expected=1)

    assert not result.wrong_pair


# -- the oracle boundary ---------------------------------------------------


def test_measuring_never_mutates_the_transcript(tmp_path):
    # The hard rule: the novel is a test oracle, never an input. A benchmark
    # that rewrites what it measures is worthless.
    directory = _novel(tmp_path, {1: "Xyrus Academy Elenoir. " * 30})
    index = NovelIndex(directory)
    path = tmp_path / "chapter_00001.txt"
    original = _transcript(1, "XYRUS ACADEMY ELENOIR")
    path.write_text(original, encoding="utf-8")

    measure(index, comic_number=1, text=path.read_text(encoding="utf-8"))

    assert path.read_text(encoding="utf-8") == original


def test_the_benchmark_is_not_imported_by_the_pipeline():
    # If the reader or summariser ever imported this, the novel would reach a
    # transcript. Checked as source text so the rule is enforced, not just
    # documented.
    source_root = Path(__file__).resolve().parent.parent / "src"
    for name in ("comic_reader.py", "comic_check.py", "comic_text.py"):
        source = (source_root / name).read_text(encoding="utf-8")
        assert "comic_benchmark" not in source, f"{name} imports the benchmark"
