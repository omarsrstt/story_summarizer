"""Tests for the transcript validator.

The validator answers one question without any novel: does this transcript
faithfully represent the page? Its findings are split by how certain the
evidence is, because a gate that fires on ambiguous signals gets ignored:

* **fail** -- unambiguous. A CJK glyph cannot be story text in an English
  comic; a fused run the shipped code would split means the transcript is
  stale or the pipeline regressed.
* **warn** -- a real smell, but the same shape as legitimate text. "man a"
  is what MANA became, but it is also ordinary English, so it cannot fail a run.
"""

from pathlib import Path

from comic_check import (
    check_pages,
    check_transcript,
    load_glossary,
    parse_transcript,
    suspicious_rate,
    suspicious_tokens,
)
from comic_text import keep_credit_mask


def _transcript(chapter: int = 1, body: str = "") -> str:
    return f"# TBATE chapter {chapter}\n\n{body}"


# -- parsing ---------------------------------------------------------------


def test_parse_reads_the_chapter_number_and_page_markers():
    text = _transcript(
        147,
        "--- page 1 ---\nHELLO\n--- page 2 ---\nWORLD\n",
    )
    t = parse_transcript(text)

    assert t.chapter == 147
    assert t.pages == [1, 2]
    assert [line.text for line in t.lines] == ["HELLO", "WORLD"]


def test_parse_skips_the_header_and_blank_lines():
    t = parse_transcript(_transcript(3, "--- page 4 ---\n\nART\n"))
    assert [(line.page, line.text) for line in t.lines] == [(4, "ART")]


def test_parse_of_a_chapter_without_markers_is_not_an_error():
    # Hand-written or repaired transcripts exist; they must not crash the check.
    t = parse_transcript(_transcript(9, "JUST SOME TEXT\n"))
    assert t.chapter == 9
    assert t.pages == []


# -- unambiguous failures --------------------------------------------------


def test_a_hallucinated_cjk_glyph_fails():
    # PP-OCR emitted this from an English bubble; it is never story text.
    t = parse_transcript(_transcript(1, "--- page 1 ---\nI HAVE\u513fTO ADMIT\n"))
    findings = check_transcript(t, glossary=[])

    assert [f.kind for f in findings if f.severity == "fail"] == ["cjk"]


def test_a_fused_run_the_pipeline_would_split_fails():
    # This is the staleness gate: if the shipped rule splits it, a transcript
    # still holding it is out of date, or the pipeline regressed.
    t = parse_transcript(_transcript(1, "--- page 1 ---\nHEHAS YOUR HAIR!\n"))
    findings = check_transcript(t, glossary=[])

    assert "fused" in [f.kind for f in findings if f.severity == "fail"]


def test_a_glossary_name_is_never_reported_as_stale():
    # Regression: chapter 2 was re-OCR'd by the current code and still flagged,
    # because the check resegmented without the glossary and split names the
    # pipeline protects (ASHBERN, ETISTIN, DICATHEN are all in the real one).
    body = "--- page 1 ---\nDICATHEN, the continent,\nand ETISTIN too\n"
    findings = check_transcript(
        parse_transcript(_transcript(1, body)), glossary=["DICATHEN", "ETISTIN"]
    )
    assert "fused" not in [f.kind for f in findings]


def test_the_cost_gate_is_what_keeps_names_out_of_the_staleness_gate():
    # Measured over the real glossary: every name splits at 10.54 or worse
    # (ETISTIN is the cheapest of 22), while fused dialogue tops out at 9.62.
    # So a run that is present but splits *expensively* is a name, and calling
    # it stale would be a false alarm on a transcript the pipeline produced.
    t = parse_transcript(_transcript(1, "--- page 1 ---\nETISTIN and SAPIN\n"))
    assert "fused" not in [f.kind for f in check_transcript(t, [])]

    cheap = parse_transcript(_transcript(1, "--- page 1 ---\nHEHAS YOUR HAIR\n"))
    assert "fused" in [f.kind for f in check_transcript(cheap, [])]


def test_a_glossary_term_is_never_called_stale_even_when_it_splits_cheaply():
    # A term whose parts are all common words would otherwise read as fused
    # dialogue -- DARKLORD costs 9.35, inside the fused band. Listing it in the
    # glossary is precisely how the pipeline is told to keep it whole, so the
    # check must consult the same list rather than deciding for itself.
    body = "--- page 1 ---\nthe DARKLORD awakens\n"
    transcript = parse_transcript(_transcript(1, body))

    assert "fused" in [f.kind for f in check_transcript(transcript, [])]
    assert "fused" not in [f.kind for f in check_transcript(transcript, ["DARKLORD"])]


def test_a_correctly_split_line_does_not_fail():
    t = parse_transcript(_transcript(1, "--- page 1 ---\nhe has YOUR HAIR!\n"))
    assert not [f for f in check_transcript(t, glossary=[]) if f.severity == "fail"]


def test_control_characters_fail():
    t = parse_transcript(_transcript(1, "--- page 1 ---\nART\x00IS\n"))
    assert "nonprintable" in [
        f.kind for f in check_transcript(t, glossary=[]) if f.severity == "fail"
    ]


def test_a_clean_transcript_has_no_failures():
    body = (
        "--- page 1 ---\n"
        "MY MOTHER,ALICE, was a gentle\n"
        "and KIND-hearted person.\n"
        "--- page 2 ---\n"
        "SAPIN, the human kingdom,\n"
    )
    t = parse_transcript(_transcript(1, body))
    assert [f for f in check_transcript(t, glossary=["ALICE", "SAPIN"])
            if f.severity == "fail"] == []


# -- warnings --------------------------------------------------------------


def test_a_glossary_term_appearing_split_warns_but_does_not_fail():
    # MANA became "man a" in the real corpus. It is a smell, not proof: "a man
    # a day" is ordinary English, so this can never gate a run.
    t = parse_transcript(_transcript(1, "--- page 1 ---\nemit man a to the world\n"))
    findings = check_transcript(t, glossary=["MANA"])

    kinds = {(f.kind, f.severity) for f in findings}
    assert ("term_split", "warn") in kinds


def test_an_unknown_token_is_reported_as_noise():
    t = parse_transcript(_transcript(1, "--- page 1 ---\nKWNI and dWHHL\n"))
    kinds = {f.kind for f in check_transcript(t, glossary=[]) if f.severity == "warn"}
    assert "noise" in kinds


def test_a_glossary_name_is_not_reported_as_noise():
    t = parse_transcript(_transcript(1, "--- page 1 ---\nXYRUS ACADEMY\n"))
    findings = check_transcript(t, glossary=["XYRUS"])
    assert not [f for f in findings if f.kind == "noise"]


def test_a_glossary_name_carrying_an_apostrophe_is_recognised_intact():
    # This is the only case where checking the glossary term *before* stripping
    # the apostrophe matters: "O'NEIL" is not "oneil" in any dictionary, so
    # stripping first would lose it. Terms without an apostrophe are found by
    # either check, which is why the earlier probe could not distinguish them.
    t = parse_transcript(_transcript(1, "--- page 1 ---\nO'NEIL\n"))
    assert "noise" not in {f.kind for f in check_transcript(t, ["O'NEIL"])}


def test_a_line_repeated_many_times_warns():
    # An OCR loop prints one block over and over; a real refrain is rarer.
    body = "--- page 1 ---\n" + "SCRATCH\n" * 8
    findings = check_transcript(parse_transcript(_transcript(1, body)), glossary=[])
    assert "repeat" in {f.kind for f in findings if f.severity == "warn"}


def test_credit_and_watermark_text_warns():
    body = "--- page 1 ---\nOriginal Novel and Script: TurtleMe\n"
    findings = check_transcript(parse_transcript(_transcript(1, body)), glossary=[])
    assert "credit" in {f.kind for f in findings if f.severity == "warn"}


def test_a_scrubbed_transcript_reports_no_credit_finding():
    # The validator and the reader share ``keep_credit_mask``. A transcript the
    # reader has already scrubbed must therefore be clean by the same rule --
    # this is the drift that two separate marker lists produced before.
    body = (
        "--- page 1 ---\n"
        "STORY BY\nTURTLEME\nART BY\nFUYUKI23\n"
        "scan-site.test\n"
        "SO THIS IS THE PLACE.\n"
    )
    lines = body.splitlines()[1:]
    scrubbed = keep_credit_mask(lines)
    survivors = [text for text, keep in zip(lines, scrubbed) if keep]
    assert "credit" not in {
        f.kind
        for f in check_transcript(
            parse_transcript("--- page 1 ---\n" + "\n".join(survivors)), glossary=[]
        )
    }


def test_a_bare_name_after_a_role_line_is_reported_as_a_credit():
    # The previous-line shape lives in the shared rule, so the validator sees it
    # too. A per-line marker list could not: "FUYUKI23" contains no marker.
    body = "--- page 1 ---\nART BY\nFUYUKI23\n"
    findings = check_transcript(parse_transcript(_transcript(1, body)), glossary=[])
    assert "credit" in {f.kind for f in findings if f.severity == "warn"}


def test_credit_vocabulary_in_dialogue_is_not_a_credit_finding():
    # "control" and "original" are ordinary English. The shared rule keys on
    # structure, so sentence-shaped lines are never reported.
    body = "--- page 1 ---\nto control MANA is\nNOT VERY\nORIGINAL.\n"
    findings = check_transcript(parse_transcript(_transcript(1, body)), glossary=[])
    assert "credit" not in {f.kind for f in findings if f.severity == "warn"}


# -- page integrity --------------------------------------------------------


def test_a_silent_page_is_not_a_gap():
    # Regression: transcripts only mark up a page when it produced text, so a
    # page with no dialogue leaves no marker. Counting markers against pages on
    # disk reported "7 pages missing" on a chapter that was complete.
    body = "--- page 1 ---\nA\n--- page 3 ---\nC\n"
    findings = check_transcript(
        parse_transcript(_transcript(1, body)), glossary=[], expected_pages=5
    )
    assert not [f for f in findings if f.kind == "page_out_of_range"]


def test_a_page_marker_beyond_the_pages_on_disk_fails():
    # Text attributed to a page that does not exist means the markers and the
    # images disagree -- a real integrity fault.
    body = "--- page 81 ---\nA\n"
    findings = check_transcript(
        parse_transcript(_transcript(1, body)), glossary=[], expected_pages=80
    )
    assert "page_out_of_range" in [
        f.kind for f in findings if f.severity == "fail"
    ]


def test_a_transcript_matching_its_page_count_passes():
    body = "--- page 1 ---\nA\n--- page 2 ---\nB\n"
    findings = check_transcript(
        parse_transcript(_transcript(1, body)), glossary=[], expected_pages=2
    )
    assert not [f for f in findings if f.kind == "page_out_of_range"]


# -- the novel-free quality number ----------------------------------------


def test_suspicious_rate_counts_non_dictionary_tokens():
    t = parse_transcript(
        _transcript(1, "--- page 1 ---\nTHE KING SAID TO HIS SON\n")
    )
    # Every token here is a dictionary word, so nothing is suspicious.
    assert suspicious_rate(t, glossary=[]) == 0.0


def test_suspicious_rate_ignores_glossary_terms():
    t = parse_transcript(_transcript(1, "--- page 1 ---\nXYRUS ART\n"))
    assert suspicious_rate(t, glossary=["XYRUS"]) == 0.0


def test_suspicious_rate_is_a_fraction_of_tokens():
    t = parse_transcript(_transcript(1, "--- page 1 ---\nKWNI ART\n"))
    rate = suspicious_rate(t, glossary=[])
    assert 0.0 < rate < 1.0


def test_suspicious_tokens_are_reported_as_a_count():
    # The rate alone misleads when a change alters how many words survive:
    # scrubbing credits took chapter 183 from 6 suspicious tokens to 4 while its
    # rate *rose* from 3.75% to 4.55%, because the credits were ordinary English
    # and shrank the denominator. The count is the comparable number.
    t = parse_transcript(_transcript(1, "--- page 1 ---\nKWNI ART dWHHL\n"))
    assert sorted(suspicious_tokens(t, glossary=[])) == ["KWNI", "dWHHL"]


def test_suspicious_tokens_honour_the_glossary():
    t = parse_transcript(_transcript(1, "--- page 1 ---\nXYRUS KWNI\n"))
    assert suspicious_tokens(t, glossary=["XYRUS"]) == ["KWNI"]


# -- glossary loading ------------------------------------------------------


def test_glossary_is_read_from_the_tracked_config(tmp_path: Path):
    config = tmp_path / "cfg.json"
    config.write_text('{"glossary": ["ALICE", "SAPIN"]}', encoding="utf-8")
    assert load_glossary(config) == ["ALICE", "SAPIN"]


def test_a_contraction_is_not_reported_as_noise():
    # Regression: the lookup stripped apostrophes before consulting the
    # dictionary, so "they're" became "theyre" -- not a word -- and a sentence
    # of ordinary dialogue was reported as 2 noise tokens. 15 of the 103
    # apostrophe tokens in the real corpus were affected.
    t = parse_transcript(
        _transcript(1, "--- page 1 ---\nI don't know why they're here.\n")
    )
    assert "noise" not in {f.kind for f in check_transcript(t, [])}
    assert suspicious_rate(t, glossary=[]) == 0.0


def test_a_possessive_is_not_reported_as_noise():
    # The other half of the same rule, and the reason the stripped form is kept
    # as a fallback instead of being dropped: the dictionary has "wife" and
    # "wifes" but never "wife's", so this token is only known *after* the
    # apostrophe comes off. Three real tokens depended on it.
    t = parse_transcript(_transcript(1, "--- page 1 ---\nHIS WIFE'S NAME\n"))
    assert "noise" not in {f.kind for f in check_transcript(t, [])}
    assert suspicious_rate(t, glossary=[]) == 0.0


def test_a_missing_glossary_file_yields_an_empty_list(tmp_path: Path):
    # A comic with no config still has to be checkable.
    assert load_glossary(tmp_path / "nope.json") == []


# -- pages on disk vs the manifest ----------------------------------------


def test_a_chapter_short_of_its_listed_page_count_is_reported(tmp_path: Path):
    # The real defect: Chapter 144 lists 81 pages and the site served 80 (HTTP
    # 200, zero bytes). Nothing downstream notices -- the transcript is simply
    # one page shorter -- so the count has to be compared to the manifest.
    import json

    (tmp_path / "c_000147").mkdir()
    for page in range(1, 81):
        (tmp_path / "c_000147" / f"c_000147_{page:03d}.jpg").write_bytes(b"x")
    (tmp_path / "chapters.json").write_text(
        json.dumps(
            {
                "series": "T",
                "site": "",
                "chapters": [
                    {"ordinal": 147, "slug": "s", "label": "Chapter 144",
                     "number": 144.0, "pages": 81},
                ],
            }
        ),
        encoding="utf-8",
    )

    findings = check_pages(tmp_path)

    assert [f.kind for f in findings] == ["page_short"]
    assert findings[0].severity == "fail"
    assert "147" in findings[0].detail


def test_a_chapter_matching_its_listed_count_is_clean(tmp_path: Path):
    import json

    (tmp_path / "c_000001").mkdir()
    for page in range(1, 4):
        (tmp_path / "c_000001" / f"c_000001_{page:03d}.jpg").write_bytes(b"x")
    (tmp_path / "chapters.json").write_text(
        json.dumps(
            {
                "series": "T",
                "site": "",
                "chapters": [
                    {"ordinal": 1, "slug": "s", "label": "Chapter 1",
                     "number": 1.0, "pages": 3},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert check_pages(tmp_path) == []


def test_a_detected_duplicate_is_not_reported_short(tmp_path: Path):
    # A re-upload (``Chapter 144`` and ``Chapter 144.0``) is fingerprinted and
    # deliberately never downloaded, so it has pages in the manifest and no
    # folder. That is the design working, not a hole -- the first version of
    # this check reported it as "80 pages listed, 0 on disk".
    import json

    (tmp_path / "c_000147").mkdir()
    for page in range(1, 81):
        (tmp_path / "c_000147" / f"c_000147_{page:03d}.jpg").write_bytes(b"x")
    (tmp_path / "chapters.json").write_text(
        json.dumps(
            {
                "series": "T",
                "site": "",
                "chapters": [
                    {"ordinal": 147, "slug": "a", "label": "Chapter 144",
                     "number": 144.0, "pages": 81},
                    {"ordinal": 148, "slug": "b", "label": "Chapter 144.0",
                     "number": 144.0, "pages": 80, "duplicate_of": 147},
                ],
            }
        ),
        encoding="utf-8",
    )

    findings = check_pages(tmp_path)

    assert len(findings) == 1
    assert "147" in findings[0].detail
    assert "148" not in findings[0].detail


def test_a_chapter_never_scanned_is_not_reported_as_short(tmp_path: Path):
    # An unscanned chapter has no page count in the manifest, so there is
    # nothing to compare it against. Flagging it would report every chapter of
    # a half-finished series as short.
    import json

    (tmp_path / "chapters.json").write_text(
        json.dumps(
            {
                "series": "T",
                "site": "",
                "chapters": [
                    {"ordinal": 5, "slug": "s", "label": "Chapter 5",
                     "number": 5.0, "pages": 0},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert check_pages(tmp_path) == []


def test_a_scanned_chapter_with_nothing_on_disk_is_a_hole(tmp_path: Path):
    # The complement of the case above, and the reason there is no "no folder"
    # guard: the manifest says the site listed pages, so an empty folder is a
    # real gap. Skipping it would hide exactly the chapters that are missing.
    import json

    (tmp_path / "chapters.json").write_text(
        json.dumps(
            {
                "series": "T",
                "site": "",
                "chapters": [
                    {"ordinal": 7, "slug": "s", "label": "Chapter 7",
                     "number": 7.0, "pages": 12},
                ],
            }
        ),
        encoding="utf-8",
    )

    findings = check_pages(tmp_path)

    assert [f.kind for f in findings] == ["page_short"]
    assert findings[0].count == 12


# -- the CLI --------------------------------------------------------------


def test_the_cli_passes_the_page_count_so_marker_drift_is_caught(
    tmp_path: Path, monkeypatch, capsys
):
    # Otherwise ``page_out_of_range`` is dead code from the command line: the
    # check exists, is unit-tested, and can never fire on a real run.
    import json

    from comic_check import main

    (tmp_path / "novels" / "comic-T").mkdir(parents=True)
    (tmp_path / "novels" / "comic-T" / "chapter_00001.txt").write_text(
        "# T chapter 1\n\n--- page 1 ---\nHELLO\n--- page 9 ---\nBYE\n",
        encoding="utf-8",
    )
    series = tmp_path / "comics" / "T"
    (series / "c_000001").mkdir(parents=True)
    for page in range(1, 4):
        (series / "c_000001" / f"c_000001_{page:03d}.jpg").write_bytes(b"x")
    (series / "chapters.json").write_text(
        json.dumps(
            {
                "series": "T",
                "site": "",
                "chapters": [
                    {"ordinal": 1, "slug": "s", "label": "Chapter 1",
                     "number": 1.0, "pages": 3},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "comic_config.json").write_text(
        '{"glossary": []}', encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    code = main(["-n", "comic-T", "--comics", "comics/T", "--json"])

    document = json.loads(capsys.readouterr().out)
    assert code == 1
    assert "page_out_of_range" in [f["kind"] for f in document["findings"]]


def test_json_output_stays_parseable_when_page_counts_are_checked(
    tmp_path: Path, monkeypatch, capsys
):
    # Regression: the manifest findings were printed as text after the JSON
    # document, so ``--json --comics`` emitted a stream no parser accepts.
    import json

    from comic_check import main

    (tmp_path / "novels" / "comic-T").mkdir(parents=True)
    (tmp_path / "novels" / "comic-T" / "chapter_00001.txt").write_text(
        "# T chapter 1\n\n--- page 1 ---\nHELLO THERE\n", encoding="utf-8"
    )
    series = tmp_path / "comics" / "T"
    (series / "c_000001").mkdir(parents=True)
    for page in range(1, 3):
        (series / "c_000001" / f"c_000001_{page:03d}.jpg").write_bytes(b"x")
    (series / "chapters.json").write_text(
        json.dumps(
            {
                "series": "T",
                "site": "",
                "chapters": [
                    {"ordinal": 1, "slug": "s", "label": "Chapter 1",
                     "number": 1.0, "pages": 5},
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "comic_config.json").write_text(
        '{"glossary": []}', encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    code = main(["-n", "comic-T", "--comics", "comics/T", "--json"])

    document = json.loads(capsys.readouterr().out)
    assert code == 1
    assert [f["kind"] for f in document["findings"]] == ["page_short"]
    assert document["failures"] == 1
